from __future__ import annotations

import asyncio

import fakeredis.aioredis
import pytest
from fastapi.testclient import TestClient

from v_cropper.service.app import create_app
from v_cropper.service.config import Settings

TOKEN = "test-token-long-enough-to-not-warn"
AUTH = {"Authorization": f"Bearer {TOKEN}"}


def _app(**overrides):
    redis = fakeredis.aioredis.FakeRedis()
    # Settings reads env at class-definition time, so build it explicitly here.
    fields = dict(
        redis_url="redis://unused",
        api_token=TOKEN,
        job_ttl_sec=60,
        event_ttl_sec=60,
        max_duration_sec=60,
        max_output_bytes=1_000_000,
        allowed_hosts=("objects.example.test",),
        allow_private_hosts=True,
        work_root="/tmp/v-cropper-tests",
        http_timeout_sec=10,
    )
    fields.update(overrides)
    app = create_app(Settings(**fields), redis)

    async def idle_worker():
        await asyncio.Event().wait()

    app.state.worker.run = idle_worker
    return app


@pytest.fixture(autouse=True)
def _ffmpeg_present(monkeypatch):
    """Readiness runs the ffmpeg check first, so absent tooling masks every other branch.

    These tests are about the readiness logic, not about what happens to be installed on
    the machine running them. `test_readiness_requires_ffmpeg` overrides this to assert the
    missing-ffmpeg branch specifically.
    """
    monkeypatch.setattr("v_cropper.service.app.shutil.which", lambda name: f"/usr/bin/{name}")


def test_operations_and_openapi_endpoints(monkeypatch):
    monkeypatch.setenv("VCROPPER_API_KEY", "test-key")
    with TestClient(_app()) as client:
        assert client.get("/healthz").json() == {"status": "ok"}
        assert client.get("/readyz").json() == {"status": "ready"}
        assert "version" in client.get("/version").json()
        assert client.get("/docs").status_code == 200
        assert client.get("/openapi.json").status_code == 200


def test_readiness_requires_model_credentials(monkeypatch):
    monkeypatch.setenv("VCROPPER_PROVIDER", "openai")
    monkeypatch.delenv("VCROPPER_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with TestClient(_app()) as client:
        response = client.get("/readyz")
        assert response.status_code == 503
        assert response.json()["detail"] == "VLM API key is not configured"


def test_bedrock_readiness_requires_region_not_api_key(monkeypatch):
    monkeypatch.setenv("VCROPPER_PROVIDER", "bedrock")
    monkeypatch.delenv("VCROPPER_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("BEDROCK_REGION", raising=False)
    monkeypatch.delenv("AWS_REGION", raising=False)
    with TestClient(_app()) as client:
        response = client.get("/readyz")
        assert response.status_code == 503
        assert response.json()["detail"] == "AWS Bedrock region is not configured"
        monkeypatch.setenv("AWS_REGION", "us-east-1")
        assert client.get("/readyz").json() == {"status": "ready"}


def test_job_contract_never_returns_signed_urls():
    payload = {
        "source_url": "https://objects.example.test/input.mp4?X-Amz-Signature=source-secret",
        "destination_url": "https://objects.example.test/output.mp4?X-Amz-Signature=destination-secret",
        "in_s": 1,
        "out_s": 3,
        "sport": "football",
    }
    with TestClient(_app()) as client:
        response = client.post("/v1/jobs", json=payload, headers=AUTH)
        assert response.status_code == 202
        job_id = response.json()["job_id"]
        status = client.get(f"/v1/jobs/{job_id}", headers=AUTH)
        assert status.status_code == 200
        serialized = status.text
        assert "source_url" not in serialized
        assert "destination_url" not in serialized
        assert "Signature" not in serialized
        cancelled = client.delete(f"/v1/jobs/{job_id}", headers=AUTH)
        assert cancelled.status_code == 202
        assert cancelled.json()["status"] == "cancelled"


class SSEDriver:
    """Calls the ASGI app directly to observe an unbounded SSE stream.

    Neither TestClient nor httpx's ASGITransport works here: the event stream only ends at a
    terminal event, and both wait for the response to complete before handing anything back.
    Driving the app directly also makes `http.disconnect` reachable, which is the only way the
    stream's hang-up path can be exercised in process.
    """

    def __init__(self, app, path, headers=AUTH):
        self._disconnected = False
        self.chunks: list[str] = []
        self.status: int | None = None
        scope = {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.1"},
            "http_version": "1.1",
            "method": "GET",
            "scheme": "http",
            "path": path,
            "raw_path": path.encode(),
            "query_string": b"",
            "root_path": "",
            "headers": [(b"host", b"test")]
            + [(k.lower().encode(), v.encode()) for k, v in headers.items()],
            "client": ("127.0.0.1", 54321),
            "server": ("test", 80),
        }
        self.task = asyncio.create_task(app(scope, self._receive, self._send))

    async def _receive(self):
        # Returning without awaiting matters: Starlette polls this inside an already-cancelled
        # scope, so a message is only seen if it is available synchronously.
        if self._disconnected:
            return {"type": "http.disconnect"}
        await asyncio.Event().wait()

    async def _send(self, message):
        if message["type"] == "http.response.start":
            self.status = message["status"]
        elif message["type"] == "http.response.body":
            self.chunks.append(message["body"].decode())

    @property
    def body(self) -> str:
        return "".join(self.chunks)

    def disconnect(self) -> None:
        self._disconnected = True

    async def read_until(self, predicate, *, stop: bool = True, timeout: float = 10) -> str:
        """Wait for the accumulated body to satisfy `predicate`, then optionally stop the app."""
        async def poll():
            while not predicate(self.body):
                if self.task.done():
                    self.task.result()
                    raise AssertionError(f"stream ended early: {self.body!r}")
                await asyncio.sleep(0.05)

        try:
            await asyncio.wait_for(poll(), timeout=timeout)
        finally:
            if stop:
                self.task.cancel()
                await asyncio.gather(self.task, return_exceptions=True)
        return self.body


class TestEventStream:
    """SSE is how a client follows a job without polling."""

    PAYLOAD = {
        "source_url": "https://objects.example.test/input.mp4",
        "destination_url": "https://objects.example.test/output.mp4",
    }

    def test_stream_replays_progress_then_closes_on_a_terminal_event(self):
        app = _app()
        with TestClient(app) as client:
            job_id = client.post("/v1/jobs", json=self.PAYLOAD, headers=AUTH).json()["job_id"]

            store = app.state.store
            asyncio.run(store.update(job_id, status="running", phase="cropping", progress_pct=50))
            asyncio.run(store.update(job_id, status="succeeded", phase="complete", progress_pct=100))

            with client.stream("GET", f"/v1/jobs/{job_id}/events", headers=AUTH) as response:
                assert response.status_code == 200
                assert response.headers["cache-control"] == "no-cache"
                body = "".join(response.iter_text())

        assert "event: progress" in body
        assert "event: terminal" in body
        assert '"status":"succeeded"' in body.replace(" ", "")
        # The stream must end itself at a terminal event rather than hanging on the client.
        assert body.rstrip().endswith("}")

    def test_stream_never_exposes_signed_urls(self):
        app = _app()
        payload = dict(self.PAYLOAD,
                       source_url="https://objects.example.test/in.mp4?X-Amz-Signature=leak-me")
        with TestClient(app) as client:
            job_id = client.post("/v1/jobs", json=payload, headers=AUTH).json()["job_id"]
            asyncio.run(app.state.store.update(job_id, status="failed", phase="failed"))
            with client.stream("GET", f"/v1/jobs/{job_id}/events", headers=AUTH) as response:
                body = "".join(response.iter_text())
        assert "leak-me" not in body
        assert "Signature" not in body

    def test_stream_for_unknown_job_is_404(self):
        with TestClient(_app()) as client:
            assert client.get("/v1/jobs/nope/events", headers=AUTH).status_code == 404

    @pytest.mark.asyncio
    async def test_stream_emits_keepalives_while_a_job_is_still_running(self):
        """Without these an idle proxy closes the connection mid-job."""
        app = _app()
        with TestClient(app) as client:
            job_id = client.post("/v1/jobs", json=self.PAYLOAD, headers=AUTH).json()["job_id"]

        driver = SSEDriver(app, f"/v1/jobs/{job_id}/events")
        body = await driver.read_until(lambda text: text.count(": keepalive") >= 2)

        assert "event: progress" in body       # the queued event
        assert body.count(": keepalive") >= 2  # then it idles rather than closing
        assert "event: terminal" not in body

    @pytest.mark.asyncio
    async def test_stream_stops_when_the_client_hangs_up(self):
        """A stream that ignored disconnects would poll Redis forever per abandoned client."""
        app = _app()
        with TestClient(app) as client:
            job_id = client.post("/v1/jobs", json=self.PAYLOAD, headers=AUTH).json()["job_id"]

        driver = SSEDriver(app, f"/v1/jobs/{job_id}/events")
        await driver.read_until(lambda text: ": keepalive" in text, stop=False)
        driver.disconnect()

        # The generator returns on the next poll rather than being cancelled by the test.
        await asyncio.wait_for(driver.task, timeout=10)
        assert "event: terminal" not in driver.body


def test_rejects_a_clip_longer_than_the_duration_limit():
    """The cap bounds worst-case cost and wall time per job."""
    with TestClient(_app(max_duration_sec=60)) as client:
        response = client.post(
            "/v1/jobs",
            json={
                "source_url": "https://objects.example.test/input.mp4",
                "destination_url": "https://objects.example.test/output.mp4",
                "in_s": 10,
                "out_s": 500,
            },
            headers=AUTH,
        )
        assert response.status_code == 422
        assert "duration limit" in response.json()["detail"]

        # A clip inside the window is still accepted.
        accepted = client.post(
            "/v1/jobs",
            json={
                "source_url": "https://objects.example.test/input.mp4",
                "destination_url": "https://objects.example.test/output.mp4",
                "in_s": 10,
                "out_s": 60,
            },
            headers=AUTH,
        )
        assert accepted.status_code == 202


def test_owned_redis_connection_is_closed_on_shutdown(monkeypatch):
    """create_app opens the client when one is not injected, so it must close it."""
    fake = fakeredis.aioredis.FakeRedis()
    closed = False
    original = fake.aclose

    async def track_close():
        nonlocal closed
        closed = True
        await original()

    monkeypatch.setattr(fake, "aclose", track_close)
    monkeypatch.setattr("v_cropper.service.app.Redis.from_url", lambda url: fake)

    app = create_app(Settings(api_token=TOKEN, allowed_hosts=("objects.example.test",)))

    async def idle_worker():
        await asyncio.Event().wait()

    app.state.worker.run = idle_worker
    with TestClient(app) as client:
        assert client.get("/healthz").status_code == 200
    assert closed


def test_cancel_is_idempotent_and_unknown_job_is_404():
    payload = {
        "source_url": "https://objects.example.test/input.mp4",
        "destination_url": "https://objects.example.test/output.mp4",
    }
    with TestClient(_app()) as client:
        job_id = client.post("/v1/jobs", json=payload, headers=AUTH).json()["job_id"]
        assert client.delete(f"/v1/jobs/{job_id}", headers=AUTH).status_code == 202
        # A second cancel must not error; clients retry.
        assert client.delete(f"/v1/jobs/{job_id}", headers=AUTH).status_code == 202
        assert client.delete("/v1/jobs/unknown", headers=AUTH).status_code == 404
        assert client.get("/v1/jobs/unknown", headers=AUTH).status_code == 404


def test_readiness_requires_ffmpeg(monkeypatch):
    monkeypatch.setenv("VCROPPER_API_KEY", "test-key")
    monkeypatch.setattr("v_cropper.service.app.shutil.which", lambda name: None)
    with TestClient(_app()) as client:
        response = client.get("/readyz")
        assert response.status_code == 503
        assert "ffmpeg" in response.json()["detail"]


def test_readiness_rejects_an_unknown_provider(monkeypatch):
    monkeypatch.setenv("VCROPPER_PROVIDER", "not-a-provider")
    with TestClient(_app()) as client:
        response = client.get("/readyz")
        assert response.status_code == 503
        assert "openai or bedrock" in response.json()["detail"]


def test_readiness_reports_unavailable_redis(monkeypatch):
    settings = Settings(redis_url="redis://unused", api_token=TOKEN, allowed_hosts=("objects.example.test",))

    class DeadRedis(fakeredis.aioredis.FakeRedis):
        async def ping(self):
            raise ConnectionError("redis is down")

    app = create_app(settings, DeadRedis())

    async def idle_worker():
        await asyncio.Event().wait()

    app.state.worker.run = idle_worker
    with TestClient(app) as client:
        response = client.get("/readyz")
        assert response.status_code == 503
        assert response.json()["detail"] == "Redis unavailable"


def test_rejects_disallowed_host_and_invalid_trim():
    base = {
        "source_url": "https://attacker.test/input.mp4",
        "destination_url": "https://objects.example.test/output.mp4",
    }
    with TestClient(_app()) as client:
        assert client.post("/v1/jobs", json=base, headers=AUTH).status_code == 422
        base["source_url"] = "https://objects.example.test/input.mp4"
        base.update({"in_s": 5, "out_s": 2})
        assert client.post("/v1/jobs", json=base, headers=AUTH).status_code == 422
