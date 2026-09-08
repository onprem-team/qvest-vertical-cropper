"""Adversarial probes against the deployed HTTP surface.

These are written from the attacker's side rather than the feature's: each one is a way the
service could be turned into a proxy into the VPC, made to hand back a signed URL, or made to
consume unbounded resources. Grouped by what an attacker is trying to achieve.
"""
from __future__ import annotations

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import fakeredis.aioredis
import pytest
from fastapi.testclient import TestClient

from v_cropper.service.app import create_app
from v_cropper.service.config import Settings
from v_cropper.service.models import CropJobRequest
from v_cropper.service.security import (
    UnsafeURLError,
    redact_url,
    validate_redirect_chain,
    validate_transport_url,
)
from v_cropper.service.store import JobStore
from v_cropper.service.worker import JobWorker

TOKEN = "adversarial-token-long-enough-not-to-warn"
AUTH = {"Authorization": f"Bearer {TOKEN}"}
ALLOWED = ("objects.example.test",)
SOURCE = "https://objects.example.test/in.mp4?X-Amz-Signature=source-secret-value"
DESTINATION = "https://objects.example.test/out.mp4?X-Amz-Signature=dest-secret-value"


def _settings(**overrides) -> Settings:
    fields = dict(
        redis_url="redis://unused",
        api_token=TOKEN,
        job_ttl_sec=60,
        event_ttl_sec=60,
        max_duration_sec=60,
        max_output_bytes=1_000_000,
        allowed_hosts=ALLOWED,
        allow_private_hosts=True,
        work_root="/tmp/v-cropper-adversarial",
        http_timeout_sec=5,
    )
    fields.update(overrides)
    return Settings(**fields)


def _app(**overrides):
    app = create_app(_settings(**overrides), fakeredis.aioredis.FakeRedis())

    async def idle_worker():
        await asyncio.Event().wait()

    app.state.worker.run = idle_worker
    return app


# --------------------------------------------------------------------------- #
# Turning the service into a proxy into the VPC
# --------------------------------------------------------------------------- #
class TestSSRF:
    @pytest.mark.parametrize("url", [
        "file:///etc/passwd",
        "gopher://objects.example.test/_test",
        "ftp://objects.example.test/x.mp4",
        "data:video/mp4;base64,AAAA",
        "//objects.example.test/x.mp4",
        "http:///x.mp4",                                   # no host
        "https://user:pass@objects.example.test/x.mp4",    # smuggled credentials
        "https://evil.test/x.mp4",                         # off-allowlist
        "https://objects.example.test.evil.test/x.mp4",    # suffix confusion
        "https://evilobjects.example.test/x.mp4",          # prefix confusion
    ])
    def test_submitting_a_hostile_url_is_refused(self, url):
        with TestClient(_app()) as client:
            response = client.post(
                "/v1/jobs", json={"source_url": url, "destination_url": DESTINATION},
                headers=AUTH,
            )
            assert response.status_code == 422, url

    def test_a_wildcard_entry_does_not_match_a_lookalike_registrable_domain(self):
        """`*.example.test` must not admit `notexample.test` or the bare apex."""
        for host in ("https://evil-example.test/x", "https://example.test.evil.test/x"):
            with pytest.raises(UnsafeURLError):
                validate_transport_url(
                    host, allowed_hosts=("*.example.test",), resolve_dns=False)

        assert validate_transport_url(
            "https://objects.example.test/x", allowed_hosts=("*.example.test",), resolve_dns=False)

    def test_the_error_returned_never_reflects_the_submitted_url(self):
        """A reflected URL turns the validator into an oracle for internal hostnames."""
        with TestClient(_app()) as client:
            response = client.post(
                "/v1/jobs",
                json={"source_url": "https://internal-secret-host.evil.test/x.mp4",
                      "destination_url": DESTINATION},
                headers=AUTH,
            )
            assert "internal-secret-host" not in response.text


class TestRedirectSSRF:
    """ffmpeg fetches the source itself and follows redirects.

    Found by probing the deployed path rather than reading it: an allowlisted host that
    answers 302 reached an arbitrary address, so the allowlist and the private-address check
    were both bypassable by anyone able to submit a job.
    """

    @staticmethod
    def _serve(handler_factory):
        server = HTTPServer(("127.0.0.1", 0), handler_factory)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        return server

    @pytest.fixture
    def redirector(self):
        target = {"location": None}

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(302)
                self.send_header("Location", target["location"])
                self.end_headers()

            def log_message(self, *a):
                pass

        server = self._serve(Handler)
        yield server, target
        server.shutdown()

    @pytest.mark.asyncio
    async def test_a_redirect_off_the_allowlist_is_refused(self, redirector):
        server, target = redirector
        host, port = server.server_address[0], server.server_address[1]
        target["location"] = "http://169.254.169.254/latest/meta-data/"

        with pytest.raises(UnsafeURLError, match="not allowlisted"):
            await validate_redirect_chain(
                f"http://{host}:{port}/source.mp4",
                allowed_hosts=(host,), allow_private=True,
            )

    @pytest.mark.asyncio
    async def test_a_redirect_to_a_private_address_is_refused_even_when_allowlisted(
        self, redirector, monkeypatch
    ):
        """An attacker who controls DNS for an allowlisted name cannot point it inward."""
        server, target = redirector
        host, port = server.server_address[0], server.server_address[1]
        target["location"] = f"http://metadata.example.test:{port}/latest/"

        with pytest.raises(UnsafeURLError, match="private or non-routable"):
            await validate_redirect_chain(
                f"http://{host}:{port}/source.mp4",
                allowed_hosts=(host, "metadata.example.test"),
                allow_private=False,
            )

    @pytest.mark.asyncio
    async def test_a_redirect_loop_terminates(self, redirector):
        server, target = redirector
        host, port = server.server_address[0], server.server_address[1]
        target["location"] = f"http://{host}:{port}/source.mp4"

        with pytest.raises(UnsafeURLError, match="redirect limit"):
            await validate_redirect_chain(
                f"http://{host}:{port}/source.mp4", allowed_hosts=(host,), allow_private=True)

    @pytest.mark.asyncio
    async def test_a_redirect_without_a_location_is_refused(self):
        """Ambiguous to us but not necessarily to ffmpeg, so it must not be waved through."""
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(302)
                self.send_header("Content-Length", "0")
                self.end_headers()

            def log_message(self, *a):
                pass

        server = self._serve(Handler)
        try:
            host, port = server.server_address[0], server.server_address[1]
            with pytest.raises(UnsafeURLError, match="without a location"):
                await validate_redirect_chain(
                    f"http://{host}:{port}/source.mp4",
                    allowed_hosts=(host,), allow_private=True,
                )
        finally:
            server.shutdown()

    @pytest.mark.asyncio
    async def test_an_unreachable_source_is_refused_without_echoing_the_url(self):
        with pytest.raises(UnsafeURLError, match="could not be reached") as exc_info:
            await validate_redirect_chain(
                "http://127.0.0.1:1/source.mp4?X-Amz-Signature=secret",
                allowed_hosts=("127.0.0.1",), allow_private=True, timeout_sec=2,
            )
        assert "secret" not in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_a_direct_response_is_accepted(self):
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.send_header("Content-Length", "1")
                self.end_headers()
                self.wfile.write(b"x")

            def log_message(self, *a):
                pass

        server = self._serve(Handler)
        try:
            host, port = server.server_address[0], server.server_address[1]
            url = f"http://{host}:{port}/source.mp4"
            assert await validate_redirect_chain(
                url, allowed_hosts=(host,), allow_private=True) == url
        finally:
            server.shutdown()

    @pytest.mark.asyncio
    async def test_the_worker_fails_the_job_rather_than_fetching_an_unsafe_source(self):
        """The check has to run in the worker: that is where the URL reaches ffmpeg."""
        store = JobStore(fakeredis.aioredis.FakeRedis(), ttl_sec=60)
        worker = JobWorker(store, _settings(allow_private_hosts=False))
        await store.create("job1", CropJobRequest(
            source_url=SOURCE, destination_url=DESTINATION, in_s=0, out_s=5))
        await store.dequeue()

        await worker.run_job("job1")

        state = await store.get("job1")
        assert state["status"] == "failed"
        assert state["error"]["code"] == "unsafe_source_url"
        assert "source-secret-value" not in json.dumps(state)

    @pytest.mark.asyncio
    async def test_the_upload_transport_does_not_follow_redirects(self):
        """The output PUT goes through httpx, which does not follow redirects by default.

        That is the only reason the destination URL needs no equivalent chain walk, so it is
        asserted rather than assumed.
        """
        import httpx

        async with httpx.AsyncClient() as probe:
            assert probe.follow_redirects is False

        redirected = []

        async def handler(request: httpx.Request) -> httpx.Response:
            redirected.append(str(request.url))
            return httpx.Response(302, headers={"Location": "http://169.254.169.254/"})

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            response = await client.put("https://objects.example.test/out.mp4", content=b"x")

        assert response.status_code == 302
        assert redirected == ["https://objects.example.test/out.mp4"]


# --------------------------------------------------------------------------- #
# Getting in without a credential
# --------------------------------------------------------------------------- #
class TestAuth:
    @pytest.mark.parametrize("header", [
        {},
        {"Authorization": ""},
        {"Authorization": TOKEN},                       # no scheme
        {"Authorization": f"Basic {TOKEN}"},
        {"Authorization": f"Bearer {TOKEN}x"},          # suffix
        {"Authorization": f"Bearer x{TOKEN}"},          # prefix
        {"Authorization": f"Bearer {TOKEN[:-1]}"},      # truncated
        {"Authorization": f"Bearer {TOKEN.upper()}"},   # case flipped
        {"Authorization": "Bearer "},
        {"Authorization": f"Bearer {TOKEN} extra"},
    ])
    def test_credential_variations_are_all_rejected(self, header):
        with TestClient(_app()) as client:
            response = client.post(
                "/v1/jobs",
                json={"source_url": SOURCE, "destination_url": DESTINATION},
                headers=header,
            )
            assert response.status_code == 401, header

    def test_surrounding_whitespace_in_the_credential_is_tolerated(self):
        """RFC 7230 treats optional whitespace around a header value as insignificant.

        The token still has to match exactly; only the padding is ignored.
        """
        with TestClient(_app()) as client:
            response = client.post(
                "/v1/jobs",
                json={"source_url": SOURCE, "destination_url": DESTINATION},
                headers={"Authorization": f"Bearer  {TOKEN}  "},
            )
            assert response.status_code == 202

    def test_the_scheme_is_matched_case_insensitively(self):
        """RFC 7235 makes the scheme case-insensitive; rejecting `bearer` breaks clients."""
        with TestClient(_app()) as client:
            response = client.post(
                "/v1/jobs",
                json={"source_url": SOURCE, "destination_url": DESTINATION},
                headers={"Authorization": f"bEaReR {TOKEN}"},
            )
            assert response.status_code == 202

    def test_a_rejection_never_reveals_the_expected_token(self):
        with TestClient(_app()) as client:
            response = client.post(
                "/v1/jobs",
                json={"source_url": SOURCE, "destination_url": DESTINATION},
                headers={"Authorization": "Bearer wrong"},
            )
            assert TOKEN not in response.text
            assert TOKEN not in str(response.headers)

    def test_auth_is_checked_before_the_body_is_validated(self):
        """Otherwise the validation error becomes an unauthenticated oracle."""
        with TestClient(_app()) as client:
            response = client.post("/v1/jobs", json={"nonsense": True})
            assert response.status_code == 401


# --------------------------------------------------------------------------- #
# Extracting the signed URLs the service was trusted with
# --------------------------------------------------------------------------- #
class TestSecretLeakage:
    @pytest.mark.asyncio
    async def test_no_failure_path_puts_a_signature_into_job_state(self):
        store = JobStore(fakeredis.aioredis.FakeRedis(), ttl_sec=60)
        worker = JobWorker(store, _settings())
        await store.create("job1", CropJobRequest(
            source_url=SOURCE, destination_url=DESTINATION, in_s=0, out_s=5))
        await store.dequeue()

        # No such host, so this fails deep inside the fetch with the URL in the exception.
        await worker.run_job("job1")

        dumped = json.dumps(await store.get("job1")) + json.dumps(await store.events("job1"))
        for secret in ("source-secret-value", "dest-secret-value", "X-Amz-Signature"):
            assert secret not in dumped

    def test_the_api_never_returns_the_urls_it_was_given(self):
        with TestClient(_app()) as client:
            job_id = client.post(
                "/v1/jobs", json={"source_url": SOURCE, "destination_url": DESTINATION},
                headers=AUTH,
            ).json()["job_id"]
            body = client.get(f"/v1/jobs/{job_id}", headers=AUTH).text
            assert "source-secret-value" not in body
            assert "X-Amz-Signature" not in body

    @pytest.mark.parametrize("url,expected", [
        ("https://u:p@host.test/a/b?sig=x#frag", "https://host.test/a/b"),
        ("https://host.test:8443/a?sig=x", "https://host.test:8443/a"),
        ("https://host.test/a", "https://host.test/a"),
    ])
    def test_redaction_drops_credentials_query_and_fragment(self, url, expected):
        assert redact_url(url) == expected

    def test_the_openapi_schema_does_not_advertise_a_token(self):
        with TestClient(_app()) as client:
            assert TOKEN not in json.dumps(client.get("/openapi.json").json())


# --------------------------------------------------------------------------- #
# Making the service do more work than it agreed to
# --------------------------------------------------------------------------- #
class TestResourceExhaustion:
    def test_a_trim_window_beyond_the_limit_is_refused(self):
        with TestClient(_app(max_duration_sec=30)) as client:
            response = client.post(
                "/v1/jobs",
                json={"source_url": SOURCE, "destination_url": DESTINATION,
                      "in_s": 0, "out_s": 100_000},
                headers=AUTH,
            )
            assert response.status_code == 422

    @pytest.mark.parametrize("field,value", [
        ("concurrency", 100_000),
        ("concurrency", 0),
        ("send_width", 100_000),
        ("send_width", 0),
        ("sample_fps", 0),
        ("sample_fps", 10_000),
        ("scoreboard_sample_count", 100_000),
        ("spring_k", -1),
        ("in_s", -5),
        ("out_s", -1),
    ])
    def test_out_of_range_tuning_knobs_are_refused(self, field, value):
        """Each of these is a multiplier on VLM spend or CPU time."""
        with TestClient(_app()) as client:
            response = client.post(
                "/v1/jobs",
                json={"source_url": SOURCE, "destination_url": DESTINATION, field: value},
                headers=AUTH,
            )
            assert response.status_code == 422, f"{field}={value}"

    def test_an_inverted_trim_window_is_refused(self):
        with TestClient(_app()) as client:
            response = client.post(
                "/v1/jobs",
                json={"source_url": SOURCE, "destination_url": DESTINATION,
                      "in_s": 30, "out_s": 10},
                headers=AUTH,
            )
            assert response.status_code == 422

    def test_an_oversized_prompt_is_refused(self):
        """The prompt is sent to the VLM on every keyframe."""
        with TestClient(_app()) as client:
            response = client.post(
                "/v1/jobs",
                json={"source_url": SOURCE, "destination_url": DESTINATION,
                      "prompt": "x" * 100_000},
                headers=AUTH,
            )
            assert response.status_code == 422

    def test_unknown_fields_do_not_reach_the_pipeline(self):
        with TestClient(_app()) as client:
            response = client.post(
                "/v1/jobs",
                json={"source_url": SOURCE, "destination_url": DESTINATION,
                      "work_root": "/etc", "api_token": "x", "__proto__": {"x": 1}},
                headers=AUTH,
            )
            assert response.status_code in (202, 422)
            if response.status_code == 202:
                job_id = response.json()["job_id"]
                view = client.get(f"/v1/jobs/{job_id}", headers=AUTH).json()
                assert "work_root" not in json.dumps(view)


# --------------------------------------------------------------------------- #
# Malformed input
# --------------------------------------------------------------------------- #
class TestMalformedInput:
    @pytest.mark.parametrize("body", [
        b"",
        b"not json",
        b"[]",
        b"null",
        b'{"source_url": null, "destination_url": null}',
        b'{"source_url": 12345, "destination_url": true}',
        b'{"source_url": {"a": 1}, "destination_url": []}',
        b'{"source_url": "' + b"x" * 50_000 + b'"}',
    ])
    def test_a_malformed_body_is_a_4xx_never_a_500(self, body):
        with TestClient(_app()) as client:
            response = client.request(
                "POST", "/v1/jobs", content=body,
                headers={**AUTH, "Content-Type": "application/json"},
            )
            assert 400 <= response.status_code < 500, (response.status_code, body[:60])

    @pytest.mark.parametrize("job_id", [
        "../../etc/passwd", "..%2f..%2fetc", "a" * 5000, "with space", "*",
    ])
    def test_a_hostile_job_id_is_not_found_rather_than_an_error(self, job_id):
        """The id is interpolated into Redis keys, so it must not escape its namespace."""
        with TestClient(_app()) as client:
            response = client.get(f"/v1/jobs/{job_id}", headers=AUTH)
            assert response.status_code in (404, 422), job_id

    @pytest.mark.parametrize("sport", ["curling", "\x00\x01\x02", "x" * 500])
    def test_an_unknown_sport_is_accepted_at_submit_and_fails_in_the_worker(self, sport):
        """Known divergence from the CLI, which rejects an unknown --sport up front.

        Not a security issue: the job fails cleanly in the worker when the prompt cannot be
        resolved, with the generic error message and no leak. It is recorded here so the
        looser service-side validation is visible rather than merely untested.
        """
        with TestClient(_app()) as client:
            response = client.post(
                "/v1/jobs",
                json={"source_url": SOURCE, "destination_url": DESTINATION, "sport": sport},
                headers=AUTH,
            )
            assert response.status_code == 202

    def test_a_wildcard_job_id_cannot_match_another_job(self):
        with TestClient(_app()) as client:
            real = client.post(
                "/v1/jobs", json={"source_url": SOURCE, "destination_url": DESTINATION},
                headers=AUTH,
            ).json()["job_id"]
            assert client.get("/v1/jobs/*", headers=AUTH).status_code == 404
            assert client.get(f"/v1/jobs/{real}", headers=AUTH).status_code == 200


# --------------------------------------------------------------------------- #
# Races around cancellation and restart
# --------------------------------------------------------------------------- #
class TestRaces:
    @pytest.mark.asyncio
    async def test_cancelling_twice_concurrently_is_safe(self):
        store = JobStore(fakeredis.aioredis.FakeRedis(), ttl_sec=60)
        await store.create("job1", CropJobRequest(
            source_url=SOURCE, destination_url=DESTINATION))

        results = await asyncio.gather(*(store.cancel("job1") for _ in range(8)))
        assert all(results)
        assert (await store.get("job1"))["status"] == "cancelled"

    @pytest.mark.asyncio
    async def test_a_job_cancelled_before_the_worker_starts_is_never_processed(self):
        """The queue entry outlives the cancel, so the worker has to re-check."""
        store = JobStore(fakeredis.aioredis.FakeRedis(), ttl_sec=60)
        worker = JobWorker(store, _settings())
        await store.create("job1", CropJobRequest(
            source_url=SOURCE, destination_url=DESTINATION))
        await store.cancel("job1")
        job_id = await store.dequeue()

        await worker.run_job(job_id)

        assert (await store.get("job1"))["status"] == "cancelled"

    @pytest.mark.asyncio
    async def test_a_job_whose_state_expired_is_a_no_op(self):
        """Redis TTLs can drop state while the id is still queued after a restart."""
        redis = fakeredis.aioredis.FakeRedis()
        store = JobStore(redis, ttl_sec=60)
        worker = JobWorker(store, _settings())
        await store.create("job1", CropJobRequest(
            source_url=SOURCE, destination_url=DESTINATION))
        await store.dequeue()
        await redis.delete(store.state_key("job1"), store.request_key("job1"))

        await worker.run_job("job1")  # must not raise

        assert await store.get("job1") is None

    @pytest.mark.asyncio
    async def test_stopping_the_worker_ends_the_run_loop(self):
        """A worker that ignored stop() would block container shutdown until the timeout."""
        store = JobStore(fakeredis.aioredis.FakeRedis(), ttl_sec=60)
        worker = JobWorker(store, _settings())
        task = asyncio.create_task(worker.run())
        await asyncio.sleep(0.1)
        worker.stop()
        await asyncio.wait_for(task, timeout=5)


# --------------------------------------------------------------------------- #
# Issued API keys: forgery, replay, privilege escalation, leakage
# --------------------------------------------------------------------------- #
class TestIssuedKeys:
    def _mint(self, client, name="client"):
        response = client.post("/v1/admin/keys", json={"name": name}, headers=AUTH)
        assert response.status_code == 201
        return response.json()

    def test_a_forged_secret_with_a_real_key_id_is_rejected(self):
        with TestClient(_app()) as client:
            created = self._mint(client)
            forged = f"vcrop_{created['key_id']}_not-the-real-secret"
            response = client.post(
                "/v1/jobs",
                json={"source_url": SOURCE, "destination_url": DESTINATION},
                headers={"Authorization": f"Bearer {forged}"},
            )
            assert response.status_code == 401
            assert created["token"] not in response.text

    def test_a_revoked_key_cannot_be_replayed(self):
        with TestClient(_app()) as client:
            created = self._mint(client)
            headers = {"Authorization": f"Bearer {created['token']}"}
            assert client.delete(f"/v1/admin/keys/{created['key_id']}", headers=AUTH).status_code == 204
            response = client.post(
                "/v1/jobs",
                json={"source_url": SOURCE, "destination_url": DESTINATION},
                headers=headers,
            )
            assert response.status_code == 401

    def test_a_minted_key_cannot_mint_or_revoke(self):
        with TestClient(_app()) as client:
            created = self._mint(client, "attacker")
            headers = {"Authorization": f"Bearer {created['token']}"}
            assert client.post("/v1/admin/keys", json={"name": "escalated"}, headers=headers).status_code == 403
            assert client.delete(f"/v1/admin/keys/{created['key_id']}", headers=headers).status_code == 403
            # The key itself is still valid for ordinary work.
            assert client.post(
                "/v1/jobs",
                json={"source_url": SOURCE, "destination_url": DESTINATION},
                headers=headers,
            ).status_code == 202

    @pytest.mark.parametrize("key_id", [
        "../jobs",
        "v-cropper:jobs",
        "bootstrap",
        "../../etc/passwd",
        "0123456789abcdef/../x",
        "*",
    ])
    def test_a_hostile_key_id_cannot_escape_the_redis_namespace(self, key_id):
        with TestClient(_app()) as client:
            response = client.delete(f"/v1/admin/keys/{key_id}", headers=AUTH)
            assert response.status_code in (400, 404, 422), key_id

    def test_key_material_never_appears_in_list_or_error_bodies(self):
        with TestClient(_app()) as client:
            created = self._mint(client)
            token = created["token"]
            listed = client.get("/v1/admin/keys", headers=AUTH)
            denied = client.post("/v1/jobs", json={"source_url": SOURCE, "destination_url": DESTINATION})
            forged = client.post(
                "/v1/jobs",
                json={"source_url": SOURCE, "destination_url": DESTINATION},
                headers={"Authorization": f"Bearer {token}x"},
            )
            for response in (listed, denied, forged):
                assert token not in response.text
                assert "hash" not in response.text or response.status_code != 200
            assert token not in listed.text
            assert created["key_id"] in listed.text
