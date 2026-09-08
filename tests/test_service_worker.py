from __future__ import annotations

import asyncio
import time

import fakeredis.aioredis
import pytest

from v_cropper.pipeline import PipelineCancelled, PipelineResult
from v_cropper.service.config import Settings
from v_cropper.service.models import CropJobRequest
from v_cropper.service.store import JobStore
from v_cropper.service.worker import JobWorker, _options


@pytest.fixture(autouse=True)
def _allow_test_urls(monkeypatch):
    """Skip the source-URL redirect preflight, which would try to reach the network.

    These tests cover worker orchestration; the URL policy itself is exercised against real
    redirecting servers in tests/test_service_adversarial.py.
    """
    async def passthrough(url, **_kwargs):
        return url

    monkeypatch.setattr("v_cropper.service.worker.validate_redirect_chain", passthrough)


def _settings(tmp_path):
    return Settings(
        redis_url="redis://unused",
        job_ttl_sec=60,
        event_ttl_sec=60,
        max_duration_sec=60,
        max_output_bytes=1_000_000,
        allowed_hosts=("objects.example.test",),
        allow_private_hosts=True,
        work_root=str(tmp_path),
        http_timeout_sec=10,
    )


@pytest.mark.asyncio
async def test_worker_runs_pipeline_uploads_and_publishes_public_result(tmp_path, monkeypatch):
    redis = fakeredis.aioredis.FakeRedis()
    store = JobStore(redis, ttl_sec=60)
    request = CropJobRequest(
        source_url="https://objects.example.test/input.mp4?sig=source-secret",
        destination_url="https://objects.example.test/output.mp4?sig=destination-secret",
        in_s=1,
        out_s=2,
    )
    await store.create("job1", request)

    def fake_pipeline(source, output, **kwargs):
        output.write_bytes(b"video")
        kwargs["progress"]("cropping", 75, None)
        return PipelineResult(
            output_path=output,
            metrics={"rtf": 0.1},
            media={
                "duration_sec": 1.0, "size_bytes": 5, "video_codec": "h264",
                "width": 108, "height": 192, "audio_codec": "aac", "has_audio": True,
            },
        )

    async def fake_upload(url, path, **kwargs):
        assert url.endswith("destination-secret")
        return path.stat().st_size

    monkeypatch.setattr("v_cropper.service.worker.run_pipeline", fake_pipeline)
    monkeypatch.setattr("v_cropper.service.worker.upload_output", fake_upload)
    await JobWorker(store, _settings(tmp_path)).run_job("job1")
    state = await store.get("job1")
    assert state["status"] == "succeeded"
    assert state["result"]["has_audio"]
    assert "source-secret" not in str(state)
    assert "destination-secret" not in str(state)


@pytest.mark.asyncio
async def test_worker_rejects_oversized_requested_trim(tmp_path):
    redis = fakeredis.aioredis.FakeRedis()
    store = JobStore(redis, ttl_sec=60)
    request = CropJobRequest(
        source_url="https://objects.example.test/input.mp4",
        destination_url="https://objects.example.test/output.mp4",
        in_s=0,
        out_s=61,
    )
    await store.create("job1", request)
    await JobWorker(store, _settings(tmp_path)).run_job("job1")
    state = await store.get("job1")
    assert state["status"] == "failed"
    assert state["error"]["code"] == "duration_limit"


def _request(**overrides):
    fields = dict(
        source_url="https://objects.example.test/input.mp4?sig=source-secret",
        destination_url="https://objects.example.test/output.mp4?sig=destination-secret",
    )
    fields.update(overrides)
    return CropJobRequest(**fields)


async def _queued(store, job_id="job1", **overrides):
    request = _request(**overrides)
    await store.create(job_id, request)
    return request


@pytest.fixture
def store():
    return JobStore(fakeredis.aioredis.FakeRedis(), ttl_sec=60)


class TestRunLoop:
    @pytest.mark.asyncio
    async def test_consumes_queued_jobs_until_stopped(self, tmp_path, store):
        await _queued(store)
        worker = JobWorker(store, _settings(tmp_path))
        seen = []

        async def fake_run_job(job_id):
            seen.append(job_id)
            worker.stop()

        worker.run_job = fake_run_job
        await asyncio.wait_for(worker.run(), timeout=10)
        assert seen == ["job1"]

    @pytest.mark.asyncio
    async def test_stops_without_work(self, tmp_path, store):
        worker = JobWorker(store, _settings(tmp_path))
        worker.stop()
        await asyncio.wait_for(worker.run(), timeout=10)


class TestGuards:
    @pytest.mark.asyncio
    async def test_unknown_job_is_a_no_op(self, tmp_path, store):
        await JobWorker(store, _settings(tmp_path)).run_job("missing")
        assert await store.get("missing") is None

    @pytest.mark.asyncio
    async def test_already_cancelled_job_is_not_processed(self, tmp_path, store, monkeypatch):
        await _queued(store)
        await store.cancel("job1")
        called = False

        def fake_pipeline(*a, **k):
            nonlocal called
            called = True

        monkeypatch.setattr("v_cropper.service.worker.run_pipeline", fake_pipeline)
        await JobWorker(store, _settings(tmp_path)).run_job("job1")
        assert not called
        assert (await store.get("job1"))["status"] == "cancelled"

    @pytest.mark.asyncio
    async def test_untrimmed_source_over_the_limit_is_rejected(self, tmp_path, store, monkeypatch):
        """Without a trim window the source itself must be probed before any spend."""
        await _queued(store)
        monkeypatch.setattr("v_cropper.service.worker.probe_media",
                            lambda url: {"duration_sec": 9999.0})
        ran = False

        def fake_pipeline(*a, **k):
            nonlocal ran
            ran = True

        monkeypatch.setattr("v_cropper.service.worker.run_pipeline", fake_pipeline)
        await JobWorker(store, _settings(tmp_path)).run_job("job1")
        state = await store.get("job1")
        assert state["status"] == "failed"
        assert not ran, "the VLM must not be called for an over-long source"

    @pytest.mark.asyncio
    async def test_processed_clip_over_the_limit_is_rejected(self, tmp_path, store, monkeypatch):
        await _queued(store, in_s=0, out_s=5)

        def fake_pipeline(source, output, **kwargs):
            output.write_bytes(b"video")
            return PipelineResult(output_path=output, metrics={},
                                  media={"duration_sec": 9999.0})

        uploaded = False

        async def fake_upload(*a, **k):
            nonlocal uploaded
            uploaded = True
            return 5

        monkeypatch.setattr("v_cropper.service.worker.run_pipeline", fake_pipeline)
        monkeypatch.setattr("v_cropper.service.worker.upload_output", fake_upload)
        await JobWorker(store, _settings(tmp_path)).run_job("job1")
        assert (await store.get("job1"))["status"] == "failed"
        assert not uploaded


class TestFailureHandling:
    @pytest.mark.asyncio
    async def test_pipeline_cancellation_is_reported_as_cancelled(self, tmp_path, store, monkeypatch):
        await _queued(store, in_s=0, out_s=5)

        def fake_pipeline(*a, **k):
            raise PipelineCancelled("cancelled")

        monkeypatch.setattr("v_cropper.service.worker.run_pipeline", fake_pipeline)
        await JobWorker(store, _settings(tmp_path)).run_job("job1")
        state = await store.get("job1")
        assert state["status"] == "cancelled"
        assert state["error"]["code"] == "cancelled"

    @pytest.mark.asyncio
    async def test_cancellation_after_render_still_cancels(self, tmp_path, store, monkeypatch):
        """A cancel that lands while rendering must not publish a result."""
        await _queued(store, in_s=0, out_s=5)

        def fake_pipeline(source, output, **kwargs):
            output.write_bytes(b"video")
            # Outlive one watch_cancel poll so the cancel is observed deterministically
            # rather than depending on how the thread and the loop interleave.
            time.sleep(0.6)
            return PipelineResult(output_path=output, metrics={}, media={"duration_sec": 1.0})

        async def fake_upload(*a, **k):
            raise AssertionError("must not upload a cancelled job")

        monkeypatch.setattr("v_cropper.service.worker.run_pipeline", fake_pipeline)
        monkeypatch.setattr("v_cropper.service.worker.upload_output", fake_upload)

        await store.redis.set(store.cancel_key("job1"), "1")
        await JobWorker(store, _settings(tmp_path)).run_job("job1")
        assert (await store.get("job1"))["status"] == "cancelled"

    @pytest.mark.asyncio
    async def test_failure_message_never_leaks_the_signed_url(self, tmp_path, store, monkeypatch):
        """Exception text routinely contains the full presigned URL."""
        await _queued(store, in_s=0, out_s=5)

        def fake_pipeline(*a, **k):
            raise RuntimeError("connection to https://objects.example.test/input.mp4?sig=source-secret failed")

        monkeypatch.setattr("v_cropper.service.worker.run_pipeline", fake_pipeline)
        await JobWorker(store, _settings(tmp_path)).run_job("job1")
        state = await store.get("job1")
        assert state["status"] == "failed"
        assert state["error"] == {"code": "processing_failed", "message": "crop processing failed"}
        assert "source-secret" not in str(state)

    @pytest.mark.asyncio
    async def test_upload_failure_is_reported_and_workdir_removed(self, tmp_path, store, monkeypatch):
        await _queued(store, in_s=0, out_s=5)

        def fake_pipeline(source, output, **kwargs):
            output.write_bytes(b"video")
            return PipelineResult(output_path=output, metrics={}, media={"duration_sec": 1.0})

        async def fake_upload(*a, **k):
            raise OSError("destination refused the PUT")

        monkeypatch.setattr("v_cropper.service.worker.run_pipeline", fake_pipeline)
        monkeypatch.setattr("v_cropper.service.worker.upload_output", fake_upload)
        await JobWorker(store, _settings(tmp_path)).run_job("job1")
        assert (await store.get("job1"))["status"] == "failed"
        assert not (tmp_path / "job1").exists(), "the working directory must be cleaned up"


class TestOptionMapping:
    def test_every_request_field_reaches_the_pipeline(self):
        """A field silently dropped here is a setting the caller thinks they set."""
        request = _request(
            sport="hockey", prompt="point at the puck", sample_fps=4.0, sample_every=7,
            send_width=1024, spring_k=0.2, concurrency=3, model="custom-model",
            scoreboard=True, scoreboard_sample_count=5, scoreboard_model="sb-model",
            scoreboard_position="top", scoreboard_height_ratio=0.2, scoreboard_opacity=0.5,
        )
        options = _options(request)
        for field in (
            "sport", "prompt", "sample_fps", "sample_every", "send_width", "spring_k",
            "concurrency", "model", "scoreboard", "scoreboard_sample_count",
            "scoreboard_model", "scoreboard_position", "scoreboard_height_ratio",
            "scoreboard_opacity",
        ):
            assert getattr(options, field) == getattr(request, field), field
