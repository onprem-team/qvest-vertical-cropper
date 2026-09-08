from __future__ import annotations

import fakeredis.aioredis
import pytest

from v_cropper.service.models import CropJobRequest
from v_cropper.service.store import JobStore


@pytest.fixture
def job_request():
    return CropJobRequest(
        source_url="https://objects.example.test/input.mp4?sig=source-secret",
        destination_url="https://objects.example.test/output.mp4?sig=destination-secret",
    )


@pytest.mark.asyncio
async def test_queue_state_events_and_ttl(job_request):
    redis = fakeredis.aioredis.FakeRedis()
    store = JobStore(redis, ttl_sec=60)
    await store.create("job1", job_request)
    assert await store.dequeue() == "job1"
    state = await store.update("job1", status="running", phase="cropping", progress_pct=50)
    assert state["progress_pct"] == 50
    events = await store.events("job1")
    assert [event["status"] for event in events] == ["queued", "running"]
    assert await redis.ttl(store.state_key("job1")) > 0
    assert await redis.ttl(store.request_key("job1")) > 0
    assert await redis.ttl(store.events_key("job1")) > 0


@pytest.mark.asyncio
async def test_cancel_queued_job_is_terminal(job_request):
    redis = fakeredis.aioredis.FakeRedis()
    store = JobStore(redis, ttl_sec=60)
    await store.create("job1", job_request)
    assert await store.cancel("job1")
    assert await store.is_cancelled("job1")
    assert (await store.get("job1"))["status"] == "cancelled"
    assert not await store.cancel("missing")


@pytest.mark.asyncio
async def test_unknown_job_reads_and_updates_are_none(job_request):
    """Expired or bogus ids must return None rather than resurrect a partial state."""
    store = JobStore(fakeredis.aioredis.FakeRedis(), ttl_sec=60)
    assert await store.get("missing") is None
    assert await store.get_request("missing") is None
    assert await store.update("missing", status="running") is None
    assert await store.events("missing") == []
    assert not await store.is_cancelled("missing")


@pytest.mark.asyncio
async def test_dequeue_returns_none_when_the_queue_stays_empty():
    """The worker loop relies on this to stay responsive to shutdown."""
    store = JobStore(fakeredis.aioredis.FakeRedis(), ttl_sec=60)
    assert await store.dequeue(timeout=1) is None


@pytest.mark.asyncio
async def test_cancelling_a_finished_job_does_not_rewrite_its_outcome(job_request):
    """A late cancel racing a completing job must not turn a success into a cancellation."""
    store = JobStore(fakeredis.aioredis.FakeRedis(), ttl_sec=60)
    await store.create("job1", job_request)
    await store.dequeue()
    await store.update("job1", status="succeeded", phase="complete", progress_pct=100)

    assert await store.cancel("job1") is True
    assert (await store.get("job1"))["status"] == "succeeded"


@pytest.mark.asyncio
async def test_events_can_be_replayed_from_an_offset(job_request):
    """SSE reconnects resume from the last event the client saw."""
    store = JobStore(fakeredis.aioredis.FakeRedis(), ttl_sec=60)
    await store.create("job1", job_request)
    await store.update("job1", status="running", phase="cropping")
    await store.update("job1", status="succeeded", phase="complete")

    assert [event["status"] for event in await store.events("job1", start=1)] == [
        "running", "succeeded"]
    assert await store.events("job1", start=99) == []


@pytest.mark.asyncio
async def test_stored_state_never_contains_the_signed_urls(job_request):
    """Job state is served over the API; the presigned URLs must stay in the request record."""
    redis = fakeredis.aioredis.FakeRedis()
    store = JobStore(redis, ttl_sec=60)
    await store.create("job1", job_request)
    await store.update("job1", status="failed", error="boom")

    serialised = (await redis.get(store.state_key("job1"))).decode()
    events = str(await store.events("job1"))
    for secret in ("source-secret", "destination-secret"):
        assert secret not in serialised
        assert secret not in events

    # The worker still needs them, so they live under the request key.
    assert (await store.get_request("job1")).source_url.endswith("sig=source-secret")
