"""Job lifecycle against a real Redis rather than fakeredis.

The store leans on behaviour fakeredis only approximates: blocking BLPOP with a timeout,
transactional pipelines, and per-key TTLs. Those are also the parts that decide whether the
worker drains the queue and whether finished jobs expire, so they are worth exercising against
the server the container actually talks to.

Skipped unless a Redis is reachable; point VCROPPER_TEST_REDIS_URL at one to run these:

    docker run -d --rm -p 6399:6379 redis:7-alpine
    VCROPPER_TEST_REDIS_URL=redis://127.0.0.1:6399/0 uv run pytest tests/test_service_redis_integration.py
"""
from __future__ import annotations

import asyncio
import os
import uuid

import pytest
import pytest_asyncio
from redis.asyncio import Redis

from v_cropper.service.models import CropJobRequest
from v_cropper.service.store import QUEUE_KEY, JobStore

pytestmark = pytest.mark.integration

REDIS_URL = os.getenv("VCROPPER_TEST_REDIS_URL", "redis://127.0.0.1:6399/0")


async def _redis_is_up() -> bool:
    client = Redis.from_url(REDIS_URL)
    try:
        await asyncio.wait_for(client.ping(), timeout=2)
        return True
    except Exception:
        return False
    finally:
        await client.aclose()


@pytest_asyncio.fixture
async def store():
    if not await _redis_is_up():
        pytest.skip(f"no Redis at {REDIS_URL}")
    client = Redis.from_url(REDIS_URL)
    await client.delete(QUEUE_KEY)
    yield JobStore(client, ttl_sec=60, event_ttl_sec=60)
    await client.delete(QUEUE_KEY)
    await client.aclose()


def _request() -> CropJobRequest:
    return CropJobRequest(
        source_url="https://objects.example.test/in.mp4?sig=source-secret",
        destination_url="https://objects.example.test/out.mp4?sig=destination-secret",
    )


@pytest.mark.asyncio
async def test_full_job_lifecycle_round_trips_through_redis(store):
    job_id = uuid.uuid4().hex
    await store.create(job_id, _request())

    assert await store.dequeue(timeout=2) == job_id
    assert (await store.get(job_id))["status"] == "queued"
    assert (await store.get_request(job_id)).source_url.endswith("sig=source-secret")

    for phase, pct in [("trimming", 5), ("analyzing", 20), ("cropping", 65), ("muxing_audio", 85)]:
        await store.update(job_id, status="running", phase=phase, progress_pct=pct)
    await store.update(job_id, status="succeeded", phase="complete", progress_pct=100)

    events = await store.events(job_id)
    assert [event["phase"] for event in events] == [
        "queued", "trimming", "analyzing", "cropping", "muxing_audio", "complete"]
    assert [event["status"] for event in events][-1] == "succeeded"
    # Progress is monotonic, which the SSE consumer relies on.
    assert [event["progress_pct"] for event in events] == sorted(
        event["progress_pct"] for event in events)


@pytest.mark.asyncio
async def test_blocking_dequeue_times_out_instead_of_hanging(store):
    """The worker loop can only notice a shutdown between dequeue timeouts."""
    loop = asyncio.get_running_loop()
    started = loop.time()
    assert await store.dequeue(timeout=1) is None
    elapsed = loop.time() - started
    assert 0.5 < elapsed < 3


@pytest.mark.asyncio
async def test_queue_is_fifo_and_each_job_is_delivered_once(store):
    """Two workers must not both pick up the same job."""
    job_ids = [uuid.uuid4().hex for _ in range(5)]
    for job_id in job_ids:
        await store.create(job_id, _request())

    drained = [await store.dequeue(timeout=2) for _ in job_ids]
    assert drained == job_ids
    assert await store.dequeue(timeout=1) is None


@pytest.mark.asyncio
async def test_concurrent_consumers_never_receive_the_same_job(store):
    job_ids = {uuid.uuid4().hex for _ in range(10)}
    for job_id in job_ids:
        await store.create(job_id, _request())

    async def consumer():
        received = []
        while (job_id := await store.dequeue(timeout=1)) is not None:
            received.append(job_id)
        return received

    batches = await asyncio.gather(*(consumer() for _ in range(3)))
    delivered = [job_id for batch in batches for job_id in batch]
    assert sorted(delivered) == sorted(job_ids)
    assert len(delivered) == len(set(delivered))


@pytest.mark.asyncio
async def test_cancellation_is_visible_to_the_worker_process(store):
    """Cancel is written by the API process and read by the worker; it must cross Redis."""
    job_id = uuid.uuid4().hex
    await store.create(job_id, _request())
    await store.dequeue(timeout=2)
    await store.update(job_id, status="running", phase="cropping", progress_pct=50)

    assert not await store.is_cancelled(job_id)
    assert await store.cancel(job_id)

    # A separate client, as the worker would be.
    other = JobStore(Redis.from_url(REDIS_URL), ttl_sec=60, event_ttl_sec=60)
    try:
        assert await other.is_cancelled(job_id)
        # A running job stays running until the worker acknowledges it.
        assert (await other.get(job_id))["status"] == "running"
    finally:
        await other.redis.aclose()


@pytest.mark.asyncio
async def test_every_key_written_carries_an_expiry(store):
    """An unexpiring key per job would grow the instance's memory without bound."""
    job_id = uuid.uuid4().hex
    await store.create(job_id, _request())
    await store.update(job_id, status="running", phase="cropping")
    await store.cancel(job_id)

    for key in (store.state_key(job_id), store.request_key(job_id),
                store.events_key(job_id), store.cancel_key(job_id)):
        assert await store.redis.ttl(key) > 0, f"{key} has no TTL"
