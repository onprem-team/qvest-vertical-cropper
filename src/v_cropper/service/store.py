"""Redis queue, durable job state, events, cancellation, and TTL handling."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from redis.asyncio import Redis

from .models import CropJobRequest

QUEUE_KEY = "v-cropper:jobs"


class JobStore:
    def __init__(self, redis: Redis, *, ttl_sec: int = 86400, event_ttl_sec: int | None = None):
        self.redis = redis
        self.ttl_sec = ttl_sec
        self.event_ttl_sec = event_ttl_sec or ttl_sec

    @staticmethod
    def state_key(job_id: str) -> str:
        return f"v-cropper:job:{job_id}:state"

    @staticmethod
    def request_key(job_id: str) -> str:
        return f"v-cropper:job:{job_id}:request"

    @staticmethod
    def events_key(job_id: str) -> str:
        return f"v-cropper:job:{job_id}:events"

    @staticmethod
    def cancel_key(job_id: str) -> str:
        return f"v-cropper:job:{job_id}:cancel"

    async def create(self, job_id: str, request: CropJobRequest) -> dict[str, Any]:
        state = {
            "job_id": job_id, "status": "queued", "phase": "queued", "progress_pct": 0,
            "metrics": None, "result": None, "error": None,
        }
        async with self.redis.pipeline(transaction=True) as pipe:
            pipe.set(self.state_key(job_id), json.dumps(state), ex=self.ttl_sec)
            pipe.set(self.request_key(job_id), request.model_dump_json(), ex=self.ttl_sec)
            pipe.rpush(QUEUE_KEY, job_id)
            await pipe.execute()
        await self.append_event(job_id, state)
        return state

    async def get(self, job_id: str) -> dict[str, Any] | None:
        raw = await self.redis.get(self.state_key(job_id))
        return json.loads(raw) if raw else None

    async def get_request(self, job_id: str) -> CropJobRequest | None:
        raw = await self.redis.get(self.request_key(job_id))
        return CropJobRequest.model_validate_json(raw) if raw else None

    async def update(self, job_id: str, **changes: Any) -> dict[str, Any] | None:
        state = await self.get(job_id)
        if state is None:
            return None
        state.update(changes)
        await self.redis.set(self.state_key(job_id), json.dumps(state), ex=self.ttl_sec)
        await self.append_event(job_id, state)
        return state

    async def append_event(self, job_id: str, state: dict[str, Any]) -> None:
        event = {"at": datetime.now(timezone.utc).isoformat(), **state}
        key = self.events_key(job_id)
        async with self.redis.pipeline(transaction=True) as pipe:
            pipe.rpush(key, json.dumps(event))
            pipe.expire(key, self.event_ttl_sec)
            await pipe.execute()

    async def events(self, job_id: str, start: int = 0) -> list[dict[str, Any]]:
        values = await self.redis.lrange(self.events_key(job_id), start, -1)
        return [json.loads(value) for value in values]

    async def dequeue(self, timeout: int = 1) -> str | None:
        item = await self.redis.blpop(QUEUE_KEY, timeout=timeout)
        if not item:
            return None
        value = item[1]
        return value.decode() if isinstance(value, bytes) else value

    async def cancel(self, job_id: str) -> bool:
        state = await self.get(job_id)
        if state is None:
            return False
        if state["status"] in {"succeeded", "failed", "cancelled"}:
            return True
        await self.redis.set(self.cancel_key(job_id), "1", ex=self.ttl_sec)
        if state["status"] == "queued":
            await self.update(job_id, status="cancelled", phase="cancelled", error=None)
        return True

    async def is_cancelled(self, job_id: str) -> bool:
        return bool(await self.redis.exists(self.cancel_key(job_id)))
