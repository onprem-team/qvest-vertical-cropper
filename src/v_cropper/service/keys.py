"""Revocable API keys and the verifier interface used by the service.

The bootstrap token in CROPPER_API_TOKEN remains the admin credential. Every other
key is minted through the admin API, stored as a hash in Redis, and identified by
the public key_id embedded in the token so lookup is one read rather than a scan.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
import secrets
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Protocol

from redis.asyncio import Redis

PREFIX = "vcrop"
# 8 bytes of hex: no underscore, so `vcrop_<id>_<secret>` splits unambiguously.
KEY_ID_PATTERN = re.compile(r"^[0-9a-f]{16}$")
NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 ._-]{0,63}$")
BOOTSTRAP_KEY_ID = "bootstrap"


@dataclass(frozen=True)
class Principal:
    key_id: str
    name: str
    is_admin: bool


BOOTSTRAP_PRINCIPAL = Principal(key_id=BOOTSTRAP_KEY_ID, name="bootstrap", is_admin=True)


class Verifier(Protocol):
    async def verify(self, presented: str) -> Principal | None:
        """Return a principal if ``presented`` is valid, else None."""


def hash_secret(secret: str) -> str:
    return hashlib.sha256(secret.encode("utf-8")).hexdigest()


def secrets_match(presented: str, expected_hash: str) -> bool:
    return hmac.compare_digest(hash_secret(presented), expected_hash)


def parse_issued_key(value: str) -> tuple[str, str] | None:
    """Split ``vcrop_<key_id>_<secret>``. Anything else is not an issued key."""
    parts = value.split("_", 2)
    if len(parts) != 3 or parts[0] != PREFIX:
        return None
    key_id, secret = parts[1], parts[2]
    if not KEY_ID_PATTERN.fullmatch(key_id) or not secret:
        return None
    return key_id, secret


def format_issued_key(key_id: str, secret: str) -> str:
    return f"{PREFIX}_{key_id}_{secret}"


def validate_key_name(name: str) -> str:
    cleaned = name.strip()
    if cleaned.lower().startswith(f"{PREFIX}_"):
        raise ValueError("name must not look like an issued API key")
    if not NAME_PATTERN.fullmatch(cleaned):
        raise ValueError(
            "name must start with an alphanumeric and be at most 64 characters of "
            "letters, digits, spaces, dots, underscores, or hyphens"
        )
    return cleaned


class KeyStore:
    """Hashed API keys in Redis. The secret is never persisted."""

    def __init__(self, redis: Redis):
        self.redis = redis

    @staticmethod
    def record_key(key_id: str) -> str:
        if not KEY_ID_PATTERN.fullmatch(key_id):
            raise ValueError("invalid key id")
        return f"v-cropper:apikey:{key_id}"

    @staticmethod
    def index_key() -> str:
        return "v-cropper:apikeys"

    async def mint(self, name: str) -> tuple[str, dict[str, Any]]:
        cleaned = validate_key_name(name)
        key_id = secrets.token_hex(8)
        secret = secrets.token_urlsafe(32)
        now = datetime.now(timezone.utc).isoformat()
        record = {
            "key_id": key_id,
            "name": cleaned,
            "hash": hash_secret(secret),
            "created_at": now,
            "last_used_at": None,
        }
        async with self.redis.pipeline(transaction=True) as pipe:
            pipe.set(self.record_key(key_id), json.dumps(record))
            pipe.sadd(self.index_key(), key_id)
            await pipe.execute()
        return format_issued_key(key_id, secret), _public_record(record)

    async def get(self, key_id: str) -> dict[str, Any] | None:
        try:
            raw = await self.redis.get(self.record_key(key_id))
        except ValueError:
            return None
        return json.loads(raw) if raw else None

    async def list(self) -> list[dict[str, Any]]:
        ids = await self.redis.smembers(self.index_key())
        records: list[dict[str, Any]] = []
        for raw_id in ids:
            key_id = raw_id.decode() if isinstance(raw_id, bytes) else raw_id
            record = await self.get(key_id)
            if record is not None:
                records.append(_public_record(record))
        records.sort(key=lambda item: item["created_at"])
        return records

    async def revoke(self, key_id: str) -> bool:
        try:
            redis_key = self.record_key(key_id)
        except ValueError:
            return False
        async with self.redis.pipeline(transaction=True) as pipe:
            pipe.delete(redis_key)
            pipe.srem(self.index_key(), key_id)
            deleted, _ = await pipe.execute()
        return bool(deleted)

    async def verify(self, presented: str) -> Principal | None:
        parsed = parse_issued_key(presented)
        if parsed is None:
            return None
        key_id, secret = parsed
        record = await self.get(key_id)
        if record is None:
            return None
        if not secrets_match(secret, record["hash"]):
            return None
        await self._touch(key_id, record)
        return Principal(key_id=key_id, name=record["name"], is_admin=False)

    async def _touch(self, key_id: str, record: dict[str, Any]) -> None:
        record["last_used_at"] = datetime.now(timezone.utc).isoformat()
        await self.redis.set(self.record_key(key_id), json.dumps(record))


class StaticTokenVerifier:
    """The configured bootstrap token. Always admin; never stored in Redis."""

    def __init__(self, expected: str):
        self.expected = expected

    async def verify(self, presented: str) -> Principal | None:
        if not self.expected:
            return None
        if not hmac.compare_digest(presented, self.expected):
            return None
        return BOOTSTRAP_PRINCIPAL


class ChainVerifier:
    """Try verifiers in order; first match wins."""

    def __init__(self, verifiers: list[Verifier]):
        self.verifiers = verifiers

    async def verify(self, presented: str) -> Principal | None:
        for verifier in self.verifiers:
            principal = await verifier.verify(presented)
            if principal is not None:
                return principal
        return None


def _public_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "key_id": record["key_id"],
        "name": record["name"],
        "created_at": record["created_at"],
        "last_used_at": record["last_used_at"],
    }
