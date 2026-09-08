"""Environment-backed service configuration."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

TRUTHY = {"1", "true", "yes"}


def _str(name: str, default: str = "") -> Any:
    return field(default_factory=lambda: os.getenv(name, default))


def _int(name: str, default: int) -> Any:
    return field(default_factory=lambda: int(os.getenv(name, str(default))))


def _bool(name: str, default: bool = False) -> Any:
    return field(default_factory=lambda: os.getenv(name, str(default)).strip().lower() in TRUTHY)


def _csv(name: str, default: str) -> Any:
    def parse() -> tuple[str, ...]:
        return tuple(
            value.strip().lower() for value in os.getenv(name, default).split(",") if value.strip()
        )

    return field(default_factory=parse)


@dataclass(frozen=True)
class Settings:
    """Service configuration, read from the environment when an instance is constructed.

    Every field resolves through a ``default_factory``, so ``Settings()`` reflects the
    environment at call time rather than at import. Tests may therefore monkeypatch the
    environment, though passing fields explicitly stays clearer for most cases.
    """

    redis_url: str = _str("CROPPER_REDIS_URL", "redis://redis:6379/1")
    # No default: an empty token fails closed, refusing every /v1 request.
    api_token: str = _str("CROPPER_API_TOKEN")
    job_ttl_sec: int = _int("CROPPER_JOB_TTL_SEC", 86400)
    event_ttl_sec: int = _int("CROPPER_EVENT_TTL_SEC", 86400)
    max_duration_sec: int = _int("CROPPER_MAX_DURATION_SEC", 900)
    max_output_bytes: int = _int("CROPPER_MAX_OUTPUT_BYTES", 2_000_000_000)
    allowed_hosts: tuple[str, ...] = _csv("CROPPER_ALLOWED_HOSTS", "minio,localhost")
    allow_private_hosts: bool = _bool("CROPPER_ALLOW_PRIVATE_HOSTS")
    work_root: str = _str("CROPPER_WORK_ROOT", "/tmp/v-cropper")
    http_timeout_sec: int = _int("CROPPER_HTTP_TIMEOUT_SEC", 300)
    # Where uvicorn listens. In a container this is necessarily 0.0.0.0 and says nothing
    # about external reachability; see public_origin below.
    bind_host: str = _str("CROPPER_HOST", "0.0.0.0")
    bind_port: int = _int("CROPPER_PORT", 8080)
    log_level: str = _str("CROPPER_LOG_LEVEL", "info")
    # Operator's declaration of how callers actually reach this service. Empty means
    # undeclared, which is the shipped default: loopback publish behind an SSH tunnel.
    public_origin: str = _str("CROPPER_PUBLIC_ORIGIN")
    allow_plaintext: bool = _bool("CROPPER_ALLOW_PLAINTEXT")


settings = Settings()
