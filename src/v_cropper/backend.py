# SPDX-FileCopyrightText: Copyright (c) 2026 Qvest.US, LLC
# SPDX-License-Identifier: Apache-2.0
"""Provider-neutral vision backends for OpenAI-compatible APIs and AWS Bedrock.

Both AI passes (focus + scoreboard OCR) use the same synchronous interface.
Provider selection and connection settings are resolved from environment
variables, with OpenAI-compatible behavior remaining the default.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable
from urllib.parse import urlparse, urlunparse

import boto3
import openai
from botocore.config import Config as BotocoreConfig
from botocore.exceptions import (
    BotoCoreError,
    ClientError,
    ConnectionClosedError,
    ConnectTimeoutError,
    EndpointConnectionError,
    ReadTimeoutError,
)

logger = logging.getLogger(__name__)

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
GEMINI_COMPAT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
# Default NVIDIA-hosted VLM. Catalog ids drift: nemotron-nano-12b-v2-vl went 410 Gone
# (EOL 2026-08-26). Re-confirm via GET /v1/models if calls start failing.
DEFAULT_NVIDIA_MODEL = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
# Cross-region inference profile, rather than a region-bound foundation-model id.
DEFAULT_BEDROCK_MODEL = "global.anthropic.claude-sonnet-4-5-20250929-v1:0"

# Disable Gemini 2.5 "thinking" over the OpenAI-compatible endpoint. Gemini's compat layer
# maps reasoning_effort="none" to thinking_budget=0 for 2.5 models; passing it through the
# SDK's extra_body sets it as a top-level request field (verified live 2026-07). Without
# this, gemini-2.5-flash burns the token budget on hidden reasoning and truncates the JSON.
GEMINI_THINKING_OFF: dict = {"reasoning_effort": "none"}

# Errors worth retrying with backoff (rate limits + transient transport/5xx).
RETRYABLE_ERRORS = (
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.InternalServerError,
)


def frame_to_data_uri(jpeg_bytes: bytes, mime: str = "image/jpeg") -> str:
    """Encode JPEG bytes as an OpenAI ``image_url`` data URI."""
    b64 = base64.b64encode(jpeg_bytes).decode("ascii")
    return f"data:{mime};base64,{b64}"


def image_part(jpeg_bytes: bytes) -> dict:
    """Build an OpenAI ``image_url`` content part from JPEG bytes."""
    return {"type": "image_url", "image_url": {"url": frame_to_data_uri(jpeg_bytes)}}


def text_part(text: str) -> dict:
    """Build an OpenAI ``text`` content part."""
    return {"type": "text", "text": text}


def _env_int(name: str) -> int | None:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        logger.warning("Invalid %s=%r; ignoring", name, raw)
        return None


def _env_json(name: str) -> dict | None:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return None
    try:
        val = json.loads(raw)
    except (TypeError, ValueError):
        # The raw value can contain sensitive provider-specific settings. Never log it.
        logger.warning("Invalid %s (not JSON); ignoring", name)
        return None
    if not isinstance(val, dict):
        logger.warning("Invalid %s: expected a JSON object; ignoring", name)
        return None
    return val


def _env_headers(name: str) -> dict[str, str]:
    """Read a JSON object of string HTTP headers without logging header values."""
    value = _env_json(name)
    if value is None:
        return {}
    if not all(isinstance(key, str) and key and isinstance(val, str)
               for key, val in value.items()):
        logger.warning("Invalid %s: expected a JSON object of non-empty string header names and values; "
                       "ignoring", name)
        return {}
    protected = {"authorization", "content-length", "host"}
    if any(key.lower() in protected for key in value):
        logger.warning("Invalid %s: protected transport headers must be configured by the client; ignoring", name)
        return {}
    return value


def normalize_base_url(value: str) -> str:
    """Validate and normalize an OpenAI-compatible endpoint without credential-bearing URLs."""
    value = value.strip()
    if not value:
        raise ValueError("VLM base URL must not be empty")
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("VLM base URL must be an absolute http(s) URL")
    if parsed.username or parsed.password:
        raise ValueError("VLM base URL must not contain credentials; use VCROPPER_API_KEY instead")
    if parsed.query or parsed.fragment:
        raise ValueError("VLM base URL must not contain a query string or fragment")

    path = parsed.path.rstrip("/")
    normalized = urlunparse((parsed.scheme, parsed.netloc, path, "", "", ""))
    if "/v1" not in path:
        logger.warning("VLM base URL %s does not contain '/v1'; confirm it is OpenAI-compatible", normalized)
    return normalized


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        logger.warning("Invalid %s=%r; using default %r", name, raw, default)
        return default


@dataclass
class BackendConfig:
    """Resolved connection settings for a single provider."""

    base_url: str
    api_key: str
    model: str
    timeout: float = 120.0
    max_retries: int = 3
    backoff_base: float = 0.5
    response_format: dict | None = None  # e.g. {"type": "json_object"}
    extra_headers: dict[str, str] = field(default_factory=dict)
    max_tokens: int | None = None  # None = provider default (no cap sent)
    extra_body: dict | None = None  # provider-specific passthrough (e.g. Gemini thinking-off)


@dataclass
class BedrockConfig:
    """Resolved settings for Bedrock Runtime's Converse API."""

    region: str
    model: str
    timeout: float = 120.0
    max_retries: int = 3
    backoff_base: float = 0.5
    max_tokens: int | None = None


@runtime_checkable
class VisionBackend(Protocol):
    """Provider-neutral vision-completion interface."""

    model: str
    last_usage: dict[str, int]  # tokens from the most recent call
    usage_totals: dict[str, int]  # thread-safe cumulative tokens + api_calls

    def complete(
        self,
        messages: list[dict],
        *,
        temperature: float = 0.1,
        response_format: dict | None = None,
        max_tokens: int | None = None,
        extra_body: dict | None = None,
    ) -> str: ...


def _provider() -> str:
    provider = os.environ.get("VCROPPER_PROVIDER", "openai").strip().lower()
    if provider not in {"openai", "bedrock"}:
        raise RuntimeError("VCROPPER_PROVIDER must be 'openai' or 'bedrock'")
    return provider


def load_bedrock_config(model: str | None = None) -> BedrockConfig:
    """Resolve Bedrock settings; AWS credentials use boto3's standard chain."""
    region = os.environ.get("BEDROCK_REGION") or os.environ.get("AWS_REGION")
    if not region:
        raise RuntimeError("No AWS Bedrock region. Set BEDROCK_REGION or AWS_REGION.")
    return BedrockConfig(
        region=region,
        model=model
        or os.environ.get("VCROPPER_MODEL")
        or os.environ.get("BEDROCK_VLM_MODEL_ID")
        or DEFAULT_BEDROCK_MODEL,
        timeout=_env_float("VCROPPER_TIMEOUT", 120.0),
        max_tokens=_env_int("VCROPPER_MAX_TOKENS"),
    )


def load_backend_config(
    api_key: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
) -> BackendConfig:
    """Resolve a BackendConfig from explicit args + environment variables.

    Precedence (first hit wins): explicit arg -> VCROPPER_* -> GEMINI_API_KEY.
    Default endpoint is NVIDIA-hosted. Back-compat: if the key came only from
    GEMINI_API_KEY and no VCROPPER_BASE_URL was set, route to Gemini's
    OpenAI-compatible endpoint so existing .env files keep working.
    """
    env_base = os.environ.get("VCROPPER_BASE_URL")
    env_key = os.environ.get("VCROPPER_API_KEY")
    gemini_key = os.environ.get("GEMINI_API_KEY")
    key = api_key or env_key or gemini_key
    if not key:
        raise RuntimeError(
            "No VLM API key. Set VCROPPER_API_KEY (+ VCROPPER_BASE_URL / VCROPPER_MODEL), "
            "or GEMINI_API_KEY, or pass --api-key."
        )

    used_gemini_fallback = (
        not api_key and not env_key and bool(gemini_key) and base_url is None and not env_base
    )
    env_model = os.environ.get("VCROPPER_MODEL")
    if used_gemini_fallback:
        resolved_base_url = GEMINI_COMPAT_BASE_URL
        resolved_model = model or env_model or DEFAULT_GEMINI_MODEL
    else:
        resolved_base_url = base_url if base_url is not None else env_base or NVIDIA_BASE_URL
        resolved_model = model or env_model or DEFAULT_NVIDIA_MODEL
    resolved_base_url = normalize_base_url(resolved_base_url)

    rf = os.environ.get("VCROPPER_RESPONSE_FORMAT")

    # extra_body: explicit env (JSON) wins; otherwise default Gemini thinking-off when the
    # resolved endpoint is Gemini's compat layer (so parity works out of the box).
    extra_body = _env_json("VCROPPER_EXTRA_BODY")
    if extra_body is None and resolved_base_url == GEMINI_COMPAT_BASE_URL:
        extra_body = dict(GEMINI_THINKING_OFF)

    return BackendConfig(
        base_url=resolved_base_url,
        api_key=key,
        model=resolved_model,
        timeout=_env_float("VCROPPER_TIMEOUT", 120.0),
        response_format={"type": rf} if rf else None,
        extra_headers=_env_headers("VCROPPER_EXTRA_HEADERS"),
        max_tokens=_env_int("VCROPPER_MAX_TOKENS"),
        extra_body=extra_body,
    )


def describe_backend_config(config: BackendConfig) -> dict[str, object]:
    """Return safe-to-log connection metadata without key or header values."""
    return {
        "base_url": config.base_url,
        "model": config.model,
        "extra_header_names": sorted(config.extra_headers),
    }


def describe_bedrock_config(config: BedrockConfig) -> dict[str, object]:
    """Return safe-to-log Bedrock metadata. Credentials come from boto3, never from here."""
    return {"provider": "bedrock", "region": config.region, "model": config.model}


class OpenAICompatBackend:
    """VisionBackend backed by the OpenAI SDK against any compatible endpoint."""

    def __init__(self, config: BackendConfig):
        self.config = config
        self._client = None
        self._usage_lock = threading.Lock()
        self.last_usage: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self.usage_totals: dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "api_calls": 0,
        }

    @property
    def model(self) -> str:
        return self.config.model

    @property
    def cache_identity(self) -> dict[str, object]:
        """Configuration that can affect a response cache, excluding the API key itself."""
        return {
            "base_url": self.config.base_url,
            "model": self.config.model,
            "response_format": self.config.response_format,
            "extra_headers": self.config.extra_headers,
            "max_tokens": self.config.max_tokens,
            "extra_body": self.config.extra_body,
        }

    @property
    def client(self):
        """Lazily construct the OpenAI client (SDK retries disabled; we retry explicitly)."""
        if self._client is None:
            self._client = openai.OpenAI(
                base_url=self.config.base_url,
                api_key=self.config.api_key,
                timeout=self.config.timeout,
                max_retries=0,
            )
        return self._client

    def _record_usage(self, resp) -> None:
        usage = getattr(resp, "usage", None)
        call = {
            "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
            "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
            "total_tokens": getattr(usage, "total_tokens", 0) or 0,
        }
        # Aggregate under a lock so concurrent workers don't corrupt the totals.
        with self._usage_lock:
            self.last_usage = call
            self.usage_totals["prompt_tokens"] += call["prompt_tokens"]
            self.usage_totals["completion_tokens"] += call["completion_tokens"]
            self.usage_totals["total_tokens"] += call["total_tokens"]
            self.usage_totals["api_calls"] += 1

    def complete(
        self,
        messages: list[dict],
        *,
        temperature: float = 0.1,
        response_format: dict | None = None,
        max_tokens: int | None = None,
        extra_body: dict | None = None,
    ) -> str:
        rf = response_format or self.config.response_format
        kwargs: dict = {"model": self.config.model, "messages": messages, "temperature": temperature}
        if rf:
            kwargs["response_format"] = rf
        if self.config.extra_headers:
            kwargs["extra_headers"] = self.config.extra_headers
        resolved_max_tokens = max_tokens if max_tokens is not None else self.config.max_tokens
        if resolved_max_tokens is not None:
            kwargs["max_tokens"] = resolved_max_tokens
        # Per-call extra_body shallow-merges over the config default (per-call keys win).
        merged_extra_body = None
        if self.config.extra_body or extra_body:
            merged_extra_body = {**(self.config.extra_body or {}), **(extra_body or {})}
        if merged_extra_body:
            kwargs["extra_body"] = merged_extra_body

        last_exc: Exception | None = None
        for attempt in range(self.config.max_retries + 1):
            try:
                resp = self.client.chat.completions.create(**kwargs)
                self._record_usage(resp)
                if not resp.choices:
                    return ""
                return resp.choices[0].message.content or ""
            except RETRYABLE_ERRORS as e:
                last_exc = e
                if attempt < self.config.max_retries:
                    wait = self.config.backoff_base * (2**attempt)
                    logger.info(
                        "VLM call %s — retry %d/%d in %.1fs",
                        type(e).__name__,
                        attempt + 1,
                        self.config.max_retries,
                        wait,
                    )
                    time.sleep(wait)
                    continue
                raise
        # Defensive: loop always returns or raises above.
        raise last_exc  # pragma: no cover


_BEDROCK_RETRYABLE_CODES = {
    "InternalServerException",
    "ModelNotReadyException",
    "ModelTimeoutException",
    "ServiceUnavailableException",
    "ThrottlingException",
}
_BEDROCK_TRANSPORT_ERRORS = (
    ConnectTimeoutError,
    ConnectionClosedError,
    EndpointConnectionError,
    ReadTimeoutError,
)


def _data_uri_image(url: str) -> dict:
    """Translate an OpenAI image data URI to a Bedrock Converse image block."""
    if not url.startswith("data:") or ";base64," not in url:
        raise ValueError("Bedrock image_url parts must contain a base64 data URI")
    header, encoded = url.split(",", 1)
    media_type = header[5:].split(";", 1)[0].lower()
    formats = {
        "image/jpeg": "jpeg",
        "image/jpg": "jpeg",
        "image/png": "png",
        "image/gif": "gif",
        "image/webp": "webp",
    }
    try:
        image_format = formats[media_type]
    except KeyError as exc:
        raise ValueError(f"Unsupported Bedrock image media type: {media_type}") from exc
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (ValueError, TypeError) as exc:
        raise ValueError("Invalid base64 image data URI") from exc
    return {"image": {"format": image_format, "source": {"bytes": raw}}}


def _bedrock_content(content: str | list[dict]) -> list[dict]:
    if isinstance(content, str):
        return [{"text": content}]
    blocks: list[dict] = []
    for part in content:
        part_type = part.get("type")
        if part_type == "text":
            blocks.append({"text": part.get("text", "")})
        elif part_type == "image_url":
            image_url = part.get("image_url", {})
            url = image_url if isinstance(image_url, str) else image_url.get("url", "")
            blocks.append(_data_uri_image(url))
        else:
            raise ValueError(f"Unsupported OpenAI content part for Bedrock: {part_type!r}")
    return blocks


def _bedrock_messages(messages: list[dict]) -> tuple[list[dict], list[dict]]:
    """Translate OpenAI chat messages into Converse messages and system blocks."""
    converted: list[dict] = []
    system: list[dict] = []
    for message in messages:
        role = message.get("role")
        content = _bedrock_content(message.get("content", ""))
        if role == "system":
            for block in content:
                if "text" not in block:
                    raise ValueError("Bedrock system messages may contain text only")
                system.append(block)
        elif role in {"user", "assistant"}:
            converted.append({"role": role, "content": content})
        else:
            raise ValueError(f"Unsupported message role for Bedrock: {role!r}")
    return converted, system


def _is_temperature_rejection(exc: ClientError) -> bool:
    error = exc.response.get("Error", {})
    text = f"{error.get('Code', '')} {error.get('Message', '')}".lower()
    return "validation" in text and "temperature" in text


def _is_retryable_bedrock_error(exc: Exception) -> bool:
    if isinstance(exc, _BEDROCK_TRANSPORT_ERRORS):
        return True
    if isinstance(exc, ClientError):
        return exc.response.get("Error", {}).get("Code") in _BEDROCK_RETRYABLE_CODES
    return False


class BedrockBackend:
    """VisionBackend backed by boto3 Bedrock Runtime's Converse API."""

    def __init__(self, config: BedrockConfig, client=None):
        self.config = config
        self._client = client
        self._client_lock = threading.Lock()
        self._usage_lock = threading.Lock()
        self.last_usage: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self.usage_totals: dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "api_calls": 0,
        }

    @property
    def model(self) -> str:
        return self.config.model

    @property
    def cache_identity(self) -> dict[str, object]:
        """Configuration that can affect a response cache. Mirrors OpenAICompatBackend so a
        cached eval run can never be shared across providers or regions."""
        return {
            "provider": "bedrock",
            "region": self.config.region,
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
        }

    @property
    def client(self):
        if self._client is None:
            with self._client_lock:
                if self._client is None:
                    self._client = boto3.client(
                        "bedrock-runtime",
                        region_name=self.config.region,
                        config=BotocoreConfig(
                            connect_timeout=self.config.timeout,
                            read_timeout=self.config.timeout,
                            retries={"total_max_attempts": 1, "mode": "standard"},
                        ),
                    )
        return self._client

    def _record_usage(self, response: dict) -> None:
        usage = response.get("usage") or {}
        call = {
            "prompt_tokens": usage.get("inputTokens", 0) or 0,
            "completion_tokens": usage.get("outputTokens", 0) or 0,
            "total_tokens": usage.get("totalTokens", 0) or 0,
        }
        with self._usage_lock:
            self.last_usage = call
            for key, value in call.items():
                self.usage_totals[key] += value
            self.usage_totals["api_calls"] += 1

    def complete(
        self,
        messages: list[dict],
        *,
        temperature: float = 0.1,
        response_format: dict | None = None,
        max_tokens: int | None = None,
        extra_body: dict | None = None,
    ) -> str:
        converted, system = _bedrock_messages(messages)
        inference_config: dict = {"temperature": temperature}
        resolved_max_tokens = max_tokens if max_tokens is not None else self.config.max_tokens
        if resolved_max_tokens is not None:
            inference_config["maxTokens"] = resolved_max_tokens
        kwargs: dict = {
            "modelId": self.config.model,
            "messages": converted,
            "inferenceConfig": inference_config,
        }
        if system:
            kwargs["system"] = system
        # Converse has no OpenAI response_format equivalent. Callers already prompt for JSON.
        if response_format:
            logger.debug("Ignoring OpenAI response_format for Bedrock Converse")
        if extra_body:
            kwargs["additionalModelRequestFields"] = extra_body

        last_exc: Exception | None = None
        retried_without_temperature = False
        attempt = 0
        while attempt <= self.config.max_retries:
            try:
                response = self.client.converse(**kwargs)
                self._record_usage(response)
                blocks = response.get("output", {}).get("message", {}).get("content", [])
                return "".join(block.get("text", "") for block in blocks if "text" in block)
            except ClientError as exc:
                if not retried_without_temperature and _is_temperature_rejection(exc):
                    retried_without_temperature = True
                    kwargs["inferenceConfig"].pop("temperature", None)
                    continue
                last_exc = exc
                if not _is_retryable_bedrock_error(exc) or attempt >= self.config.max_retries:
                    raise
            except _BEDROCK_TRANSPORT_ERRORS as exc:
                last_exc = exc
                if attempt >= self.config.max_retries:
                    raise
            except BotoCoreError:
                raise
            wait = self.config.backoff_base * (2**attempt)
            logger.info(
                "Bedrock call %s — retry %d/%d in %.1fs",
                type(last_exc).__name__,
                attempt + 1,
                self.config.max_retries,
                wait,
            )
            time.sleep(wait)
            attempt += 1
        raise last_exc  # pragma: no cover


def make_backend(
    api_key: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    config: BackendConfig | BedrockConfig | None = None,
) -> VisionBackend:
    """Construct the selected backend; OpenAI-compatible remains the default."""
    if isinstance(config, BedrockConfig):
        logger.info("Resolved VLM backend: %s", describe_bedrock_config(config))
        return BedrockBackend(config)
    if isinstance(config, BackendConfig):
        logger.info("Resolved VLM backend: %s", describe_backend_config(config))
        return OpenAICompatBackend(config)
    if _provider() == "bedrock":
        if base_url is not None:
            # Silently dropping an explicit endpoint would misreport which service ran.
            logger.warning(
                "Ignoring base URL override: VCROPPER_PROVIDER=bedrock calls the Bedrock "
                "Runtime endpoint for its region."
            )
        bedrock_config = load_bedrock_config(model=model)
        logger.info("Resolved VLM backend: %s", describe_bedrock_config(bedrock_config))
        return BedrockBackend(bedrock_config)
    openai_config = load_backend_config(api_key=api_key, model=model, base_url=base_url)
    logger.info("Resolved VLM backend: %s", describe_backend_config(openai_config))
    return OpenAICompatBackend(openai_config)
