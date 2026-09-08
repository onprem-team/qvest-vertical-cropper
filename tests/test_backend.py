"""Bug-hunting unit tests for v_cropper.backend (all offline).

The OpenAI SDK client is replaced with FakeOpenAIClient; no network or key
is required.
"""
from __future__ import annotations

import base64

import httpx
import openai
import pytest

from conftest import FakeOpenAIClient
from v_cropper import backend
from v_cropper.backend import (
    DEFAULT_GEMINI_MODEL,
    DEFAULT_NVIDIA_MODEL,
    GEMINI_COMPAT_BASE_URL,
    NVIDIA_BASE_URL,
    BackendConfig,
    OpenAICompatBackend,
    describe_backend_config,
    frame_to_data_uri,
    image_part,
    load_backend_config,
    make_backend,
    normalize_base_url,
    text_part,
)

VCROPPER_VARS = [
    "VCROPPER_BASE_URL", "VCROPPER_API_KEY", "VCROPPER_MODEL",
    "VCROPPER_TIMEOUT", "VCROPPER_RESPONSE_FORMAT",
    "VCROPPER_MAX_TOKENS", "VCROPPER_EXTRA_BODY", "VCROPPER_EXTRA_HEADERS",
    "GEMINI_API_KEY",
]


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Every test starts from a known-empty provider environment."""
    for name in VCROPPER_VARS:
        monkeypatch.delenv(name, raising=False)


def _status_error(cls, status):
    req = httpx.Request("POST", "http://x/v1/chat/completions")
    resp = httpx.Response(status, request=req)
    return cls("boom", response=resp, body=None)


# --------------------------------------------------------------------------- #
# frame_to_data_uri / content parts
# --------------------------------------------------------------------------- #
class TestDataUri:
    def test_round_trip(self):
        raw = b"\xff\xd8hello-jpeg-bytes\xff\xd9"
        uri = frame_to_data_uri(raw)
        assert uri.startswith("data:image/jpeg;base64,")
        payload = uri.split(",", 1)[1]
        assert base64.b64decode(payload) == raw

    def test_custom_mime(self):
        assert frame_to_data_uri(b"x", mime="image/png").startswith("data:image/png;base64,")

    def test_empty_bytes(self):
        assert frame_to_data_uri(b"") == "data:image/jpeg;base64,"

    def test_image_and_text_parts(self):
        ip = image_part(b"abc")
        assert ip["type"] == "image_url"
        assert ip["image_url"]["url"].startswith("data:image/jpeg;base64,")
        assert text_part("hi") == {"type": "text", "text": "hi"}


# --------------------------------------------------------------------------- #
# load_backend_config precedence matrix
# --------------------------------------------------------------------------- #
class TestLoadConfig:
    def test_nvidia_default_from_vcropper_key(self, monkeypatch):
        monkeypatch.setenv("VCROPPER_API_KEY", "nvapi-xxx")
        cfg = load_backend_config()
        assert cfg.base_url == NVIDIA_BASE_URL
        assert cfg.model == DEFAULT_NVIDIA_MODEL
        assert cfg.api_key == "nvapi-xxx"

    def test_vcropper_overrides_win(self, monkeypatch):
        monkeypatch.setenv("VCROPPER_API_KEY", "k")
        monkeypatch.setenv("VCROPPER_BASE_URL", "https://api.openai.com/v1")
        monkeypatch.setenv("VCROPPER_MODEL", "gpt-4o")
        cfg = load_backend_config()
        assert cfg.base_url == "https://api.openai.com/v1"
        assert cfg.model == "gpt-4o"

    def test_explicit_args_beat_env(self, monkeypatch):
        monkeypatch.setenv("VCROPPER_API_KEY", "envkey")
        monkeypatch.setenv("VCROPPER_MODEL", "envmodel")
        cfg = load_backend_config(api_key="argkey", model="argmodel")
        assert cfg.api_key == "argkey"
        assert cfg.model == "argmodel"

    def test_explicit_base_url_beats_env_and_is_normalized(self, monkeypatch):
        monkeypatch.setenv("VCROPPER_API_KEY", "k")
        monkeypatch.setenv("VCROPPER_BASE_URL", "https://env.example/v1")
        cfg = load_backend_config(base_url=" https://cli.example/v1/ ")
        assert cfg.base_url == "https://cli.example/v1"

    def test_extra_headers_env_parsed(self, monkeypatch):
        monkeypatch.setenv("VCROPPER_API_KEY", "k")
        monkeypatch.setenv("VCROPPER_EXTRA_HEADERS", '{"X-Gateway-Client": "v-cropper"}')
        assert load_backend_config().extra_headers == {"X-Gateway-Client": "v-cropper"}

    def test_invalid_extra_headers_ignored(self, monkeypatch, caplog):
        monkeypatch.setenv("VCROPPER_API_KEY", "k")
        monkeypatch.setenv("VCROPPER_EXTRA_HEADERS", '{"X-Count": 3}')
        assert load_backend_config().extra_headers == {}
        assert "VCROPPER_EXTRA_HEADERS" in caplog.text

    def test_extra_headers_cannot_override_authorization(self, monkeypatch, caplog):
        monkeypatch.setenv("VCROPPER_API_KEY", "k")
        monkeypatch.setenv("VCROPPER_EXTRA_HEADERS", '{"Authorization": "Bearer replacement"}')
        assert load_backend_config().extra_headers == {}
        assert "VCROPPER_EXTRA_HEADERS" in caplog.text

    def test_invalid_json_never_logs_raw_secret_like_value(self, monkeypatch, caplog):
        secret_like_value = "header-secret-should-not-appear"
        monkeypatch.setenv("VCROPPER_API_KEY", "k")
        monkeypatch.setenv("VCROPPER_EXTRA_HEADERS", secret_like_value)
        assert load_backend_config().extra_headers == {}
        assert secret_like_value not in caplog.text

    def test_gemini_fallback(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "gkey")
        cfg = load_backend_config()
        assert cfg.base_url == GEMINI_COMPAT_BASE_URL
        assert cfg.model == DEFAULT_GEMINI_MODEL
        assert cfg.api_key == "gkey"

    def test_gemini_key_but_explicit_base_disables_fallback(self, monkeypatch):
        # Key still comes from GEMINI_API_KEY, but an explicit base_url opts out
        # of the Gemini shortcut -> NVIDIA default model, explicit endpoint.
        monkeypatch.setenv("GEMINI_API_KEY", "gkey")
        monkeypatch.setenv("VCROPPER_BASE_URL", "https://openrouter.ai/api/v1")
        cfg = load_backend_config()
        assert cfg.base_url == "https://openrouter.ai/api/v1"
        assert cfg.model == DEFAULT_NVIDIA_MODEL

    def test_explicit_apikey_arg_disables_gemini_fallback(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "gkey")
        cfg = load_backend_config(api_key="argkey")
        assert cfg.base_url == NVIDIA_BASE_URL
        assert cfg.api_key == "argkey"

    def test_vcropper_key_beats_gemini_for_fallback(self, monkeypatch):
        monkeypatch.setenv("VCROPPER_API_KEY", "vk")
        monkeypatch.setenv("GEMINI_API_KEY", "gk")
        cfg = load_backend_config()
        assert cfg.api_key == "vk"
        assert cfg.base_url == NVIDIA_BASE_URL

    def test_model_env_respected_in_gemini_fallback(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "gkey")
        monkeypatch.setenv("VCROPPER_MODEL", "gemini-3-pro")
        cfg = load_backend_config()
        assert cfg.base_url == GEMINI_COMPAT_BASE_URL
        assert cfg.model == "gemini-3-pro"

    def test_missing_key_raises(self):
        with pytest.raises(RuntimeError, match="No VLM API key"):
            load_backend_config()

    def test_response_format_env(self, monkeypatch):
        monkeypatch.setenv("VCROPPER_API_KEY", "k")
        monkeypatch.setenv("VCROPPER_RESPONSE_FORMAT", "json_object")
        assert load_backend_config().response_format == {"type": "json_object"}

    def test_bad_timeout_uses_default(self, monkeypatch):
        monkeypatch.setenv("VCROPPER_API_KEY", "k")
        monkeypatch.setenv("VCROPPER_TIMEOUT", "abc")
        assert load_backend_config().timeout == 120.0

    def test_max_tokens_env_parsed(self, monkeypatch):
        monkeypatch.setenv("VCROPPER_API_KEY", "k")
        monkeypatch.setenv("VCROPPER_MAX_TOKENS", "200")
        assert load_backend_config().max_tokens == 200

    def test_max_tokens_default_none(self, monkeypatch):
        monkeypatch.setenv("VCROPPER_API_KEY", "k")
        assert load_backend_config().max_tokens is None

    def test_bad_max_tokens_ignored(self, monkeypatch):
        monkeypatch.setenv("VCROPPER_API_KEY", "k")
        monkeypatch.setenv("VCROPPER_MAX_TOKENS", "notanint")
        assert load_backend_config().max_tokens is None

    def test_extra_body_env_parsed(self, monkeypatch):
        monkeypatch.setenv("VCROPPER_API_KEY", "k")
        monkeypatch.setenv("VCROPPER_EXTRA_BODY", '{"reasoning_effort": "low"}')
        assert load_backend_config().extra_body == {"reasoning_effort": "low"}

    def test_bad_extra_body_ignored(self, monkeypatch):
        monkeypatch.setenv("VCROPPER_API_KEY", "k")  # NVIDIA path -> no default extra_body
        monkeypatch.setenv("VCROPPER_EXTRA_BODY", "{not json")
        assert load_backend_config().extra_body is None

    def test_non_object_extra_body_ignored(self, monkeypatch):
        monkeypatch.setenv("VCROPPER_API_KEY", "k")
        monkeypatch.setenv("VCROPPER_EXTRA_BODY", "[1, 2, 3]")
        assert load_backend_config().extra_body is None

    def test_gemini_fallback_disables_thinking(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "gkey")
        cfg = load_backend_config()
        assert cfg.extra_body == {"reasoning_effort": "none"}

    def test_nvidia_default_has_no_extra_body(self, monkeypatch):
        monkeypatch.setenv("VCROPPER_API_KEY", "nvapi-x")
        assert load_backend_config().extra_body is None

    def test_explicit_extra_body_overrides_gemini_default(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "gkey")
        monkeypatch.setenv("VCROPPER_EXTRA_BODY", '{"reasoning_effort": "high"}')
        assert load_backend_config().extra_body == {"reasoning_effort": "high"}

    def test_url_validation_rejects_credentials(self):
        with pytest.raises(ValueError, match="must not contain credentials"):
            normalize_base_url("https://user:password@example.test/v1")

    @pytest.mark.parametrize(
        ("value", "message"),
        [
            ("", "must not be empty"),
            ("ftp://example.test/v1", "absolute http"),
            ("https://example.test/v1?token=secret", "must not contain a query"),
        ],
    )
    def test_url_validation_rejects_invalid_endpoint_shapes(self, value, message):
        with pytest.raises(ValueError, match=message):
            normalize_base_url(value)

    def test_url_without_v1_warns_but_is_allowed(self, caplog):
        assert normalize_base_url("http://127.0.0.1:8000") == "http://127.0.0.1:8000"
        assert "does not contain '/v1'" in caplog.text

    def test_describe_config_never_contains_secret_values(self):
        cfg = BackendConfig(
            base_url="https://gateway.example/v1",
            api_key="super-secret-key",
            model="vision-model",
            extra_headers={"X-Provider-Key": "also-secret", "X-Client": "v-cropper"},
        )
        desc = describe_backend_config(cfg)
        assert desc == {
            "base_url": "https://gateway.example/v1",
            "model": "vision-model",
            "extra_header_names": ["X-Client", "X-Provider-Key"],
        }
        assert "super-secret-key" not in repr(desc)
        assert "also-secret" not in repr(desc)


# --------------------------------------------------------------------------- #
# OpenAICompatBackend.complete
# --------------------------------------------------------------------------- #
class TestComplete:
    def _backend(self, **cfg):
        base = dict(base_url="http://x/v1", api_key="k", model="m")
        base.update(cfg)
        return OpenAICompatBackend(BackendConfig(**base))

    def test_returns_content_and_records_usage(self, monkeypatch):
        ctor = FakeOpenAIClient.factory(content='{"x": 5}')
        monkeypatch.setattr(openai, "OpenAI", ctor)
        be = self._backend()
        out = be.complete([{"role": "user", "content": "hi"}], temperature=0.1)
        assert out == '{"x": 5}'
        assert be.last_usage == {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        sent = ctor.instance.calls[0]
        assert sent["model"] == "m"
        assert sent["temperature"] == 0.1
        assert "response_format" not in sent

    def test_client_constructed_with_expected_kwargs(self, monkeypatch):
        ctor = FakeOpenAIClient.factory()
        monkeypatch.setattr(openai, "OpenAI", ctor)
        be = self._backend(timeout=42.0)
        be.complete([{"role": "user", "content": "hi"}])
        init = ctor.instance.init_kwargs
        assert init["base_url"] == "http://x/v1"
        assert init["api_key"] == "k"
        assert init["timeout"] == 42.0
        assert init["max_retries"] == 0

    def test_response_format_from_config(self, monkeypatch):
        ctor = FakeOpenAIClient.factory()
        monkeypatch.setattr(openai, "OpenAI", ctor)
        be = self._backend(response_format={"type": "json_object"})
        be.complete([{"role": "user", "content": "hi"}])
        assert ctor.instance.calls[0]["response_format"] == {"type": "json_object"}

    def test_response_format_arg_overrides(self, monkeypatch):
        ctor = FakeOpenAIClient.factory()
        monkeypatch.setattr(openai, "OpenAI", ctor)
        be = self._backend()
        be.complete([{"role": "user", "content": "x"}], response_format={"type": "json_schema"})
        assert ctor.instance.calls[0]["response_format"] == {"type": "json_schema"}

    def test_extra_headers_passed(self, monkeypatch):
        ctor = FakeOpenAIClient.factory()
        monkeypatch.setattr(openai, "OpenAI", ctor)
        be = self._backend(extra_headers={"HTTP-Referer": "https://ex.com"})
        be.complete([{"role": "user", "content": "x"}])
        assert ctor.instance.calls[0]["extra_headers"] == {"HTTP-Referer": "https://ex.com"}

    def test_none_content_becomes_empty_string(self, monkeypatch):
        ctor = FakeOpenAIClient.factory(content=None)
        monkeypatch.setattr(openai, "OpenAI", ctor)
        assert self._backend().complete([{"role": "user", "content": "x"}]) == ""

    def test_usage_none_resets_to_zero(self, monkeypatch):
        # A response with usage=None must yield zeroed tallies, not stale values.
        class _Msg:
            content = "ok"

        class _Choice:
            message = _Msg()

        class _Resp:
            choices = [_Choice()]
            usage = None

        class _Completions:
            def create(self, **kwargs):
                return _Resp()

        class _Client:
            def __init__(self, **kwargs):
                self.chat = type("C", (), {"completions": _Completions()})()

        monkeypatch.setattr(openai, "OpenAI", _Client)
        be = self._backend()
        be.last_usage = {"prompt_tokens": 99, "completion_tokens": 99, "total_tokens": 99}
        be.complete([{"role": "user", "content": "x"}])
        assert be.last_usage == {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def test_model_property(self):
        be = self._backend(model="foo")
        assert be.model == "foo"

    def test_cache_identity_includes_response_config_but_not_api_key(self):
        be = self._backend(
            api_key="must-not-be-in-cache-identity",
            base_url="https://local.example/v1",
            model="foo",
            response_format={"type": "json_object"},
            extra_headers={"X-Route": "a"},
            max_tokens=64,
            extra_body={"reasoning_effort": "none"},
        )
        assert be.cache_identity == {
            "base_url": "https://local.example/v1",
            "model": "foo",
            "response_format": {"type": "json_object"},
            "extra_headers": {"X-Route": "a"},
            "max_tokens": 64,
            "extra_body": {"reasoning_effort": "none"},
        }
        assert "must-not-be-in-cache-identity" not in repr(be.cache_identity)


# --------------------------------------------------------------------------- #
# Generation params: max_tokens + extra_body passthrough (Phase 1b)
# --------------------------------------------------------------------------- #
class TestGenerationParams:
    def _backend(self, **cfg):
        base = dict(base_url="http://x/v1", api_key="k", model="m")
        base.update(cfg)
        return OpenAICompatBackend(BackendConfig(**base))

    def test_max_tokens_arg_forwarded(self, monkeypatch):
        ctor = FakeOpenAIClient.factory()
        monkeypatch.setattr(openai, "OpenAI", ctor)
        self._backend().complete([{"role": "user", "content": "x"}], max_tokens=200)
        assert ctor.instance.calls[0]["max_tokens"] == 200

    def test_max_tokens_omitted_when_none(self, monkeypatch):
        ctor = FakeOpenAIClient.factory()
        monkeypatch.setattr(openai, "OpenAI", ctor)
        self._backend().complete([{"role": "user", "content": "x"}])
        assert "max_tokens" not in ctor.instance.calls[0]

    def test_config_max_tokens_used(self, monkeypatch):
        ctor = FakeOpenAIClient.factory()
        monkeypatch.setattr(openai, "OpenAI", ctor)
        self._backend(max_tokens=99).complete([{"role": "user", "content": "x"}])
        assert ctor.instance.calls[0]["max_tokens"] == 99

    def test_arg_max_tokens_overrides_config(self, monkeypatch):
        ctor = FakeOpenAIClient.factory()
        monkeypatch.setattr(openai, "OpenAI", ctor)
        self._backend(max_tokens=99).complete([{"role": "user", "content": "x"}], max_tokens=5)
        assert ctor.instance.calls[0]["max_tokens"] == 5

    def test_extra_body_arg_forwarded(self, monkeypatch):
        ctor = FakeOpenAIClient.factory()
        monkeypatch.setattr(openai, "OpenAI", ctor)
        self._backend().complete([{"role": "user", "content": "x"}],
                                 extra_body={"reasoning_effort": "none"})
        assert ctor.instance.calls[0]["extra_body"] == {"reasoning_effort": "none"}

    def test_extra_body_omitted_when_none(self, monkeypatch):
        ctor = FakeOpenAIClient.factory()
        monkeypatch.setattr(openai, "OpenAI", ctor)
        self._backend().complete([{"role": "user", "content": "x"}])
        assert "extra_body" not in ctor.instance.calls[0]

    def test_config_extra_body_used(self, monkeypatch):
        ctor = FakeOpenAIClient.factory()
        monkeypatch.setattr(openai, "OpenAI", ctor)
        self._backend(extra_body={"reasoning_effort": "none"}).complete(
            [{"role": "user", "content": "x"}])
        assert ctor.instance.calls[0]["extra_body"] == {"reasoning_effort": "none"}

    def test_extra_body_merge_per_call_wins(self, monkeypatch):
        ctor = FakeOpenAIClient.factory()
        monkeypatch.setattr(openai, "OpenAI", ctor)
        self._backend(extra_body={"reasoning_effort": "none", "keep": 1}).complete(
            [{"role": "user", "content": "x"}], extra_body={"reasoning_effort": "high"})
        sent = ctor.instance.calls[0]["extra_body"]
        assert sent == {"reasoning_effort": "high", "keep": 1}

    def test_usage_still_recorded_with_params(self, monkeypatch):
        ctor = FakeOpenAIClient.factory(content="ok")
        monkeypatch.setattr(openai, "OpenAI", ctor)
        be = self._backend()
        be.complete([{"role": "user", "content": "x"}], max_tokens=50,
                    extra_body={"reasoning_effort": "none"})
        assert be.usage_totals["api_calls"] == 1
        assert be.last_usage["total_tokens"] == 15


# --------------------------------------------------------------------------- #
# Retry / backoff
# --------------------------------------------------------------------------- #
class TestRetry:
    def _backend(self, **cfg):
        base = dict(base_url="http://x/v1", api_key="k", model="m", max_retries=2, backoff_base=0.01)
        base.update(cfg)
        return OpenAICompatBackend(BackendConfig(**base))

    def test_retries_then_succeeds(self, monkeypatch):
        monkeypatch.setattr(backend.time, "sleep", lambda s: None)
        err = _status_error(openai.RateLimitError, 429)
        ctor = FakeOpenAIClient.factory(content="ok", raise_seq=[err, None])
        monkeypatch.setattr(openai, "OpenAI", ctor)
        out = self._backend().complete([{"role": "user", "content": "x"}])
        assert out == "ok"
        assert len(ctor.instance.calls) == 2

    def test_exhausts_retries_and_raises(self, monkeypatch):
        monkeypatch.setattr(backend.time, "sleep", lambda s: None)
        errs = [_status_error(openai.RateLimitError, 429) for _ in range(3)]
        ctor = FakeOpenAIClient.factory(content="ok", raise_seq=errs)
        monkeypatch.setattr(openai, "OpenAI", ctor)
        with pytest.raises(openai.RateLimitError):
            self._backend(max_retries=2).complete([{"role": "user", "content": "x"}])
        assert len(ctor.instance.calls) == 3  # 1 initial + 2 retries

    def test_non_retryable_raises_immediately(self, monkeypatch):
        monkeypatch.setattr(backend.time, "sleep", lambda s: None)
        err = _status_error(openai.BadRequestError, 400)
        ctor = FakeOpenAIClient.factory(content="ok", raise_seq=[err, None])
        monkeypatch.setattr(openai, "OpenAI", ctor)
        with pytest.raises(openai.BadRequestError):
            self._backend().complete([{"role": "user", "content": "x"}])
        assert len(ctor.instance.calls) == 1  # no retry


# --------------------------------------------------------------------------- #
# usage_totals accumulation (real backend, thread-safe path)
# --------------------------------------------------------------------------- #
class TestUsageTotals:
    def _backend(self, **cfg):
        base = dict(base_url="http://x/v1", api_key="k", model="m", max_retries=2, backoff_base=0.01)
        base.update(cfg)
        return OpenAICompatBackend(BackendConfig(**base))

    def test_totals_accumulate_across_calls(self, monkeypatch):
        ctor = FakeOpenAIClient.factory(content="ok")  # FakeUsage -> 10/5/15
        monkeypatch.setattr(openai, "OpenAI", ctor)
        be = self._backend()
        be.complete([{"role": "user", "content": "x"}])
        be.complete([{"role": "user", "content": "y"}])
        assert be.usage_totals["api_calls"] == 2
        assert be.usage_totals["total_tokens"] == 30
        assert be.usage_totals["prompt_tokens"] == 20

    def test_api_calls_counted_once_despite_retry(self, monkeypatch):
        # A retried-then-succeeded call must count as exactly ONE api_call.
        monkeypatch.setattr(backend.time, "sleep", lambda s: None)
        err = _status_error(openai.RateLimitError, 429)
        ctor = FakeOpenAIClient.factory(content="ok", raise_seq=[err, None])
        monkeypatch.setattr(openai, "OpenAI", ctor)
        be = self._backend()
        be.complete([{"role": "user", "content": "x"}])
        assert len(ctor.instance.calls) == 2      # one retry happened
        assert be.usage_totals["api_calls"] == 1  # but usage recorded once

    def test_totals_thread_safe_under_concurrency(self, monkeypatch):
        import concurrent.futures

        ctor = FakeOpenAIClient.factory(content="ok")
        monkeypatch.setattr(openai, "OpenAI", ctor)
        be = self._backend()
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(
                lambda _: be.complete([{"role": "user", "content": "x"}]),
                range(200),
            ))
        assert be.usage_totals["api_calls"] == 200
        assert be.usage_totals["total_tokens"] == 200 * 15


# --------------------------------------------------------------------------- #
# make_backend factory
# --------------------------------------------------------------------------- #
class TestMakeBackend:
    def test_builds_from_env(self, monkeypatch):
        monkeypatch.setenv("VCROPPER_API_KEY", "k")
        be = make_backend()
        assert isinstance(be, OpenAICompatBackend)
        assert be.model == DEFAULT_NVIDIA_MODEL

    def test_uses_supplied_config(self):
        cfg = BackendConfig(base_url="http://x/v1", api_key="k", model="m")
        be = make_backend(config=cfg)
        assert be.config is cfg
