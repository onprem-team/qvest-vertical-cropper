"""Offline tests for the native AWS Bedrock Converse backend."""

from __future__ import annotations

import concurrent.futures
import copy

import boto3
import pytest
from botocore.exceptions import ClientError, ReadTimeoutError

from v_cropper import backend
from v_cropper.backend import (
    DEFAULT_BEDROCK_MODEL,
    BedrockBackend,
    BedrockConfig,
    _bedrock_messages,
    frame_to_data_uri,
    load_bedrock_config,
    make_backend,
)

BEDROCK_ENV = (
    "VCROPPER_PROVIDER",
    "VCROPPER_MODEL",
    "BEDROCK_VLM_MODEL_ID",
    "BEDROCK_REGION",
    "AWS_REGION",
    "VCROPPER_TIMEOUT",
    "VCROPPER_MAX_TOKENS",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in BEDROCK_ENV:
        monkeypatch.delenv(name, raising=False)


class FakeBedrockClient:
    def __init__(self, responses=None, errors=None):
        self.responses = list(responses or [])
        self.errors = list(errors or [])
        self.calls = []

    def converse(self, **kwargs):
        self.calls.append(copy.deepcopy(kwargs))
        if self.errors:
            error = self.errors.pop(0)
            if error is not None:
                raise error
        if self.responses:
            return self.responses.pop(0)
        return {
            "output": {"message": {"content": [{"text": "ok"}]}},
            "usage": {"inputTokens": 10, "outputTokens": 5, "totalTokens": 15},
        }


def _client_error(code: str, message: str) -> ClientError:
    return ClientError({"Error": {"Code": code, "Message": message}}, "Converse")


def _backend(client=None, **overrides):
    values = {"region": "us-west-2", "model": "model-id", "backoff_base": 0}
    values.update(overrides)
    return BedrockBackend(BedrockConfig(**values), client=client or FakeBedrockClient())


class TestBedrockConfig:
    def test_region_and_model_precedence(self, monkeypatch):
        monkeypatch.setenv("BEDROCK_REGION", "us-east-1")
        monkeypatch.setenv("AWS_REGION", "us-west-1")
        monkeypatch.setenv("VCROPPER_MODEL", "vcropper-model")
        monkeypatch.setenv("BEDROCK_VLM_MODEL_ID", "bedrock-model")
        assert load_bedrock_config(model="cli-model").region == "us-east-1"
        assert load_bedrock_config(model="cli-model").model == "cli-model"
        assert load_bedrock_config().model == "vcropper-model"
        monkeypatch.delenv("VCROPPER_MODEL")
        assert load_bedrock_config().model == "bedrock-model"

    def test_fallback_model_and_generation_settings(self, monkeypatch):
        monkeypatch.setenv("AWS_REGION", "eu-west-1")
        monkeypatch.setenv("VCROPPER_TIMEOUT", "9.5")
        monkeypatch.setenv("VCROPPER_MAX_TOKENS", "123")
        config = load_bedrock_config()
        assert config.model == DEFAULT_BEDROCK_MODEL
        assert config.timeout == 9.5
        assert config.max_tokens == 123

    def test_region_is_required(self):
        with pytest.raises(RuntimeError, match="Bedrock region"):
            load_bedrock_config()

    def test_factory_selects_bedrock_without_api_key(self, monkeypatch):
        monkeypatch.setenv("VCROPPER_PROVIDER", "bedrock")
        monkeypatch.setenv("BEDROCK_REGION", "us-east-2")
        assert isinstance(make_backend(), BedrockBackend)

    def test_invalid_provider_rejected(self, monkeypatch):
        monkeypatch.setenv("VCROPPER_PROVIDER", "other")
        with pytest.raises(RuntimeError, match="openai.*bedrock"):
            make_backend()


class TestMessageTranslation:
    def test_text_images_roles_and_system(self):
        raw = b"\x89PNG\r\n"
        messages, system = _bedrock_messages(
            [
                {"role": "system", "content": "Return JSON"},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": frame_to_data_uri(raw, "image/png")}},
                        {"type": "text", "text": "Point"},
                    ],
                },
                {"role": "assistant", "content": "Prior answer"},
            ]
        )
        assert system == [{"text": "Return JSON"}]
        assert messages[0]["role"] == "user"
        assert messages[0]["content"][0] == {
            "image": {"format": "png", "source": {"bytes": raw}},
        }
        assert messages[0]["content"][1] == {"text": "Point"}
        assert messages[1] == {"role": "assistant", "content": [{"text": "Prior answer"}]}

    @pytest.mark.parametrize(
        "url",
        ["https://example/image.jpg", "data:image/tiff;base64,eA==", "data:image/png;base64,!"],
    )
    def test_bad_image_uri_rejected(self, url):
        with pytest.raises(ValueError):
            _bedrock_messages(
                [
                    {
                        "role": "user",
                        "content": [{"type": "image_url", "image_url": {"url": url}}],
                    }
                ]
            )

    def test_unknown_role_and_part_rejected(self):
        with pytest.raises(ValueError, match="role"):
            _bedrock_messages([{"role": "tool", "content": "x"}])
        with pytest.raises(ValueError, match="content part"):
            _bedrock_messages([{"role": "user", "content": [{"type": "audio"}]}])


class TestBedrockComplete:
    def test_request_response_usage_and_parameters(self):
        client = FakeBedrockClient(
            responses=[
                {
                    "output": {"message": {"content": [{"text": '{"x":'}, {"text": " 4}"}]}},
                    "usage": {"inputTokens": 7, "outputTokens": 3, "totalTokens": 10},
                }
            ]
        )
        be = _backend(client, max_tokens=99)
        result = be.complete(
            [{"role": "system", "content": "JSON"}, {"role": "user", "content": "point"}],
            temperature=0.2,
            max_tokens=12,
            response_format={"type": "json_object"},
            extra_body={"top_k": 20},
        )
        assert result == '{"x": 4}'
        assert client.calls[0] == {
            "modelId": "model-id",
            "messages": [{"role": "user", "content": [{"text": "point"}]}],
            "system": [{"text": "JSON"}],
            "inferenceConfig": {"temperature": 0.2, "maxTokens": 12},
            "additionalModelRequestFields": {"top_k": 20},
        }
        assert be.last_usage == {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}
        assert be.usage_totals["api_calls"] == 1

    def test_config_max_tokens_and_empty_response(self):
        client = FakeBedrockClient(responses=[{"output": {}, "usage": {}}])
        be = _backend(client, max_tokens=44)
        assert be.complete([{"role": "user", "content": "x"}]) == ""
        assert client.calls[0]["inferenceConfig"]["maxTokens"] == 44
        assert be.last_usage == {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def test_temperature_validation_retries_once_without_it(self):
        client = FakeBedrockClient(
            errors=[
                _client_error("ValidationException", "temperature is not supported"),
                None,
            ]
        )
        assert _backend(client, max_retries=0).complete([{"role": "user", "content": "x"}]) == "ok"
        assert len(client.calls) == 2
        assert "temperature" in client.calls[0]["inferenceConfig"]
        assert "temperature" not in client.calls[1]["inferenceConfig"]

    @pytest.mark.parametrize(
        "error",
        [
            _client_error("ThrottlingException", "slow down"),
            ReadTimeoutError(endpoint_url="https://bedrock.test"),
        ],
    )
    def test_transient_errors_retry(self, monkeypatch, error):
        monkeypatch.setattr(backend.time, "sleep", lambda _: None)
        client = FakeBedrockClient(errors=[error, None])
        assert _backend(client, max_retries=1).complete([{"role": "user", "content": "x"}]) == "ok"
        assert len(client.calls) == 2
        assert _backend(client).model == "model-id"

    def test_non_retryable_error_is_immediate(self):
        client = FakeBedrockClient(errors=[_client_error("AccessDeniedException", "no"), None])
        with pytest.raises(ClientError):
            _backend(client).complete([{"role": "user", "content": "x"}])
        assert len(client.calls) == 1

    def test_usage_totals_are_thread_safe(self):
        be = _backend(FakeBedrockClient())
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(lambda _: be.complete([{"role": "user", "content": "x"}]), range(100)))
        assert be.usage_totals == {
            "prompt_tokens": 1000,
            "completion_tokens": 500,
            "total_tokens": 1500,
            "api_calls": 100,
        }

    def test_lazy_client_uses_region_timeout_and_sdk_retries_disabled(self, monkeypatch):
        calls = []
        fake = FakeBedrockClient()

        def client_factory(*args, **kwargs):
            calls.append((args, kwargs))
            return fake

        monkeypatch.setattr(boto3, "client", client_factory)
        be = _backend(client=None, timeout=17)
        # _backend supplies a fake by default, so construct directly for the lazy-client path.
        be = BedrockBackend(BedrockConfig(region="ap-southeast-2", model="m", timeout=17))
        be.complete([{"role": "user", "content": "x"}])
        assert calls[0][0] == ("bedrock-runtime",)
        assert calls[0][1]["region_name"] == "ap-southeast-2"
        sdk_config = calls[0][1]["config"]
        assert sdk_config.connect_timeout == 17
        assert sdk_config.read_timeout == 17
        assert sdk_config.retries["total_max_attempts"] == 1
