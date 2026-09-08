"""The make_backend dispatch seam created by merging the router work with Bedrock.

Two branches were developed independently: one added `base_url` plumbing so a run can be
pointed at a gateway, the other added a second provider. `make_backend` is where they meet,
and a bad resolution here is silent — every combination still returns a usable backend, just
possibly the wrong one, aimed at the wrong endpoint.
"""
from __future__ import annotations

import pytest
from botocore.exceptions import (
    BotoCoreError,
    ClientError,
    EndpointConnectionError,
    ReadTimeoutError,
)

from v_cropper.backend import (
    DEFAULT_BEDROCK_MODEL,
    DEFAULT_GEMINI_MODEL,
    DEFAULT_NVIDIA_MODEL,
    GEMINI_COMPAT_BASE_URL,
    NVIDIA_BASE_URL,
    BackendConfig,
    BedrockBackend,
    BedrockConfig,
    OpenAICompatBackend,
    _bedrock_messages,
    _is_retryable_bedrock_error,
    _is_temperature_rejection,
    describe_bedrock_config,
    make_backend,
)

ENV_VARS = [
    "VCROPPER_PROVIDER", "VCROPPER_API_KEY", "VCROPPER_BASE_URL", "VCROPPER_MODEL",
    "GEMINI_API_KEY", "BEDROCK_REGION", "AWS_REGION", "BEDROCK_VLM_MODEL_ID",
]


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)


class TestProviderDispatch:
    def test_defaults_to_the_openai_compatible_backend(self, monkeypatch):
        monkeypatch.setenv("VCROPPER_API_KEY", "k")
        backend = make_backend()
        assert isinstance(backend, OpenAICompatBackend)
        assert backend.config.base_url == NVIDIA_BASE_URL
        assert backend.model == DEFAULT_NVIDIA_MODEL

    def test_provider_env_selects_bedrock(self, monkeypatch):
        monkeypatch.setenv("VCROPPER_PROVIDER", "bedrock")
        monkeypatch.setenv("BEDROCK_REGION", "us-east-1")
        backend = make_backend()
        assert isinstance(backend, BedrockBackend)
        assert backend.model == DEFAULT_BEDROCK_MODEL

    def test_bedrock_needs_no_vlm_api_key(self, monkeypatch):
        """The OpenAI-side key check must not gate the Bedrock path."""
        monkeypatch.setenv("VCROPPER_PROVIDER", "bedrock")
        monkeypatch.setenv("AWS_REGION", "eu-west-1")
        assert isinstance(make_backend(), BedrockBackend)

    @pytest.mark.parametrize("provider", ["BEDROCK", " bedrock ", "Bedrock"])
    def test_provider_is_case_and_space_insensitive(self, monkeypatch, provider):
        monkeypatch.setenv("VCROPPER_PROVIDER", provider)
        monkeypatch.setenv("BEDROCK_REGION", "us-east-1")
        assert isinstance(make_backend(), BedrockBackend)

    def test_unknown_provider_is_rejected(self, monkeypatch):
        monkeypatch.setenv("VCROPPER_PROVIDER", "anthropic-direct")
        monkeypatch.setenv("VCROPPER_API_KEY", "k")
        with pytest.raises(RuntimeError, match="openai' or 'bedrock"):
            make_backend()

    def test_explicit_config_object_wins_over_the_provider_env(self, monkeypatch):
        """A caller passing a config has already decided; env must not override it."""
        monkeypatch.setenv("VCROPPER_PROVIDER", "bedrock")
        monkeypatch.setenv("BEDROCK_REGION", "us-east-1")
        backend = make_backend(config=BackendConfig(base_url="https://x.test/v1", api_key="k", model="m"))
        assert isinstance(backend, OpenAICompatBackend)

    def test_explicit_bedrock_config_is_used_directly(self, monkeypatch):
        monkeypatch.setenv("VCROPPER_PROVIDER", "openai")
        monkeypatch.setenv("VCROPPER_API_KEY", "k")
        backend = make_backend(config=BedrockConfig(region="us-west-2", model="m"))
        assert isinstance(backend, BedrockBackend)
        assert backend.config.region == "us-west-2"


class TestBaseUrlPlumbing:
    """`--base-url` is how a run is pointed at a router; losing it would be silent."""

    def test_explicit_base_url_reaches_the_client(self, monkeypatch):
        monkeypatch.setenv("VCROPPER_API_KEY", "k")
        backend = make_backend(base_url="https://router.test/v1")
        assert backend.config.base_url == "https://router.test/v1"

    def test_explicit_base_url_beats_env(self, monkeypatch):
        monkeypatch.setenv("VCROPPER_API_KEY", "k")
        monkeypatch.setenv("VCROPPER_BASE_URL", "https://env.test/v1")
        assert make_backend(base_url="https://arg.test/v1").config.base_url == "https://arg.test/v1"

    def test_explicit_base_url_suppresses_the_gemini_fallback(self, monkeypatch):
        """A GEMINI_API_KEY must not silently redirect an explicitly targeted endpoint."""
        monkeypatch.setenv("GEMINI_API_KEY", "k")
        backend = make_backend(base_url="https://router.test/v1")
        assert backend.config.base_url == "https://router.test/v1"
        assert backend.model == DEFAULT_NVIDIA_MODEL

    def test_gemini_fallback_still_applies_without_an_explicit_base_url(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "k")
        backend = make_backend()
        assert backend.config.base_url == GEMINI_COMPAT_BASE_URL
        assert backend.model == DEFAULT_GEMINI_MODEL

    def test_base_url_under_bedrock_is_ignored_with_a_warning(self, monkeypatch, caplog):
        """Silently dropping it would misreport which endpoint served the run."""
        monkeypatch.setenv("VCROPPER_PROVIDER", "bedrock")
        monkeypatch.setenv("BEDROCK_REGION", "us-east-1")
        with caplog.at_level("WARNING"):
            backend = make_backend(base_url="https://router.test/v1")
        assert isinstance(backend, BedrockBackend)
        assert any("Ignoring base URL" in record.message for record in caplog.records)


class TestSafeLogging:
    def test_resolution_is_logged_for_both_providers(self, monkeypatch, caplog):
        monkeypatch.setenv("VCROPPER_API_KEY", "super-secret-key")
        with caplog.at_level("INFO"):
            make_backend()
        logged = " ".join(record.getMessage() for record in caplog.records)
        assert "Resolved VLM backend" in logged
        assert "super-secret-key" not in logged

    def test_bedrock_description_carries_no_credentials(self):
        described = describe_bedrock_config(BedrockConfig(region="us-east-1", model="m"))
        assert described == {"provider": "bedrock", "region": "us-east-1", "model": "m"}


class TestBedrockTranslationEdges:
    def test_system_message_may_not_carry_an_image(self):
        messages = [{"role": "system", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,QQ=="}}]}]
        with pytest.raises(ValueError, match="system messages may contain text only"):
            _bedrock_messages(messages)

    def test_temperature_rejection_is_recognised(self):
        exc = ClientError(
            {"Error": {"Code": "ValidationException", "Message": "temperature is not supported"}},
            "Converse")
        assert _is_temperature_rejection(exc) is True

    def test_unrelated_validation_error_is_not_a_temperature_rejection(self):
        exc = ClientError(
            {"Error": {"Code": "ValidationException", "Message": "model id is invalid"}}, "Converse")
        assert _is_temperature_rejection(exc) is False

    @pytest.mark.parametrize("code,expected", [
        ("ThrottlingException", True),
        ("ModelNotReadyException", True),
        ("AccessDeniedException", False),
        ("ValidationException", False),
    ])
    def test_retryable_classification(self, code, expected):
        exc = ClientError({"Error": {"Code": code, "Message": ""}}, "Converse")
        assert _is_retryable_bedrock_error(exc) is expected

    def test_unrelated_exception_is_not_retryable(self):
        assert _is_retryable_bedrock_error(ValueError("nope")) is False

    def test_transport_errors_are_retryable(self):
        """A dropped connection to Bedrock is worth another attempt; a bad request is not."""
        assert _is_retryable_bedrock_error(ReadTimeoutError(endpoint_url="https://bedrock")) is True
        assert _is_retryable_bedrock_error(
            EndpointConnectionError(endpoint_url="https://bedrock")) is True

    def test_transport_errors_are_retried_then_surface(self):
        class Client:
            def __init__(self):
                self.calls = 0

            def converse(self, **_kwargs):
                self.calls += 1
                raise ReadTimeoutError(endpoint_url="https://bedrock")

        client = Client()
        backend = BedrockBackend(
            BedrockConfig(region="us-east-1", model="m", max_retries=2, backoff_base=0), client)
        with pytest.raises(ReadTimeoutError):
            backend.complete([{"role": "user", "content": "hi"}])
        assert client.calls == 3

    def test_botocore_errors_are_not_retried(self):
        """A malformed request will fail identically on every attempt."""
        class Client:
            def __init__(self):
                self.calls = 0

            def converse(self, **_kwargs):
                self.calls += 1
                raise BotoCoreError()

        client = Client()
        backend = BedrockBackend(BedrockConfig(region="us-east-1", model="m", backoff_base=0), client)
        with pytest.raises(BotoCoreError):
            backend.complete([{"role": "user", "content": "hi"}])
        assert client.calls == 1

    def test_retries_are_exhausted_then_the_error_surfaces(self):
        class Client:
            def __init__(self):
                self.calls = 0

            def converse(self, **_kwargs):
                self.calls += 1
                raise ClientError({"Error": {"Code": "ThrottlingException", "Message": ""}}, "Converse")

        client = Client()
        backend = BedrockBackend(
            BedrockConfig(region="us-east-1", model="m", max_retries=2, backoff_base=0), client)
        with pytest.raises(ClientError):
            backend.complete([{"role": "user", "content": "hi"}])
        assert client.calls == 3
