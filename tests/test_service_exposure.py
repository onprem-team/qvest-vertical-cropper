"""Plaintext-exposure guard for the caller credential.

Callers authenticate with a bearer token in a header, so publishing the API over plain HTTP
on a routable address hands that credential to anything on the network path. The service
cannot see its own exposure — in a container it always binds 0.0.0.0 because Docker
requires it — so the operator declares an origin and this validates the declaration.
"""
from __future__ import annotations

import pytest

from v_cropper.service.app import create_app
from v_cropper.service.config import Settings
from v_cropper.service.security import (
    InsecureExposureError,
    check_public_origin,
    is_loopback_host,
)


class TestLoopbackDetection:
    @pytest.mark.parametrize("host", [
        "127.0.0.1",
        "127.0.0.53",
        "localhost",
        "LOCALHOST",
        "localhost.",
        "::1",
        "::ffff:127.0.0.1",
    ])
    def test_loopback_forms_are_recognised(self, host):
        assert is_loopback_host(host) is True

    @pytest.mark.parametrize("host", [
        "0.0.0.0",
        "10.0.0.5",
        "192.168.1.10",
        "example.com",
        "",
        "   ",
        "127.1",
        "l\u043ecalhost",
    ])
    def test_everything_else_is_not_loopback(self, host):
        """Including forms that look like loopback but are not, which must fail safe."""
        assert is_loopback_host(host) is False

    def test_a_name_that_merely_resolves_to_loopback_is_not_trusted(self):
        """Resolution is attacker-influenceable and can change after the check."""
        assert is_loopback_host("localtest.me") is False


class TestOriginPolicy:
    @pytest.mark.parametrize("origin", [
        "",
        "   ",
        "https://crop.example.com",
        "https://crop.example.com:8443/base",
        "http://127.0.0.1:8090",
        "http://localhost:8090",
        "HTTP://127.0.0.1:8090",
        "http://[::1]:8090",
        "http://[::ffff:127.0.0.1]:8090",
    ])
    def test_safe_origins_are_accepted(self, origin):
        check_public_origin(origin)

    @pytest.mark.parametrize("origin", [
        "http://crop.example.com",
        "http://10.0.0.5:8090",
        "http://0.0.0.0:8090",
        "http://localtest.me",
    ])
    def test_plaintext_on_a_routable_host_is_refused(self, origin):
        with pytest.raises(InsecureExposureError):
            check_public_origin(origin)

    @pytest.mark.parametrize("origin", ["ftp://host/x", "file:///etc/passwd", "host:8090"])
    def test_non_http_schemes_are_refused(self, origin):
        with pytest.raises(InsecureExposureError):
            check_public_origin(origin)

    def test_a_scheme_without_a_hostname_is_refused(self):
        with pytest.raises(InsecureExposureError):
            check_public_origin("http://")

    @pytest.mark.parametrize("origin", [
        "http://user:pass@crop.example.com",
        "https://user:pass@crop.example.com",
    ])
    def test_embedded_credentials_are_refused_even_over_tls(self, origin):
        with pytest.raises(InsecureExposureError):
            check_public_origin(origin)

    def test_the_opt_out_permits_plaintext(self):
        check_public_origin("http://crop.example.com", allow_plaintext=True)

    def test_the_opt_out_does_not_excuse_embedded_credentials(self):
        """The opt-out is about transport, not about putting a password in a config value."""
        with pytest.raises(InsecureExposureError):
            check_public_origin("http://user:pass@crop.example.com", allow_plaintext=True)

    def test_the_opt_out_warns_rather_than_passing_silently(self, caplog):
        with caplog.at_level("WARNING"):
            check_public_origin("http://crop.example.com", allow_plaintext=True)
        assert "CROPPER_ALLOW_PLAINTEXT" in caplog.text

    def test_the_refusal_never_echoes_the_configured_origin(self):
        """Origins are config, but one may carry credentials; never quote it back."""
        secret = "http://user:hunter2@crop.example.com"
        with pytest.raises(InsecureExposureError) as excinfo:
            check_public_origin(secret)
        assert "hunter2" not in str(excinfo.value)
        assert "crop.example.com" not in str(excinfo.value)


def _settings(**overrides) -> Settings:
    fields = dict(
        redis_url="redis://unused",
        api_token="correct-horse-battery-staple-token",
        allowed_hosts=("objects.example.test",),
    )
    fields.update(overrides)
    return Settings(**fields)


class TestStartupRefusal:
    """The point of the check is that the process does not come up at all."""

    def test_an_unsafe_origin_prevents_the_app_from_being_built(self):
        with pytest.raises(InsecureExposureError):
            create_app(_settings(public_origin="http://crop.example.com"))

    def test_the_opt_out_allows_startup(self):
        app = create_app(_settings(public_origin="http://crop.example.com", allow_plaintext=True))
        assert app is not None

    def test_an_undeclared_origin_is_the_shipped_default_and_starts_clean(self):
        """Loopback publish behind an SSH tunnel is encrypted but arrives as plain http."""
        assert create_app(_settings()) is not None

    def test_tls_origin_starts_clean(self):
        assert create_app(_settings(public_origin="https://crop.example.com")) is not None
