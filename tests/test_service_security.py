from __future__ import annotations

import socket

import pytest

from v_cropper.service.security import UnsafeURLError, redact_url, validate_transport_url


def test_redact_url_removes_signature_and_credentials():
    value = "https://user:secret@objects.example.test:9000/a.mp4?X-Amz-Signature=secret#x"
    assert redact_url(value) == "https://objects.example.test:9000/a.mp4"


@pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://objects.example.test/a", "http:///missing"])
def test_rejects_non_http_or_missing_host(url):
    with pytest.raises(UnsafeURLError):
        validate_transport_url(url, allowed_hosts=("objects.example.test",), resolve_dns=False)


def test_requires_exact_or_wildcard_allowlist_match():
    validate_transport_url(
        "https://clips.objects.example.test/a?signature=x",
        allowed_hosts=("*.objects.example.test",),
        resolve_dns=False,
    )
    with pytest.raises(UnsafeURLError):
        validate_transport_url(
            "https://objects.example.test.attacker.test/a",
            allowed_hosts=("objects.example.test",),
            resolve_dns=False,
        )


def test_rejects_url_userinfo():
    with pytest.raises(UnsafeURLError):
        validate_transport_url(
            "https://user:pass@objects.example.test/a",
            allowed_hosts=("objects.example.test",),
            resolve_dns=False,
        )


def test_trailing_dot_host_cannot_bypass_the_allowlist():
    """`example.com.` and `example.com` resolve identically."""
    validate_transport_url(
        "https://objects.example.test./a",
        allowed_hosts=("objects.example.test",),
        resolve_dns=False,
    )


class TestDNSRebindingDefence:
    """An allowlisted name that resolves inward is the interesting SSRF case.

    The host allowlist alone is not enough: an attacker who controls DNS for an allowlisted
    name can point it at 169.254.169.254 or a private address and use the service as a proxy
    into the VPC.
    """

    ALLOWED = ("objects.example.test",)

    def _resolve_to(self, monkeypatch, address):
        monkeypatch.setattr(
            socket, "getaddrinfo",
            lambda *a, **k: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, 443))],
        )

    @pytest.mark.parametrize("address", [
        "127.0.0.1",         # loopback
        "10.0.0.5",          # RFC1918
        "192.168.1.1",
        "172.16.0.1",
        "169.254.169.254",   # cloud instance metadata
        "0.0.0.0",
        "100.64.0.1",        # carrier-grade NAT
    ])
    def test_private_resolution_is_rejected(self, monkeypatch, address):
        self._resolve_to(monkeypatch, address)
        with pytest.raises(UnsafeURLError, match="private or non-routable"):
            validate_transport_url("https://objects.example.test/a", allowed_hosts=self.ALLOWED)

    def test_public_resolution_is_allowed(self, monkeypatch):
        self._resolve_to(monkeypatch, "93.184.216.34")
        assert validate_transport_url(
            "https://objects.example.test/a", allowed_hosts=self.ALLOWED
        ).endswith("/a")

    def test_any_private_answer_rejects_the_whole_name(self, monkeypatch):
        """A name answering with both a public and a private address must not pass."""
        monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443)),
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("169.254.169.254", 443)),
        ])
        with pytest.raises(UnsafeURLError):
            validate_transport_url("https://objects.example.test/a", allowed_hosts=self.ALLOWED)

    def test_unresolvable_host_is_rejected(self, monkeypatch):
        def boom(*a, **k):
            raise socket.gaierror("nope")

        monkeypatch.setattr(socket, "getaddrinfo", boom)
        with pytest.raises(UnsafeURLError, match="could not be resolved"):
            validate_transport_url("https://objects.example.test/a", allowed_hosts=self.ALLOWED)

    def test_private_resolution_allowed_when_explicitly_opted_in(self, monkeypatch):
        """The compose stack points at `minio` on a private Docker network on purpose."""
        self._resolve_to(monkeypatch, "10.0.0.5")
        assert validate_transport_url(
            "https://objects.example.test/a", allowed_hosts=self.ALLOWED, allow_private=True
        )
