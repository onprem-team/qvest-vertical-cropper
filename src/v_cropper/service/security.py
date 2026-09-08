"""Caller authentication, signed-URL validation, and safe redaction."""
from __future__ import annotations

import hmac
import ipaddress
import logging
import socket
from urllib.parse import urljoin, urlsplit, urlunsplit

import httpx

logger = logging.getLogger(__name__)

# ffmpeg fetches the source itself and follows redirects, and the http protocol exposes no
# option to disable that, so the chain has to be walked here instead.
MAX_REDIRECT_HOPS = 5

# Shorter secrets are brute-forcible; the service warns rather than refusing so an
# operator is never locked out of their own deployment by a policy change.
MIN_RECOMMENDED_TOKEN_LENGTH = 24

# Hostnames that mean "this machine" without a DNS lookup. A name that merely *resolves*
# to loopback is deliberately not accepted: resolution is attacker-influenceable and can
# change after the check, and treating it as safe is the failure direction that matters.
LOOPBACK_HOSTNAMES = frozenset({"localhost"})


class UnsafeURLError(ValueError):
    """A transport URL violates the service SSRF policy."""


class InsecureExposureError(ValueError):
    """The configured public origin would carry caller credentials in plaintext."""


def is_loopback_host(host: str) -> bool:
    """True for literal loopback addresses and ``localhost``, else False."""
    candidate = host.strip().rstrip(".").lower()
    if not candidate:
        return False
    if candidate in LOOPBACK_HOSTNAMES:
        return True
    # urlsplit leaves IPv6 literals unbracketed in .hostname, so this handles both forms.
    try:
        address = ipaddress.ip_address(candidate)
    except ValueError:
        return False
    # ::ffff:127.0.0.1 is loopback but IPv6Address.is_loopback does not say so.
    mapped = getattr(address, "ipv4_mapped", None)
    return bool(mapped.is_loopback) if mapped is not None else address.is_loopback


def check_public_origin(origin: str, *, allow_plaintext: bool = False) -> None:
    """Refuse an origin that would put the bearer token on the wire in the clear.

    Callers authenticate with a bearer token, so a plaintext origin hands that token to
    anyone on the path. This is advisory rather than authoritative: the service cannot see
    its own exposure — in a container it always binds 0.0.0.0 because Docker requires it,
    and the published host address that actually governs reachability lives in Compose. So
    the operator declares the origin and this validates the declaration. The authoritative
    check is on CROPPER_BIND_ADDRESS in the deployment scripts.

    An empty origin means "not declared", which is the shipped default: loopback publishing
    reached over an SSH tunnel. That is encrypted on the wire but arrives as plain http, so
    it cannot be distinguished here and is not treated as a violation.
    """
    if not origin.strip():
        return
    parsed = urlsplit(origin.strip())
    if parsed.scheme.lower() not in {"http", "https"}:
        raise InsecureExposureError("CROPPER_PUBLIC_ORIGIN must use http or https")
    if not parsed.hostname:
        raise InsecureExposureError("CROPPER_PUBLIC_ORIGIN must include a hostname")
    if parsed.username or parsed.password:
        raise InsecureExposureError("CROPPER_PUBLIC_ORIGIN must not embed credentials")
    if parsed.scheme.lower() == "https" or is_loopback_host(parsed.hostname):
        return
    if allow_plaintext:
        logger.warning(
            "Serving over plaintext HTTP because CROPPER_ALLOW_PLAINTEXT is set. Every caller "
            "credential is readable by anything on the network path."
        )
        return
    raise InsecureExposureError(
        "CROPPER_PUBLIC_ORIGIN is plaintext HTTP on a non-loopback host, which exposes the "
        "bearer token to the network. Terminate TLS in front of the service, or set "
        "CROPPER_ALLOW_PLAINTEXT=1 to accept the risk."
    )


def extract_bearer_token(header_value: str | None) -> str | None:
    """Return the token from an ``Authorization: Bearer <token>`` header, else None."""
    if not header_value:
        return None
    scheme, _, token = header_value.partition(" ")
    if scheme.lower() != "bearer":
        return None
    token = token.strip()
    return token or None


def token_is_valid(presented: str | None, expected: str) -> bool:
    """Compare tokens in constant time. An unset expected token never validates."""
    if not expected or not presented:
        return False
    return hmac.compare_digest(presented, expected)


def redact_url(value: str) -> str:
    """Remove credentials, query, and fragment while retaining a useful location."""
    parsed = urlsplit(value)
    host = parsed.hostname or ""
    if parsed.port:
        host = f"{host}:{parsed.port}"
    return urlunsplit((parsed.scheme, host, parsed.path, "", ""))


def _host_allowed(host: str, allowlist: tuple[str, ...]) -> bool:
    return any(host == item or (item.startswith("*.") and host.endswith(item[1:])) for item in allowlist)


def validate_transport_url(
    value: str,
    *,
    allowed_hosts: tuple[str, ...],
    allow_private: bool = False,
    resolve_dns: bool = True,
) -> str:
    """Require HTTP(S), an allowlisted host, and non-private DNS targets by default."""
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise UnsafeURLError("URL must use http or https and include a hostname")
    if parsed.username or parsed.password:
        raise UnsafeURLError("userinfo is not allowed in transport URLs")
    host = parsed.hostname.rstrip(".").lower()
    if not _host_allowed(host, allowed_hosts):
        raise UnsafeURLError("URL host is not allowlisted")
    if resolve_dns and not allow_private:
        try:
            addresses = {item[4][0] for item in socket.getaddrinfo(host, parsed.port or 443)}
        except socket.gaierror as exc:
            raise UnsafeURLError("URL hostname could not be resolved") from exc
        for address in addresses:
            ip = ipaddress.ip_address(address)
            if not ip.is_global:
                raise UnsafeURLError("URL resolves to a private or non-routable address")
    return value


async def validate_redirect_chain(
    value: str,
    *,
    allowed_hosts: tuple[str, ...],
    allow_private: bool = False,
    resolve_dns: bool = True,
    timeout_sec: float = 30.0,
) -> str:
    """Validate every URL the source fetch will visit, not just the submitted one.

    The submitted URL is handed to ffmpeg, which follows redirects. Without this, an
    allowlisted host answering with a 302 reaches any address the service can route to,
    which makes both the host allowlist and the private-address check decorative.

    A residual race remains: ffmpeg re-requests the chain, so a host that redirects
    differently on the second request is not caught here. Closing that would mean fetching
    the source ourselves and giving up ffmpeg's ranged reads for trims.
    """
    current = validate_transport_url(
        value, allowed_hosts=allowed_hosts, allow_private=allow_private, resolve_dns=resolve_dns
    )
    async with httpx.AsyncClient(
        follow_redirects=False, timeout=httpx.Timeout(timeout_sec)
    ) as client:
        for _ in range(MAX_REDIRECT_HOPS):
            try:
                # A ranged GET rather than HEAD: presigned URLs are commonly signed for GET
                # only, and would answer HEAD with a 403 regardless of the real target.
                response = await client.get(current, headers={"Range": "bytes=0-0"})
            except httpx.HTTPError as exc:
                raise UnsafeURLError("source URL could not be reached") from exc

            if not response.is_redirect:
                return current

            location = response.headers.get("location")
            if not location:
                raise UnsafeURLError("source URL returned a redirect without a location")
            logger.info("Source URL redirected to %s", redact_url(urljoin(current, location)))
            current = validate_transport_url(
                urljoin(current, location),
                allowed_hosts=allowed_hosts,
                allow_private=allow_private,
                resolve_dns=resolve_dns,
            )
    raise UnsafeURLError("source URL exceeded the redirect limit")
