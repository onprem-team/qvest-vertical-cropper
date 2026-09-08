"""Caller authentication on the job API.

The service holds a provider API key with real spend attached and will fetch and write
arbitrary allowlisted URLs on a caller's behalf, so an unauthenticated /v1 route is a
budget-drain and SSRF-proxy primitive, not just an information leak.
"""
from __future__ import annotations

import asyncio

import fakeredis.aioredis
import pytest
from fastapi.testclient import TestClient

from v_cropper.service.app import create_app
from v_cropper.service.config import Settings
from v_cropper.service.security import (
    MIN_RECOMMENDED_TOKEN_LENGTH,
    extract_bearer_token,
    token_is_valid,
)

TOKEN = "correct-horse-battery-staple-token"
AUTH = {"Authorization": f"Bearer {TOKEN}"}
PAYLOAD = {
    "source_url": "https://objects.example.test/input.mp4",
    "destination_url": "https://objects.example.test/output.mp4",
}


def _app(**overrides):
    fields = dict(
        redis_url="redis://unused",
        api_token=TOKEN,
        job_ttl_sec=60,
        event_ttl_sec=60,
        max_duration_sec=60,
        max_output_bytes=1_000_000,
        allowed_hosts=("objects.example.test",),
        allow_private_hosts=True,
        work_root="/tmp/v-cropper-tests",
        http_timeout_sec=10,
    )
    fields.update(overrides)
    app = create_app(Settings(**fields), fakeredis.aioredis.FakeRedis())

    async def idle_worker():
        await asyncio.Event().wait()

    app.state.worker.run = idle_worker
    return app


@pytest.fixture(autouse=True)
def _ffmpeg_present(monkeypatch):
    """Readiness checks ffmpeg before the token, so absent tooling masks the token branch."""
    monkeypatch.setattr("v_cropper.service.app.shutil.which", lambda name: f"/usr/bin/{name}")


class TestBearerParsing:
    @pytest.mark.parametrize("header", [
        None, "", "Bearer", "Bearer ", "Bearer   ", "Basic abc", "token abc", "abc",
    ])
    def test_malformed_headers_yield_no_token(self, header):
        assert extract_bearer_token(header) is None

    @pytest.mark.parametrize("header,expected", [
        ("Bearer abc", "abc"),
        ("bearer abc", "abc"),        # scheme is case-insensitive per RFC 7235
        ("BEARER abc", "abc"),
        ("Bearer  abc  ", "abc"),
    ])
    def test_well_formed_headers(self, header, expected):
        assert extract_bearer_token(header) == expected

    @pytest.mark.parametrize("presented,expected,ok", [
        ("abc", "abc", True),
        ("abc", "abd", False),
        ("ab", "abc", False),         # prefix must not validate
        ("abcd", "abc", False),
        (None, "abc", False),
        ("", "abc", False),
        ("abc", "", False),           # unset server token never validates
        (None, "", False),
    ])
    def test_token_comparison(self, presented, expected, ok):
        assert token_is_valid(presented, expected) is ok


class TestRouteProtection:
    def test_health_endpoints_stay_open(self):
        """Probes must work without a credential or the platform cannot health-check."""
        with TestClient(_app()) as client:
            assert client.get("/healthz").status_code == 200
            assert client.get("/version").status_code == 200

    @pytest.mark.parametrize("method,path", [
        ("post", "/v1/jobs"),
        ("get", "/v1/jobs/abc123"),
        ("delete", "/v1/jobs/abc123"),
        ("get", "/v1/jobs/abc123/events"),
    ])
    @pytest.mark.parametrize("headers", [
        {},
        {"Authorization": "Bearer wrong-token"},
        {"Authorization": f"Basic {TOKEN}"},
        {"Authorization": TOKEN},
        {"Authorization": "Bearer "},
    ])
    def test_every_job_route_rejects_bad_credentials(self, method, path, headers):
        with TestClient(_app()) as client:
            response = client.request(method, path, headers=headers, json=PAYLOAD)
            assert response.status_code == 401
            assert response.headers["WWW-Authenticate"] == "Bearer"

    def test_auth_is_checked_before_the_resource_exists(self):
        """An unauthenticated caller must not learn which job ids are real."""
        with TestClient(_app()) as client:
            assert client.get("/v1/jobs/does-not-exist").status_code == 401
            assert client.get("/v1/jobs/does-not-exist", headers=AUTH).status_code == 404

    def test_valid_token_is_accepted(self):
        with TestClient(_app()) as client:
            assert client.post("/v1/jobs", json=PAYLOAD, headers=AUTH).status_code == 202

    def test_all_v1_routes_carry_the_auth_dependency(self):
        """Guard against a future /v1 route being registered without auth.

        Driven off the published OpenAPI schema and asserted behaviourally rather than by
        inspecting FastAPI internals, so it covers routes added later and does not depend
        on how a given FastAPI version stores included routers.
        """
        app = _app()
        checked = []
        with TestClient(app) as client:
            for path, operations in app.openapi()["paths"].items():
                if not path.startswith("/v1"):
                    continue
                concrete = path.replace("{job_id}", "abc123").replace("{key_id}", "0123456789abcdef")
                for method in operations:
                    response = client.request(method, concrete, json=PAYLOAD)
                    assert response.status_code == 401, f"{method.upper()} {path} is unauthenticated"
                    checked.append(f"{method.upper()} {path}")
        # Sanity: the schema actually contained job and admin routes rather than none.
        assert sorted(checked) == [
            "DELETE /v1/admin/keys/{key_id}",
            "DELETE /v1/jobs/{job_id}",
            "GET /v1/admin/keys",
            "GET /v1/jobs/{job_id}",
            "GET /v1/jobs/{job_id}/events",
            "POST /v1/admin/keys",
            "POST /v1/jobs",
        ]

    def test_admin_routes_reject_a_non_admin_principal(self):
        with TestClient(_app()) as client:
            minted = client.post("/v1/admin/keys", json={"name": "worker"}, headers=AUTH)
            assert minted.status_code == 201
            headers = {"Authorization": f"Bearer {minted.json()['token']}"}
            assert client.get("/v1/admin/keys", headers=headers).status_code == 403
            assert client.post("/v1/admin/keys", json={"name": "x"}, headers=headers).status_code == 403
            assert client.delete(
                f"/v1/admin/keys/{minted.json()['key_id']}", headers=headers
            ).status_code == 403


class TestFailClosed:
    def test_unconfigured_token_refuses_job_routes(self):
        """No token configured must not mean no authentication."""
        with TestClient(_app(api_token="")) as client:
            response = client.post("/v1/jobs", json=PAYLOAD)
            assert response.status_code == 503
            assert response.json()["detail"] == "API token is not configured"

    def test_unconfigured_token_is_not_ready(self, monkeypatch):
        monkeypatch.setenv("VCROPPER_PROVIDER", "openai")
        monkeypatch.setenv("VCROPPER_API_KEY", "test-key")
        with TestClient(_app(api_token="")) as client:
            response = client.get("/readyz")
            assert response.status_code == 503
            assert response.json()["detail"] == "API token is not configured"

    def test_configured_token_is_ready(self, monkeypatch):
        monkeypatch.setenv("VCROPPER_PROVIDER", "openai")
        monkeypatch.setenv("VCROPPER_API_KEY", "test-key")
        with TestClient(_app()) as client:
            assert client.get("/readyz").json() == {"status": "ready"}

    def test_empty_token_cannot_be_satisfied_by_an_empty_header(self):
        with TestClient(_app(api_token="")) as client:
            assert client.post(
                "/v1/jobs", json=PAYLOAD, headers={"Authorization": "Bearer "}
            ).status_code == 503


class TestNoSecretLeakage:
    def test_rejection_never_echoes_the_expected_token(self):
        with TestClient(_app()) as client:
            response = client.post(
                "/v1/jobs", json=PAYLOAD, headers={"Authorization": "Bearer guess"}
            )
            assert TOKEN not in response.text
            assert TOKEN not in str(dict(response.headers))

    def test_short_token_is_warned_about(self, caplog):
        with caplog.at_level("WARNING"):
            _app(api_token="short")
        assert any(str(MIN_RECOMMENDED_TOKEN_LENGTH) in record.message for record in caplog.records)

    def test_adequate_token_is_not_warned_about(self, caplog):
        with caplog.at_level("WARNING"):
            _app()
        assert not any("CROPPER_API_TOKEN" in record.message for record in caplog.records)
