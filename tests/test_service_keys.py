"""Unit and functional tests for revocable API keys."""

from __future__ import annotations

import asyncio
import json

import fakeredis.aioredis
import pytest
from fastapi.testclient import TestClient

from v_cropper.service.app import create_app
from v_cropper.service.config import Settings
from v_cropper.service.keys import (
    BOOTSTRAP_KEY_ID,
    BOOTSTRAP_PRINCIPAL,
    ChainVerifier,
    KeyStore,
    StaticTokenVerifier,
    format_issued_key,
    hash_secret,
    parse_issued_key,
    secrets_match,
    validate_key_name,
)

TOKEN = "correct-horse-battery-staple-token"
AUTH = {"Authorization": f"Bearer {TOKEN}"}
PAYLOAD = {
    "source_url": "https://objects.example.test/input.mp4",
    "destination_url": "https://objects.example.test/output.mp4",
}


def _settings(**overrides) -> Settings:
    fields = dict(
        redis_url="redis://unused",
        api_token=TOKEN,
        job_ttl_sec=60,
        event_ttl_sec=60,
        max_duration_sec=60,
        max_output_bytes=1_000_000,
        allowed_hosts=("objects.example.test",),
        allow_private_hosts=True,
        work_root="/tmp/v-cropper-keys",
        http_timeout_sec=10,
    )
    fields.update(overrides)
    return Settings(**fields)


def _app(redis=None, **overrides):
    client = redis or fakeredis.aioredis.FakeRedis()
    app = create_app(_settings(**overrides), client)

    async def idle_worker():
        await asyncio.Event().wait()

    app.state.worker.run = idle_worker
    return app


class TestKeyFormat:
    def test_round_trip(self):
        token = format_issued_key("0123456789abcdef", "secret-value")
        assert parse_issued_key(token) == ("0123456789abcdef", "secret-value")

    def test_rejects_the_bootstrap_token_shape(self):
        assert parse_issued_key(TOKEN) is None

    @pytest.mark.parametrize(
        "value",
        [
            "",
            "vcrop_",
            "vcrop_short_secret",
            "vcrop_0123456789abcdefg_secret",  # 17 hex chars
            "vcrop_GHIJKLMN01234567_secret",  # not hex
            "vcrop_0123456789abcdef_",  # empty secret
            "v-crop_0123456789abcdef_secret",
            "vcrop_0123456789abcdef",  # missing secret
            "vcrop_0123456789abcdef_x_y",  # extra segment is part of secret — still 3 parts after split 2
        ],
    )
    def test_rejects_malformed_tokens(self, value):
        parsed = parse_issued_key(value)
        if value == "vcrop_0123456789abcdef_x_y":
            assert parsed == ("0123456789abcdef", "x_y")
        else:
            assert parsed is None

    def test_hash_is_stable_and_one_way(self):
        digest = hash_secret("abc")
        assert digest == hash_secret("abc")
        assert digest != hash_secret("abd")
        assert "abc" not in digest

    def test_comparison_is_constant_time_and_rejects_the_raw_secret(self):
        digest = hash_secret("abc")
        assert secrets_match("abc", digest) is True
        assert secrets_match("abd", digest) is False
        assert secrets_match(digest, digest) is False


class TestKeyName:
    @pytest.mark.parametrize("name", ["app", "renderer-1", "Qvest backend", "a" * 64])
    def test_accepts_plain_names(self, name):
        assert validate_key_name(name) == name

    @pytest.mark.parametrize(
        "name",
        [
            "",
            " ",
            "x" * 65,
            "../etc/passwd",
            "vcrop_0123456789abcdef_secret",
            "name\nwith-newline",
            "name;rm -rf",
        ],
    )
    def test_rejects_hostile_names(self, name):
        with pytest.raises(ValueError):
            validate_key_name(name)


class TestVerifiers:
    @pytest.mark.asyncio
    async def test_static_verifier_returns_the_admin_principal(self):
        verifier = StaticTokenVerifier(TOKEN)
        assert await verifier.verify(TOKEN) == BOOTSTRAP_PRINCIPAL
        assert await verifier.verify("wrong") is None
        assert await StaticTokenVerifier("").verify(TOKEN) is None

    @pytest.mark.asyncio
    async def test_chain_prefers_the_first_match(self):
        store = KeyStore(fakeredis.aioredis.FakeRedis())
        token, _ = await store.mint("client")
        chain = ChainVerifier([StaticTokenVerifier(TOKEN), store])
        assert (await chain.verify(TOKEN)).is_admin is True
        assert (await chain.verify(token)).is_admin is False
        assert await chain.verify("nope") is None


class TestKeyStore:
    @pytest.mark.asyncio
    async def test_mint_stores_only_the_hash(self):
        redis = fakeredis.aioredis.FakeRedis()
        store = KeyStore(redis)
        token, record = await store.mint("renderer")
        assert record["name"] == "renderer"
        assert record["key_id"] in token
        raw = json.loads(await redis.get(store.record_key(record["key_id"])))
        assert "hash" in raw
        assert token not in json.dumps(raw)
        assert parse_issued_key(token)[1] not in json.dumps(raw)

    @pytest.mark.asyncio
    async def test_verify_and_revoke(self):
        store = KeyStore(fakeredis.aioredis.FakeRedis())
        token, record = await store.mint("renderer")
        principal = await store.verify(token)
        assert principal is not None
        assert principal.key_id == record["key_id"]
        assert principal.is_admin is False
        assert await store.revoke(record["key_id"]) is True
        assert await store.verify(token) is None

    @pytest.mark.asyncio
    async def test_forged_secret_with_a_real_key_id_fails(self):
        store = KeyStore(fakeredis.aioredis.FakeRedis())
        token, record = await store.mint("renderer")
        forged = format_issued_key(record["key_id"], "not-the-secret")
        assert token != forged
        assert await store.verify(forged) is None

    @pytest.mark.asyncio
    async def test_key_id_outside_the_namespace_is_rejected(self):
        store = KeyStore(fakeredis.aioredis.FakeRedis())
        with pytest.raises(ValueError):
            store.record_key("../jobs")
        with pytest.raises(ValueError):
            store.record_key("abc:def:hij1")
        assert await store.revoke("not-hex-and-too-long-to-match") is False
        assert await store.get("not-a-key-id") is None

    @pytest.mark.asyncio
    async def test_list_skips_an_orphaned_index_entry(self):
        redis = fakeredis.aioredis.FakeRedis()
        store = KeyStore(redis)
        token, record = await store.mint("kept")
        await redis.sadd(store.index_key(), "0123456789abcdef")
        listed = await store.list()
        assert [item["key_id"] for item in listed] == [record["key_id"]]
        assert await store.verify(token) is not None


class TestAdminApi:
    def test_bootstrap_token_still_submits_jobs(self):
        with TestClient(_app()) as client:
            assert client.post("/v1/jobs", json=PAYLOAD, headers=AUTH).status_code == 202

    def test_mint_use_revoke_lifecycle(self):
        with TestClient(_app()) as client:
            created = client.post("/v1/admin/keys", json={"name": "renderer"}, headers=AUTH)
            assert created.status_code == 201
            body = created.json()
            token = body["token"]
            key_id = body["key_id"]
            assert token.startswith("vcrop_")
            assert parse_issued_key(token)[0] == key_id

            listed = client.get("/v1/admin/keys", headers=AUTH)
            assert listed.status_code == 200
            assert listed.json()[0]["key_id"] == key_id
            assert "token" not in listed.json()[0]
            assert "hash" not in listed.json()[0]

            minted_auth = {"Authorization": f"Bearer {token}"}
            assert client.post("/v1/jobs", json=PAYLOAD, headers=minted_auth).status_code == 202
            assert client.post("/v1/admin/keys", json={"name": "other"}, headers=minted_auth).status_code == 403
            assert client.get("/v1/admin/keys", headers=minted_auth).status_code == 403

            assert client.delete(f"/v1/admin/keys/{key_id}", headers=AUTH).status_code == 204
            assert client.post("/v1/jobs", json=PAYLOAD, headers=minted_auth).status_code == 401
            assert client.post("/v1/jobs", json=PAYLOAD, headers=AUTH).status_code == 202

    def test_a_second_key_survives_revoking_the_first(self):
        with TestClient(_app()) as client:
            first = client.post("/v1/admin/keys", json={"name": "one"}, headers=AUTH).json()
            second = client.post("/v1/admin/keys", json={"name": "two"}, headers=AUTH).json()
            client.delete(f"/v1/admin/keys/{first['key_id']}", headers=AUTH)
            surviving = {"Authorization": f"Bearer {second['token']}"}
            assert client.post("/v1/jobs", json=PAYLOAD, headers=surviving).status_code == 202
            revoked = {"Authorization": f"Bearer {first['token']}"}
            assert client.post("/v1/jobs", json=PAYLOAD, headers=revoked).status_code == 401

    def test_keys_survive_a_rebuilt_app_against_the_same_redis(self):
        redis = fakeredis.aioredis.FakeRedis()
        with TestClient(_app(redis=redis)) as client:
            created = client.post("/v1/admin/keys", json={"name": "durable"}, headers=AUTH).json()
        with TestClient(_app(redis=redis)) as client:
            minted = {"Authorization": f"Bearer {created['token']}"}
            assert client.post("/v1/jobs", json=PAYLOAD, headers=minted).status_code == 202

    def test_bootstrap_cannot_be_revoked_through_the_api(self):
        with TestClient(_app()) as client:
            response = client.delete(f"/v1/admin/keys/{BOOTSTRAP_KEY_ID}", headers=AUTH)
            assert response.status_code == 400
            assert client.post("/v1/jobs", json=PAYLOAD, headers=AUTH).status_code == 202

    def test_unknown_key_is_404(self):
        with TestClient(_app()) as client:
            assert client.delete("/v1/admin/keys/0123456789abcdef", headers=AUTH).status_code == 404

    def test_minted_token_is_returned_exactly_once(self):
        with TestClient(_app()) as client:
            created = client.post("/v1/admin/keys", json={"name": "once"}, headers=AUTH)
            token = created.json()["token"]
            listed = client.get("/v1/admin/keys", headers=AUTH).text
            assert token not in listed

    def test_a_name_that_looks_like_a_token_is_refused(self):
        with TestClient(_app()) as client:
            response = client.post(
                "/v1/admin/keys",
                json={"name": "vcrop_0123456789abcdef_secret"},
                headers=AUTH,
            )
            assert response.status_code == 422
            assert "issued API key" in response.text
