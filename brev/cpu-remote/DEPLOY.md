# CPU Remote-VLM deployment

The path for deploying the v-cropper HTTP API on a Brev CPU instance. The
instance runs no model: it calls a hosted VLM (NVIDIA, a gateway/router, Gemini, or AWS
Bedrock) over an API and exposes a job API to your own backend.

Source: <https://github.com/onprem-team/qvest-vertical-cropper>

## Before you launch

- Restrict the Launchable to people who should be able to spend the provider key.
- Use a scoped provider key with an explicit spend limit.
- Acknowledge that frames leave the VM for the chosen provider.

## Why VM Mode

Create a Brev **Launchable** in VM Mode whose git URL is this public repository. Brev
clones the repo on the instance and runs `brev/cpu-remote/setup.sh`.

Do not publish a TCP port or a Secure Link. The API binds to the instance's loopback
interface and is reached through `brev port-forward`, an authenticated SSH tunnel. A
published port would put a service holding a spendable provider key on the public internet.

## Launch parameters

Supply at least:

- `CROPPER_API_TOKEN` — bootstrap admin bearer token (prefer *Use a secret*).
- `VCROPPER_API_KEY` — hosted VLM credential (or the Bedrock/AWS equivalents you use).

Optional: `VCROPPER_BASE_URL`, `VCROPPER_MODEL`, `VCROPPER_PROVIDER`, and the other
variables listed in `brev/cpu-remote/profile.env.example`. The Brev secret's *name* does
not have to match the parameter name: the parameter becomes the environment variable.

## Configure

`setup.sh` writes `~/.config/v-cropper/provider.env` from Brev **Launch parameters** on
first boot and then starts the API. `CROPPER_API_TOKEN` is required — a generated token
would be unreachable without SSH, which breaks the one-click Launchable.

Launch parameters are supplied once, at launch. They are **not** re-supplied to later
container runs, so the profile file is what makes `up` work again after a stop/start.

To configure by hand instead, from the cloned repository root:

```bash
bash brev/cpu-remote/setup.sh

mkdir -p ~/.config/v-cropper
cp brev/cpu-remote/profile.env.example ~/.config/v-cropper/provider.env
chmod 600 ~/.config/v-cropper/provider.env
python3 -c 'import secrets; print("CROPPER_API_TOKEN=" + secrets.token_urlsafe(32))'
```

Edit the file with your endpoint, key, model, and that token. The wrappers refuse to run if
it is group- or world-readable, or if the token is shorter than 24 characters.

## Run

```bash
brev/cpu-remote/vcropper-service up       # start API + Redis, wait for health
brev/cpu-remote/vcropper-service status
brev/cpu-remote/vcropper-service logs -f
brev/cpu-remote/vcropper-service down
```

## Verify

```bash
brev/cpu-remote/deploy-check.sh           # config only, spends nothing
brev/cpu-remote/vcropper-service verify   # full end-to-end, makes real VLM calls
```

`verify` starts a throwaway MinIO alongside the stack and drives a real job through the
deployed HTTP surface: presigned GET in, presigned PUT out, bearer auth enforced, and the
resulting object probed for a 9:16 stream. It also asserts the VLM was actually reached —
a crop is still produced when every model call fails, so "succeeded" alone proves nothing.

Retain only non-secret evidence: `verify-evidence.json`, `runtime-manifest.json`, the
resolved endpoint host/path, model id, source revision, and keyframe-failure rate.

## Calling the service

The API is not reachable from your workstation until you open the tunnel:

```bash
brev/cpu-remote/vcropper-tunnel <instance-name>     # holds 127.0.0.1:8090, reconnects on drop
```

Leave that running. In another shell, the service behaves as an ordinary HTTP API on
`http://127.0.0.1:8090`. Every `/v1` route requires a bearer token;
`/healthz`, `/readyz`, and `/version` are open so the platform can probe them.

### Caller credentials

`CROPPER_API_TOKEN` is the bootstrap **admin** credential, supplied as a Launch parameter
(prefer *Use a secret*). Mint a distinct key for each calling application so you can
revoke one without rotating the others:

```bash
# Returned once. Store the token; list/get never show it again.
curl -X POST http://127.0.0.1:8090/v1/admin/keys \
  -H "Authorization: Bearer $CROPPER_API_TOKEN" \
  -H 'content-type: application/json' \
  -d '{"name": "renderer"}'

curl -H "Authorization: Bearer $CROPPER_API_TOKEN" http://127.0.0.1:8090/v1/admin/keys
curl -X DELETE -H "Authorization: Bearer $CROPPER_API_TOKEN" \
  http://127.0.0.1:8090/v1/admin/keys/<key_id>
```

Minted keys authenticate job routes and are refused on `/v1/admin/*`. They live hashed in
Redis (`cropper-redis` volume) and survive an instance stop/start.

**Rotating the bootstrap token is not "update the secret and restart".** Brev pins the
exact secret version at deploy, the setup script does not rerun on restart, and
`setup.sh` preserves an existing `provider.env`. To rotate it, redeploy a fresh instance
selecting the new version, or edit `provider.env` on the instance and restart the
service. The same caveat applies to `VCROPPER_API_KEY`, which we only hold and cannot
revoke. In practice, mint an application key once with the bootstrap token and then
leave the bootstrap token alone.

Publishing the API off loopback without TLS puts the bearer token on the wire. The
wrappers refuse a non-loopback `CROPPER_BIND_ADDRESS` unless `CROPPER_ALLOW_PLAINTEXT=1`.

The service never fetches or stores your media directly — you pass **presigned URLs** and it
reads the source and writes the result itself. Both hosts must be in `CROPPER_ALLOWED_HOSTS`.

### Worked example

```python
import time
import boto3
import httpx

BASE = "http://127.0.0.1:8090"
AUTH = {"Authorization": f"Bearer {CROPPER_API_TOKEN}"}

s3 = boto3.client("s3")
source_url = s3.generate_presigned_url(
    "get_object", Params={"Bucket": "media", "Key": "game.mp4"}, ExpiresIn=3600)
destination_url = s3.generate_presigned_url(
    "put_object",
    Params={"Bucket": "media", "Key": "game_vertical.mp4", "ContentType": "video/mp4"},
    ExpiresIn=3600,
)

job = httpx.post(f"{BASE}/v1/jobs", headers=AUTH, timeout=30, json={
    "source_url": source_url,
    "destination_url": destination_url,
    "in_s": 12.0,           # optional trim window
    "out_s": 42.0,
    "sport": "football",
}).raise_for_status().json()

while True:                  # or stream GET /v1/jobs/{id}/events (SSE)
    view = httpx.get(f"{BASE}/v1/jobs/{job['job_id']}", headers=AUTH, timeout=30).json()
    if view["status"] in {"succeeded", "failed", "cancelled"}:
        break
    time.sleep(2)

assert view["status"] == "succeeded", view["error"]
# The crop is now at the destination URL; view["metrics"] carries keyframe and cost data.
```

`POST /v1/jobs` returns `202` with a `job_id`. Poll `GET /v1/jobs/{job_id}` or subscribe to
`GET /v1/jobs/{job_id}/events` for Server-Sent Events (`progress` then a final `terminal`).
`DELETE /v1/jobs/{job_id}` cancels at the next phase boundary. Job views never echo your
signed URLs.

Check `metrics.keyframe_fail_fraction` on success: a high value means the model struggled
even though the job completed. A job where *every* keyframe fails is reported as `failed`.

### Errors

| Status | Meaning |
|---|---|
| `401` | Missing or wrong bearer token |
| `403` | A minted key was used on an admin route |
| `422` | URL host not allowlisted, bad trim window, or invalid parameters |
| `503` | Service unconfigured (no API token, no provider credential, Redis down) |

Failures return a generic `crop processing failed` message; the specific cause is in
`vcropper-service logs` so that provider details are not exposed to callers.

## Benchmarking the deployed configuration

Run the labeled benchmark with the **same** provider settings the service uses:

```bash
set -a; source ~/.config/v-cropper/provider.env; set +a
export VCROPPER_EVAL_DATA=/path/to/your/dataset
uv run python eval/run_eval.py --split val --base-url "$VCROPPER_BASE_URL" \
  --model "$VCROPPER_MODEL" --out /tmp/vcropper-cpu-eval --no-cache
```

The harness derives its keyframe stride the same way the service does, so leave
`--sample-every` unset unless you are deliberately benchmarking a different sampling rate;
the resolved stride is recorded in `metrics-<split>.json` under `sampling`.

Do not share `/tmp/vcropper-cpu-eval/.cache` across providers, endpoints, models, or prompts.

## Operational limits

- One instance runs one worker in-process; concurrency is bounded by that container.
- Redis state is durable across restarts (`cropper-redis` volume); job artifacts are not.
- Stopping the instance stops the API. `vcropper-service up` restarts it from `provider.env`.
