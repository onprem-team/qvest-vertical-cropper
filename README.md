# v-cropper-cli — sports cropper

[![CI](https://github.com/onprem-team/qvest-vertical-cropper/actions/workflows/ci.yml/badge.svg)](https://github.com/onprem-team/qvest-vertical-cropper/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

Turn a landscape sports video into a smooth 9:16 vertical crop that follows the action.
A vision model points at the current focus of play (the ball/puck or the player carrying
it) on each sampled keyframe; those points are outlier-filtered, interpolated to every
frame, and a critically-damped spring pans the crop window smoothly to follow them.

Works with **any OpenAI-compatible vision model** (Gemini, OpenAI, OpenRouter,
NVIDIA-hosted, …) and with Anthropic Claude vision models through the native
AWS Bedrock Converse API.

## Install

Requires [uv](https://docs.astral.sh/uv/) and `ffmpeg` on PATH (for h264 output; without
it you still get a playable mp4v file).

```bash
uv sync
```



### Install as a standalone CLI

To get a `v-cropper` command on your `PATH` (usable from any directory), install it as a uv
tool from the project root:

```bash
uv tool install .                       # installs `v-cropper` to ~/.local/bin
v-cropper --help

uv tool install . --force --reinstall   # update after changing the code
uv tool uninstall v-cropper-cli         # remove it
```

This installs a snapshot — it won't pick up source edits until you reinstall. For active
development, use `uv run v-cropper …` from the project instead (it always runs the current code).

## Usage

```bash
export VCROPPER_API_KEY=your-key        # or use a .env file, or --api-key
uv run v-cropper input.mp4              # → input_vertical.mp4 beside the source
uv run v-cropper input.mp4 --debug      # also writes input_debug.mp4 (focus point + crop box)
uv run v-cropper input.mp4 -o out.mp4   # custom output path
```



### Provider / model

Set `VCROPPER_PROVIDER=openai|bedrock`; `openai` is the default for backward compatibility.
In OpenAI-compatible mode, the connector is configured from env vars (or `.env`) and is
overridable per run. **The default
endpoint + model is NVIDIA-hosted** (`nvidia/nemotron-3-nano-omni-30b-a3b-reasoning` at
`https://integrate.api.nvidia.com/v1`) — set `VCROPPER_API_KEY` and you're ready. Point at any
other OpenAI-compatible provider by overriding the base URL + model:

```bash
VCROPPER_API_KEY=...        # required
VCROPPER_BASE_URL=...       # provider endpoint (omit to use the NVIDIA default)
VCROPPER_MODEL=...           # model id (omit to use the provider default)
```

Use `--base-url URL` to override `$VCROPPER_BASE_URL` for one run. Custom endpoints should expose
an OpenAI-compatible `/v1` API. Gateways that require additional headers can use
`VCROPPER_EXTRA_HEADERS` as a JSON object; do not put credentials in a URL or commit header values.

For native Bedrock, boto3 uses the standard AWS credential chain (environment variables,
shared AWS config/profile, container credentials, or an EC2/ECS role):

```bash
VCROPPER_PROVIDER=bedrock
BEDROCK_REGION=us-east-1             # AWS_REGION is the fallback
BEDROCK_VLM_MODEL_ID=global.anthropic.claude-sonnet-4-5-20250929-v1:0
# Optional for local static credentials; workload roles are preferred:
AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=... AWS_SESSION_TOKEN=...
uv run v-cropper input.mp4
```

The Bedrock model precedence is `--model` → `VCROPPER_MODEL` →
`BEDROCK_VLM_MODEL_ID` → the built-in global Claude Sonnet inference profile. The selected
profile must be enabled in the configured AWS account and region. `VCROPPER_TIMEOUT` and
`VCROPPER_MAX_TOKENS` apply to both provider modes. `--base-url` and `VCROPPER_EXTRA_HEADERS`
apply to the OpenAI-compatible mode only and are ignored (with a warning) under Bedrock.

Examples:

```bash
# OpenAI
VCROPPER_BASE_URL=https://api.openai.com/v1     VCROPPER_MODEL=gpt-4o
# OpenRouter
VCROPPER_BASE_URL=https://openrouter.ai/api/v1  VCROPPER_MODEL=qwen/qwen3-vl-235b-a22b-instruct
# Gemini (also the GEMINI_API_KEY back-compat target)
VCROPPER_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/  VCROPPER_MODEL=gemini-2.5-flash
```

The API key is resolved as: `--api-key` → `$VCROPPER_API_KEY` → `$GEMINI_API_KEY` →
`.env`. As a convenience, if only `GEMINI_API_KEY` is set (no `VCROPPER_BASE_URL`), the
connector automatically routes to Gemini's OpenAI-compatible endpoint. See
[.env.example](.env.example) for every setting.

> **Provider notes:** the shipped default is NVIDIA-hosted
> `nemotron-3-nano-omni-30b-a3b-reasoning` (live-checked 2026-09: accepts vision
> chat completions and returns parseable `{x,y}` points). For the best tracking
> accuracy, **Gemini** (`gemini-flash-latest`) leads — see [Performance](#performance-case-study).
> If a given model mis-formats its point replies (watch the keyframe-fail % in the run summary),
> switch to Gemini.



## Sports & pointing prompts

The pointing prompt is **configurable**. Football is the default. Choose a built-in sport
preset, or supply your own prompt for any sport/content:

```bash
uv run v-cropper game.mp4 --sport basketball          # built-in preset
uv run v-cropper game.mp4 --prompt "point at the ..."  # inline custom prompt
uv run v-cropper game.mp4 --prompt-file my_prompt.txt  # prompt from a file
```

Precedence: `--prompt` (inline) → `--prompt-file` → `--sport` preset → football default.
Each is also settable via env (`VCROPPER_PROMPT`, `VCROPPER_PROMPT_FILE`, `VCROPPER_SPORT`).
A custom prompt must ask the model to reply with `{"x": <0-1000>, "y": <0-1000>}`
normalized image coordinates (the CLI warns if yours doesn't mention it).

### Built-in presets & tested sports


| Preset               | Points at                                        | Status                                                          |
| -------------------- | ------------------------------------------------ | --------------------------------------------------------------- |
| `football` (default) | ball carrier / ball in flight / formation center | **Benchmarked to parity** with the R&D v0013 winner (see below) |
| `basketball`         | player with the ball / ball in flight            | Preset provided                                                 |
| `soccer`             | player with the ball / ball in flight            | Preset provided                                                 |
| `hockey`             | player with the puck / loose puck                | Preset provided                                                 |
| `general`            | the main subject a director would keep in frame  | Preset provided (fallback for any sport)                        |


American **football** is the sport validated end-to-end against the labeled benchmark
dataset. The other presets share the same, proven pointing/smoothing pipeline and only
differ in prompt wording; tune them with `--prompt`/`--prompt-file` for your footage.

### Tuning (optional)

- `--sample-fps F` — keyframes sampled per second (default 2.0), or `--sample-every N`.
- `--send-width PX` — downscale width sent to the VLM (default 768; smaller = cheaper).
- `--spring-k K` — spring stiffness; lower = smoother, slower pan.
- `--concurrency N` — parallel VLM calls.
- `--base-url URL` — per-run OpenAI-compatible endpoint override.
- `--model ID` — model id (else `$VCROPPER_MODEL`, else the provider default).
- `--metrics-json PATH` — write run metrics (wall time, rtf, tokens, est. cost).

Each of `send-width`, `spring-k`, `sport` also has a matching `VCROPPER_*` env var.

### Scoreboard overlay (optional)

When a landscape broadcast is cropped to 9:16, the on-screen scoreboard (score, clock,
period, team names) is usually cut off. With `--scoreboard`, a VLM reads the original
scoreboard from a handful of sampled frames, builds a consensus game state, and re-draws a
clean overlay band on the vertical output.

```bash
uv run v-cropper input.mp4 --scoreboard            # add the reconstructed scoreboard
uv run v-cropper input.mp4 --scoreboard --debug    # overlay + OCR consensus debug panel
```

It is off by default and works across sports (team matchups and individual events). If no
readable scoreboard is found, the overlay is skipped and the output is the plain crop.

Knobs (each also settable via the matching `SCOREBOARD_*` env var / `.env`):
`--scoreboard-sample-count N` (frames sent to the VLM, default 10),
`--scoreboard-model ID` (default: the main model),
`--scoreboard-position {bottom,top}`, `--scoreboard-height-ratio` (band height fraction),
`--scoreboard-opacity` (band background opacity).

Note: `--scoreboard` makes extra VLM calls (one per sampled frame, ~10 by default) in
addition to the focus-extraction calls; they run in parallel with focus extraction.

## How it works

1. **Point** — sample keyframes (default 2/s), downscale each to `send_width`, and ask the
  vision model to reply with `{"x","y"}` (0–1000 normalized) pointing at the current focus
   of play. One small image per call, so it is cheap and runs on single-image models.
2. **Filter + interpolate** — a median-of-3 pass rejects a single stray point, then the
  sparse per-keyframe x-focus is linearly interpolated to every frame.
3. **Smooth + crop** — a critically-damped spring pans a full-height 9:16 crop window to
  follow the focus without jitter, clamped to the frame edges.



## Use as a library

Beyond the CLI, the `v_cropper` package is importable. The core pieces:

- `focus.extract_focus_points(video, *, backend=None, sport=None, prompt=None, …)` — sample
keyframes and return `(focus_map, reasoning, n_failed)` (a per-keyframe x-focus map).
- `render.compute_crop_path(video, focus_map, spring_k=…)` — turn the focus map into a smoothed
per-frame 9:16 crop window; `render.render(...)` writes the cropped video (+ optional debug view).
- `prompts.resolve_prompt(...)` — resolve the sport/inline/file pointing prompt.
- `backend.make_backend(...)` — build the selected OpenAI-compatible or Bedrock backend.
- Scoreboard reconstruction: `scoreboard_ocr` / `scoreboard_state` / `scoreboard_render`.

```python
from v_cropper.backend import make_backend
from v_cropper.focus import extract_focus_points
from v_cropper.render import render

backend = make_backend()  # reads VCROPPER_PROVIDER and provider-specific environment
focus_map, _reasoning, n_failed = extract_focus_points("game.mp4", backend=backend, sport="football")
render("game.mp4", focus_map, "game_vertical.mp4")
```

Install into your environment from source/git (there is no PyPI release):

```bash
uv pip install .            # from a checkout
pip install git+https://github.com/onprem-team/qvest-vertical-cropper.git  # directly from git
```



## Async crop-job service

The same cropper is available as a standalone FastAPI service backed by Redis. It runs one
bounded in-process worker and transports media through presigned HTTP GET/PUT URLs; the
service does not need object-store credentials. Start it on `http://localhost:8090`:

```bash
export VCROPPER_API_KEY=...
export CROPPER_API_TOKEN=$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')
export CROPPER_ALLOWED_HOSTS=minio,host.docker.internal
docker compose up --build
curl http://localhost:8090/readyz
```

This stack is self-contained: it brings up its own Redis and needs no external object
store beyond the presigned URLs you submit. Set `CROPPER_PUBLISHED_PORT` to move it off
8090 when another deployment already publishes the cropper there.

**Authentication is mandatory.** Every `/v1` route requires
`Authorization: Bearer <token>`; `/healthz`, `/readyz`, and `/version` stay open
for platform probes. Compose refuses to start without `CROPPER_API_TOKEN`, and the service
rejects every job request if it is unset — the service holds a spendable provider key and
fetches caller-supplied URLs, so an open `/v1` is a budget drain and an SSRF proxy, not just
an information leak.

The published port binds to `127.0.0.1` by default (`CROPPER_BIND_ADDRESS`). Widening that
to a routable address without TLS in front puts the bearer token on the wire in plaintext;
the wrappers refuse unless you set `CROPPER_ALLOW_PLAINTEXT=1`. For local Compose, talk to
`http://127.0.0.1:8090`. On a Brev VM the same loopback bind is reached over SSH — see
[brev/cpu-remote/DEPLOY.md](brev/cpu-remote/DEPLOY.md). If you terminate TLS in front of a
public bind, declare it with `CROPPER_PUBLIC_ORIGIN=https://…`.

`CROPPER_API_TOKEN` is the **bootstrap** admin credential. Use it to mint ordinary keys
for each calling application; those keys can be revoked without a restart and without
rotating anyone else:

```bash
# Mint — the token is returned once. Store it; it cannot be shown again.
curl -X POST http://localhost:8090/v1/admin/keys \
  -H "authorization: Bearer $CROPPER_API_TOKEN" \
  -H 'content-type: application/json' \
  -d '{"name": "renderer"}'
# {"key_id":"…","name":"renderer","token":"vcrop_<key_id>_<secret>",…}

# List metadata only — never the secret.
curl -H "authorization: Bearer $CROPPER_API_TOKEN" http://localhost:8090/v1/admin/keys

# Revoke. The next request with that token is 401; other keys are untouched.
curl -X DELETE -H "authorization: Bearer $CROPPER_API_TOKEN" \
  http://localhost:8090/v1/admin/keys/<key_id>
```

Minted keys authenticate job routes and are refused on `/v1/admin/*`. The bootstrap token
cannot be revoked through the API — it is configuration, and rotating it means changing
`CROPPER_API_TOKEN` and restarting. An OAuth2 proxy or API gateway in front is the
supported path if you need federated identity.

Submit a clip (times are optional seconds; all crop/scoreboard options accepted by the
request schema are shown in `/docs`). Hosts in the URLs must appear in
`CROPPER_ALLOWED_HOSTS` — the Compose quickstart allowlists `minio,host.docker.internal`.
For a public store, set the allowlist first:

```bash
export CROPPER_ALLOWED_HOSTS=objects.example
```

```bash
curl -X POST http://localhost:8090/v1/jobs \
  -H "authorization: Bearer $CROPPER_API_TOKEN" \
  -H 'content-type: application/json' \
  -d '{
    "source_url": "https://objects.example/input.mp4?<presigned-get-query>",
    "destination_url": "https://objects.example/output.mp4?<presigned-put-query>",
    "in_s": 30.0,
    "out_s": 45.0,
    "sport": "football",
    "sample_fps": 2.0
  }'

AUTH="authorization: Bearer $CROPPER_API_TOKEN"
curl -H "$AUTH" http://localhost:8090/v1/jobs/<job_id>
curl -H "$AUTH" -N http://localhost:8090/v1/jobs/<job_id>/events
curl -H "$AUTH" -X DELETE http://localhost:8090/v1/jobs/<job_id>
```

`POST /v1/jobs` returns `202`; poll `GET /v1/jobs/{job_id}` or consume its SSE stream.
Status includes `job_id`, `status`, `phase`, `progress_pct`, `metrics`, `result`, and
`error`. Presigned URLs are private job inputs and are never returned or logged. Redis
state and event history expire after `CROPPER_JOB_TTL_SEC` (24 hours by default).

A job whose keyframes *all* fail is reported as `failed` rather than `succeeded`: the
renderer falls back to a static centre crop when the focus map is empty, which otherwise
returns a plausible-looking but untracked clip. On success, check
`metrics.keyframe_fail_fraction` — a high value means the model struggled even though the
job completed.

To deploy this on a Brev CPU instance, see
[brev/cpu-remote/DEPLOY.md](brev/cpu-remote/DEPLOY.md).

Security limits are configured with `CROPPER_ALLOWED_HOSTS` (exact names or `*.suffix`
wildcards), `CROPPER_ALLOW_PRIVATE_HOSTS`, `CROPPER_MAX_DURATION_SEC`, and
`CROPPER_MAX_OUTPUT_BYTES`. Production deployments should keep private-address access
disabled unless the allowlisted object store is intentionally on a trusted private
network. Generate GET permission for the source and PUT permission for the destination;
the source server should support byte ranges so ffmpeg input seeking can avoid fetching
unneeded media. Compose defaults `CROPPER_ALLOW_PRIVATE_HOSTS` to true so Docker-network
MinIO (used by `verify`) works; set it false for public S3 unless the store is on a
trusted private network.

The container includes ffmpeg and the OpenCV runtime but **no local model bundle or model
server**. Configure either a hosted OpenAI-compatible endpoint with `VCROPPER_BASE_URL`,
`VCROPPER_MODEL`, and `VCROPPER_API_KEY`, or native Bedrock with `VCROPPER_PROVIDER=bedrock`,
`BEDROCK_REGION`/`AWS_REGION`, and optional model/credential variables. On AWS, prefer
injecting credentials through the task or instance role. Cost and throughput scale with
clip length, `sample_fps`, scoreboard sampling, and provider latency; one service instance
processes one job at a time.

Operational endpoints are `/healthz` (process), `/readyz` (Redis, ffmpeg, and model
credentials), `/version`, `/docs`, and `/openapi.json`. A failed PUT commonly means the destination signature lacks PUT
permission, has expired, or S3 returned **307** to another host (this client does not
follow PUT redirects — sign for `s3-<region>.amazonaws.com` when that happens). A `422`
submission commonly means its host is absent from `CROPPER_ALLOWED_HOSTS`.

## Performance (case study)

We measured several backends on a small internal football benchmark (frozen scorer;
higher coverage is better, `1.000` = every labeled action landed inside the crop). The
clips are copyrighted, so the dataset is **not** distributable — but the harness that
produced these numbers ships in [eval/](eval/README.md), so you can reproduce the *shape*
of this table on your own footage.


| Backend                                        | val coverage / worst | test coverage / worst | notes                                         |
| ---------------------------------------------- | -------------------- | --------------------- | --------------------------------------------- |
| Gemini (`gemini-flash-latest`)                 | 0.957 / 0.875        | 1.000 / 1.000         | accuracy leader; matches the R&D v0013 winner |
| NVIDIA `nemotron-3-nano-omni-30b-a3b-reasoning` (**default**) | not re-benchmarked | not re-benchmarked | current hosted default (2026-09); old `nano-12b-v2-vl` is EOL |
| NVIDIA `nemotron-nano-12b-v2-vl` (retired) | 0.872 / 0.750        | 0.818 / 0.500         | 0% parse-fail, smoothest path; below Gemini; **410 Gone** since 2026-08-26 |
| OpenRouter `qwen3-vl-235b`                     | 0.872 / 0.625        | 0.727 / 0.571         | strong, but needs the format-tolerant parser  |


> Coverage depends on the model. The shipped default (**NVIDIA Nemotron**) is chosen for clean
> point-format compliance and smooth paths; **Gemini** remains the recommended provider when you
> want the best tracking accuracy. Numbers vary with your clips, prompt, and sampling.



### Benchmark it yourself

The `eval/` harness lets anyone label their own clips and score the cropper end-to-end —
coverage, worst-clip, smoothness, perf, and cost, plus GT-vs-prediction debug videos. Because
scoring only needs the manifest (not the video), results are reproducible and shareable without
your footage. Point it at an external dataset root via `$VCROPPER_EVAL_DATA`; it is excluded from
the installed wheel. You can benchmark **any config you can run** — a built-in `--sport` preset or
a custom `--prompt`/`--prompt-file` for a new sport — by passing the *same* flags to
`eval/run_eval.py`. See [eval/README.md](eval/README.md) for the bring-your-own-dataset workflow
and a **"benchmark a new sport" quickstart** (manifest → label → ingest → run → score → debug).

## Limitations & troubleshooting

- **Model choice matters.** Accuracy depends on how faithfully the model returns the
`{"x","y"}` point format. Gemini is validated to parity; the parser tolerates common
non-standard variants, but a high keyframe-fail % in the run summary means tracking degraded
to interpolation. Prefer Gemini, or tune the prompt for your model.
- `ffmpeg` **not found** → output falls back to an `mp4v` file (still playable). Install ffmpeg
for H.264: `brew install ffmpeg` (macOS) · `apt install ffmpeg` (Debian/Ubuntu) ·
`choco install ffmpeg` (Windows).
- **"No keyframes were produced"** → the input isn't decodable by OpenCV; re-encode the source.
- **Cost / rate limits** → lower `--sample-fps`, `--send-width`, or `--concurrency`.
  NVIDIA hosted models may **429** on long clips; the job can still succeed if enough
  keyframes parse. Shorten the clip or drop `--sample-fps` if fail rates spike.



## License

```
Copyright 2026 Qvest.US, LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

See [LICENSE](LICENSE), [NOTICE](NOTICE), and [CONTRIBUTING.md](CONTRIBUTING.md) (DCO sign-off).

