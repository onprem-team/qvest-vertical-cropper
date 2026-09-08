"""Opt-in LIVE smoke tests against real provider APIs.

These are the tests you run with real credentials. They are marked `live` and
are EXCLUDED from the default suite (see pyproject addopts). They only run when:

    VCROPPER_LIVE_TEST=1  and  at least one provider key is present.

Each provider whose key is set is validated in one run — this fans out over EVERY provider
key present in the environment (including any loaded from a shell profile or `.env`), so to
test a single provider, run with only that one key set (e.g. `env -u GEMINI_API_KEY
-u OPENROUTER_API_KEY ...`). Example (NVIDIA only):

    VCROPPER_LIVE_TEST=1 \
    NVIDIA_API_KEY=nvapi-... \
    uv run pytest -m live -v

Optional overrides: NVIDIA_MODEL, OPENAI_MODEL, GEMINI_MODEL, OPENROUTER_MODEL,
NVIDIA_SINGLE_IMAGE_MODEL (single-image model check), and VCROPPER_LIVE_CLIP (path to
a real landscape clip; defaults to ./hockey.mp4).
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from v_cropper.backend import (
    GEMINI_COMPAT_BASE_URL,
    NVIDIA_BASE_URL,
    BackendConfig,
    OpenAICompatBackend,
)

pytestmark = pytest.mark.live

_LIVE_ENABLED = os.environ.get("VCROPPER_LIVE_TEST") == "1"


# (provider_id, base_url, key_env, default_model)
_PROVIDERS = [
    ("nvidia", NVIDIA_BASE_URL, "NVIDIA_API_KEY", "nvidia/nemotron-nano-12b-v2-vl"),
    ("openai", "https://api.openai.com/v1", "OPENAI_API_KEY", "gpt-4o"),
    ("gemini", GEMINI_COMPAT_BASE_URL, "GEMINI_API_KEY", "gemini-2.5-flash"),
    ("openrouter", "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY",
     "qwen/qwen3-vl-235b-a22b-instruct"),
]


def _available():
    out = []
    for pid, base, key_env, default_model in _PROVIDERS:
        key = os.environ.get(key_env)
        if key:
            model = os.environ.get(f"{pid.upper()}_MODEL", default_model)
            out.append((pid, base, key, model))
    return out


def _params():
    avail = _available()
    if not avail:
        return [pytest.param(None, marks=pytest.mark.skip(
            reason="no provider key set (NVIDIA_API_KEY / OPENAI_API_KEY / GEMINI_API_KEY / OPENROUTER_API_KEY)"))]
    return [pytest.param(p, id=p[0]) for p in avail]


def _clip_path():
    p = Path(os.environ.get("VCROPPER_LIVE_CLIP", "hockey.mp4"))
    if not p.exists():
        pytest.skip(f"live clip not found: {p} (set VCROPPER_LIVE_CLIP)")
    return str(p)


def _backend(base, key, model):
    return OpenAICompatBackend(BackendConfig(base_url=base, api_key=key, model=model))


@pytest.mark.skipif(not _LIVE_ENABLED, reason="set VCROPPER_LIVE_TEST=1 to run live tests")
@pytest.mark.parametrize("provider", _params())
def test_live_focus(provider):
    from v_cropper.focus import extract_focus_points

    pid, base, key, model = provider
    clip = _clip_path()
    be = _backend(base, key, model)
    focus_map, reasoning, n_failed = extract_focus_points(clip, backend=be, sample_every=15, concurrency=4)
    total = len(focus_map) + n_failed
    assert total > 0, f"[{pid}] no keyframes produced"
    assert len(focus_map) >= 1, f"[{pid}] no valid points returned"
    # All points map to in-frame pixels.
    assert all(px >= 0 for px in focus_map.values())
    # Tolerate some failures but flag a broken backend.
    assert n_failed / total < 0.5, f"[{pid}] {n_failed}/{total} keyframes failed"


@pytest.mark.skipif(not _LIVE_ENABLED, reason="set VCROPPER_LIVE_TEST=1 to run live tests")
@pytest.mark.parametrize("provider", _params())
def test_live_scoreboard(provider):
    from v_cropper.scoreboard_ocr import extract_scoreboard, sample_frames

    pid, base, key, model = provider
    clip = _clip_path()
    frames = sample_frames(clip, sample_count=4)
    be = _backend(base, key, model)
    readings = extract_scoreboard(frames, backend=be)
    assert len(readings) == len(frames), f"[{pid}] wrong number of readings"
    # At least one frame should yield a plausible (non-empty) reading.
    assert any(r.confidence > 0 for r in readings), f"[{pid}] no plausible scoreboard reading"
    assert be.usage_totals["api_calls"] >= 1


@pytest.mark.skipif(not _LIVE_ENABLED, reason="set VCROPPER_LIVE_TEST=1 to run live tests")
@pytest.mark.skipif("NVIDIA_API_KEY" not in os.environ, reason="needs NVIDIA_API_KEY")
def test_live_single_image_model():
    """The point prompt sends exactly one image per call, so single-image models work natively."""
    from v_cropper.focus import extract_focus_points

    model = os.environ.get("NVIDIA_SINGLE_IMAGE_MODEL", "meta/llama-3.2-90b-vision-instruct")
    clip = _clip_path()
    be = _backend(NVIDIA_BASE_URL, os.environ["NVIDIA_API_KEY"], model)
    focus_map, _r, n_failed = extract_focus_points(clip, backend=be, sample_every=15, concurrency=4)
    total = len(focus_map) + n_failed
    assert total > 0
    assert len(focus_map) >= 1, "single-image model returned no valid points"
