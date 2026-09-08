"""Phase 6 cross-seam integration + failure-injection tests (all offline).

Drives the whole CLI -> focus -> scoreboard -> consensus -> render pipeline on a
synthetic clip with injected FakeBackends, probing graceful degradation when the
VLM fails partially or completely. Real ffmpeg/OpenCV run; no network.
"""
from __future__ import annotations

import json
import threading

import pytest

from conftest import FakeBackend, write_clip
from v_cropper.cli import main

pytestmark = pytest.mark.integration

GOOD_POINT = json.dumps({"x": 150, "y": 500})
MATCHUP = json.dumps({
    "sport": "hockey",
    "teams": [{"name": "BOS", "score": 3}, {"name": "MTL", "score": 1}],
    "period": "2nd", "clock": "10:00",
})


class _RaiseOnCallsBackend:
    """Backend that raises on a chosen set of (0-based) call indices."""

    def __init__(self, content, raise_on, *, model="fake"):
        self.model = model
        self._content = content
        self._raise_on = set(raise_on)
        self._n = 0
        self._lock = threading.Lock()
        self.last_usage = {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
        self.usage_totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "api_calls": 0}
        self.calls = []

    def complete(self, messages, *, temperature=0.1, response_format=None,
                 max_tokens=None, extra_body=None):
        with self._lock:
            i = self._n
            self._n += 1
            self.calls.append(messages)
        if i in self._raise_on:
            raise RuntimeError(f"injected failure on call {i}")
        with self._lock:
            self.usage_totals["api_calls"] += 1
            self.usage_totals["total_tokens"] += 2
        return self._content


def _run(clip, out, monkeypatch, focus_backend, sb_backend=None, extra_args=()):
    monkeypatch.setenv("VCROPPER_API_KEY", "k")
    monkeypatch.setattr("v_cropper.cli.load_dotenv", lambda *a, **k: None)
    # The CLI builds the focus backend first, then (if enabled) the scoreboard backend.
    q = [focus_backend] + ([sb_backend] if sb_backend is not None else [])

    def factory(**kw):
        return q.pop(0) if len(q) > 1 else q[0]

    monkeypatch.setattr("v_cropper.cli.make_backend", factory)
    argv = [str(clip), "-o", str(out), "--sample-every", "10", "--concurrency", "1", *extra_args]
    return main(argv)


def test_happy_path_full_pipeline_debug(tmp_path, monkeypatch, capsys):
    clip = write_clip(tmp_path / "in.mp4", n_frames=40, w=320, h=180)
    out = tmp_path / "out.mp4"
    rc = _run(clip, out, monkeypatch, FakeBackend(GOOD_POINT), FakeBackend(MATCHUP),
              extra_args=["--scoreboard", "--scoreboard-sample-count", "3", "--debug"])
    assert rc == 0
    assert out.exists() and out.stat().st_size > 0
    assert (tmp_path / "in_debug.mp4").exists()
    printed = capsys.readouterr().out
    assert "Scoreboard: hockey" in printed
    # Every focus keyframe returned a valid point -> zero failures on the happy path.
    assert "0 failed" in printed


def test_metrics_json_end_to_end(tmp_path, monkeypatch):
    clip = write_clip(tmp_path / "in.mp4", n_frames=40, w=320, h=180)
    out = tmp_path / "out.mp4"
    mpath = tmp_path / "metrics.json"
    rc = _run(clip, out, monkeypatch, FakeBackend(GOOD_POINT),
              extra_args=["--metrics-json", str(mpath)])
    assert rc == 0
    m = json.loads(mpath.read_text())
    assert m["keyframes_ok"] == 4 and m["keyframes_failed"] == 0
    assert m["sport"] == "football"
    assert m["focus_usage"]["api_calls"] == 4


def test_partial_focus_failures_still_render(tmp_path, monkeypatch, capsys):
    clip = write_clip(tmp_path / "in.mp4", n_frames=40, w=320, h=180)
    out = tmp_path / "out.mp4"
    # 4 keyframes; fail calls 0 and 2 -> 50% failure, still renders from the rest.
    focus_be = _RaiseOnCallsBackend(GOOD_POINT, raise_on={0, 2})
    rc = _run(clip, out, monkeypatch, focus_be)
    assert rc == 0
    assert out.exists() and out.stat().st_size > 0
    assert len(focus_be.calls) == 4
    assert "failed their VLM call" in capsys.readouterr().out


def test_scoreboard_exception_skips_overlay(tmp_path, monkeypatch, capsys):
    clip = write_clip(tmp_path / "in.mp4", n_frames=40, w=320, h=180)
    out = tmp_path / "out.mp4"
    sb_be = FakeBackend(MATCHUP, raise_exc=RuntimeError("scoreboard down"))
    rc = _run(clip, out, monkeypatch, FakeBackend(GOOD_POINT), sb_be,
              extra_args=["--scoreboard", "--scoreboard-sample-count", "3"])
    assert rc == 0
    assert out.exists() and out.stat().st_size > 0
    assert "No readable scoreboard" in capsys.readouterr().out


def test_all_focus_fail_center_crop(tmp_path, monkeypatch, capsys):
    clip = write_clip(tmp_path / "in.mp4", n_frames=40, w=320, h=180)
    out = tmp_path / "out.mp4"
    # Every focus call raises -> empty focus map -> render falls back to center crop.
    focus_be = FakeBackend(GOOD_POINT, raise_exc=RuntimeError("all down"))
    rc = _run(clip, out, monkeypatch, focus_be)
    assert rc == 0
    assert out.exists() and out.stat().st_size > 0
    assert "failed their VLM call" in capsys.readouterr().out


def test_backend_totals_survive_mixed_success_failure(tmp_path, monkeypatch):
    """Usage accounting stays correct when some scoreboard calls fail."""
    from v_cropper.scoreboard_ocr import extract_scoreboard, sample_frames

    clip = write_clip(tmp_path / "in.mp4", n_frames=30)
    frames = sample_frames(str(clip), sample_count=4)
    sb_be = _RaiseOnCallsBackend(MATCHUP, raise_on={1})  # 1 of 4 fails
    readings = extract_scoreboard(frames, backend=sb_be)
    assert len(readings) == 4
    assert sum(1 for r in readings if r.confidence > 0) == 3
    assert sb_be.usage_totals["api_calls"] == 3  # only successes counted


# --- cross-seam helpers ---------------------------------------------------

def _text_of(messages):
    """Pull the text-part string out of a focus `complete(messages)` payload."""
    parts = messages[0]["content"]
    return next(p["text"] for p in parts if p.get("type") == "text")


# --- Phase 5 ADVERSARIAL 1 (cross-seam bug hunt) --------------------------

def test_sport_flag_reaches_focus_backend_message(tmp_path, monkeypatch):
    """The CLI's resolved prompt (via --sport) must be the exact text sent to the VLM."""
    from v_cropper.prompts import PRESETS

    clip = write_clip(tmp_path / "in.mp4", n_frames=40, w=320, h=180)
    be = FakeBackend(GOOD_POINT)
    rc = _run(clip, tmp_path / "o.mp4", monkeypatch, be, extra_args=["--sport", "basketball"])
    assert rc == 0
    sent = _text_of(be.calls[0])
    assert sent == PRESETS["basketball"]
    assert sent != PRESETS["football"]


def test_focus_generation_params_forwarded(tmp_path, monkeypatch):
    """temperature=0.0 and the point token cap must survive the CLI->focus->backend seam."""
    from v_cropper.focus import POINT_MAX_TOKENS

    clip = write_clip(tmp_path / "in.mp4", n_frames=40, w=320, h=180)
    be = FakeBackend(GOOD_POINT)
    assert _run(clip, tmp_path / "o.mp4", monkeypatch, be) == 0
    kw = be.call_kwargs[0]
    assert kw["temperature"] == 0.0
    assert kw["max_tokens"] == POINT_MAX_TOKENS


def test_inline_prompt_overrides_sport_across_seam(tmp_path, monkeypatch):
    clip = write_clip(tmp_path / "in.mp4", n_frames=40, w=320, h=180)
    be = FakeBackend(GOOD_POINT)
    custom = 'point at the mascot; reply {"x":<0-1000>,"y":<0-1000>} json'
    rc = _run(clip, tmp_path / "o.mp4", monkeypatch, be,
              extra_args=["--sport", "hockey", "--prompt", custom])
    assert rc == 0
    assert _text_of(be.calls[0]) == custom


# --- Phase 5 ADVERSARIAL 2 (cross-seam robustness) ------------------------

def test_focus_and_scoreboard_use_independent_backends(tmp_path, monkeypatch):
    """Concurrent focus + scoreboard passes must each hit their own backend/usage tally."""
    clip = write_clip(tmp_path / "in.mp4", n_frames=40, w=320, h=180)
    focus_be = FakeBackend(GOOD_POINT)
    sb_be = FakeBackend(MATCHUP)
    mpath = tmp_path / "m.json"
    rc = _run(clip, tmp_path / "o.mp4", monkeypatch, focus_be, sb_be,
              extra_args=["--scoreboard", "--scoreboard-sample-count", "3",
                          "--metrics-json", str(mpath)])
    assert rc == 0
    # No cross-contamination: focus saw point prompts, scoreboard saw its own frames.
    assert focus_be.usage_totals["api_calls"] == 4
    assert sb_be.usage_totals["api_calls"] == 3
    m = json.loads(mpath.read_text())
    assert m["focus_usage"]["api_calls"] == 4
    assert m["scoreboard_usage"]["api_calls"] == 3
    # est_cost must fold in BOTH backends' tokens.
    assert m["est_cost_usd"] > 0


def test_partial_focus_failure_reflected_in_metrics(tmp_path, monkeypatch):
    """A failing seam must degrade gracefully AND be reported honestly in metrics."""
    clip = write_clip(tmp_path / "in.mp4", n_frames=40, w=320, h=180)
    focus_be = _RaiseOnCallsBackend(GOOD_POINT, raise_on={1})  # 1 of 4 fails
    mpath = tmp_path / "m.json"
    rc = _run(clip, tmp_path / "o.mp4", monkeypatch, focus_be,
              extra_args=["--metrics-json", str(mpath)])
    assert rc == 0
    m = json.loads(mpath.read_text())
    assert m["keyframes_ok"] == 3
    assert m["keyframes_failed"] == 1
    assert abs(m["keyframe_fail_fraction"] - 0.25) < 1e-9


# --- Phase 5 ADVERSARIAL 3 (cross-seam edge cases) ------------------------

def test_unreadable_video_exits_cleanly(tmp_path, monkeypatch):
    """An empty/garbage clip yields zero keyframes -> CLI errors instead of crashing render."""
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"not a video")
    with pytest.raises(SystemExit) as ei:
        _run(bad, tmp_path / "o.mp4", monkeypatch, FakeBackend(GOOD_POINT))
    assert ei.value.code != 0


def test_send_width_and_spring_k_plumbed_to_metrics(tmp_path, monkeypatch):
    clip = write_clip(tmp_path / "in.mp4", n_frames=40, w=320, h=180)
    mpath = tmp_path / "m.json"
    rc = _run(clip, tmp_path / "o.mp4", monkeypatch, FakeBackend(GOOD_POINT),
              extra_args=["--send-width", "256", "--spring-k", "3.5",
                          "--metrics-json", str(mpath)])
    assert rc == 0
    m = json.loads(mpath.read_text())
    assert m["send_width"] == 256
    assert m["spring_k"] == 3.5
