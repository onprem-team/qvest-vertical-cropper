"""Unit + adversarial tests for v_cropper.cli (all offline).

The VLM backend is injected via v_cropper.cli.make_backend; render/ffmpeg run for real so
the full CLI -> focus -> render seam is exercised and a real output mp4 is produced.
"""

from __future__ import annotations

import json

import pytest

from conftest import FakeBackend, write_clip
from v_cropper.cli import _env_truthy, _has_api_key, _provider_is_configured, _resolve, _sample_every, main
from v_cropper.prompts import PRESETS

POINT = json.dumps({"x": 500, "y": 500})
KEY_VARS = [
    "VCROPPER_PROVIDER",
    "VCROPPER_API_KEY",
    "GEMINI_API_KEY",
    "VCROPPER_BASE_URL",
    "VCROPPER_MODEL",
    "BEDROCK_REGION",
    "AWS_REGION",
    "VCROPPER_SPORT",
    "VCROPPER_PROMPT",
    "VCROPPER_PROMPT_FILE",
    "VCROPPER_SEND_WIDTH",
    "VCROPPER_SPRING_K",
    "SCOREBOARD_ENABLED",
    "VCROPPER_EXTRA_HEADERS",
]


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for v in KEY_VARS:
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setattr("v_cropper.cli.load_dotenv", lambda *a, **k: None)


def _patch_backends(monkeypatch, *backends):
    """Route v_cropper.cli.make_backend to the given backends in call order.

    The CLI builds the focus backend first, then (if enabled) the scoreboard backend.
    """
    q = list(backends)

    def factory(**kw):
        return q.pop(0) if len(q) > 1 else q[0]

    monkeypatch.setattr("v_cropper.cli.make_backend", factory)
    return backends


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #
class TestResolvers:
    def test_env_truthy(self, monkeypatch):
        for val, expect in [
            ("yes", True),
            ("1", True),
            ("true", True),
            ("TRUE", True),
            ("0", False),
            ("no", False),
            ("", False),
        ]:
            monkeypatch.setenv("X", val)
            assert _env_truthy("X") is expect

    def test_resolve_precedence(self, monkeypatch):
        assert _resolve("cli", "E", 10, int) == "cli"
        monkeypatch.setenv("E", "20")
        assert _resolve(None, "E", 10, int) == 20
        monkeypatch.setenv("E", "bad")
        assert _resolve(None, "E", 10, int) == 10
        monkeypatch.delenv("E")
        assert _resolve(None, "E", 10, int) == 10

    def test_has_api_key(self, monkeypatch):
        assert _has_api_key(None) is False
        assert _has_api_key("k") is True
        monkeypatch.setenv("GEMINI_API_KEY", "g")
        assert _has_api_key(None) is True

    def test_bedrock_configuration_needs_region_not_api_key(self, monkeypatch):
        monkeypatch.setenv("VCROPPER_PROVIDER", "bedrock")
        assert _provider_is_configured(None) is False
        monkeypatch.setenv("BEDROCK_REGION", "us-west-2")
        assert _provider_is_configured(None) is True

    def test_sample_every_explicit_wins(self):
        assert _sample_every(10, 2.0, 60.0) == 10

    def test_sample_every_from_fps(self):
        assert _sample_every(None, 2.0, 60.0) == 30
        assert _sample_every(None, 2.0, 59.94) == 30
        assert _sample_every(None, 3.0, 25.0) == 8  # round(8.33)

    def test_sample_every_fallbacks(self):
        assert _sample_every(None, 2.0, 0.0) == 30  # no fps
        assert _sample_every(None, 0.0, 60.0) == 30  # bad sample_fps
        assert _sample_every(0, 2.0, 60.0) == 1  # clamped to >=1


# --------------------------------------------------------------------------- #
# Error branches
# --------------------------------------------------------------------------- #
class TestErrorBranches:
    def test_missing_video_errors(self, tmp_path, monkeypatch):
        monkeypatch.setenv("VCROPPER_API_KEY", "k")
        with pytest.raises(SystemExit):
            main([str(tmp_path / "nope.mp4")])

    def test_missing_key_errors(self, tmp_path):
        clip = write_clip(tmp_path / "c.mp4", n_frames=10)
        with pytest.raises(SystemExit):
            main([str(clip)])

    def test_unconfigured_bedrock_names_the_region_not_the_api_key(self, tmp_path, monkeypatch, capsys):
        """Under Bedrock the generic 'no API key' advice sends the user down the wrong path."""
        monkeypatch.setenv("VCROPPER_PROVIDER", "bedrock")
        clip = write_clip(tmp_path / "c.mp4", n_frames=10)
        with pytest.raises(SystemExit):
            main([str(clip)])
        message = capsys.readouterr().err
        assert "BEDROCK_REGION" in message
        assert "VCROPPER_API_KEY" not in message

    def test_bad_prompt_file_errors(self, tmp_path, monkeypatch):
        monkeypatch.setenv("VCROPPER_API_KEY", "k")
        clip = write_clip(tmp_path / "c.mp4", n_frames=10)
        with pytest.raises(SystemExit):
            main([str(clip), "--prompt-file", str(tmp_path / "missing.txt")])

    def test_unknown_sport_errors(self, tmp_path, monkeypatch):
        monkeypatch.setenv("VCROPPER_API_KEY", "k")
        clip = write_clip(tmp_path / "c.mp4", n_frames=10)
        with pytest.raises(SystemExit):
            main([str(clip), "--sport", "curling"])

    def test_no_keyframes_errors(self, tmp_path, monkeypatch):
        monkeypatch.setenv("VCROPPER_API_KEY", "k")
        _patch_backends(monkeypatch, FakeBackend(POINT))
        monkeypatch.setattr("v_cropper.cli.extract_focus_points", lambda *a, **k: ({}, {}, 0))
        clip = write_clip(tmp_path / "in.mp4", n_frames=40, w=320, h=180)
        with pytest.raises(SystemExit):
            main([str(clip), "-o", str(tmp_path / "out.mp4"), "--sample-every", "10"])


# --------------------------------------------------------------------------- #
# Main integration
# --------------------------------------------------------------------------- #
class TestMainIntegration:
    def test_base_url_flag_passes_to_backend_factory(self, tmp_path, monkeypatch):
        monkeypatch.setenv("VCROPPER_API_KEY", "k")
        be = FakeBackend(POINT)
        captured = {}

        def factory(**kwargs):
            captured.update(kwargs)
            return be

        monkeypatch.setattr("v_cropper.cli.make_backend", factory)
        clip = write_clip(tmp_path / "in.mp4", n_frames=40, w=320, h=180)
        assert main([str(clip), "-o", str(tmp_path / "out.mp4"), "--sample-every", "10",
                     "--base-url", "http://127.0.0.1:8000/v1"]) == 0
        assert captured["base_url"] == "http://127.0.0.1:8000/v1"

    def test_produces_output(self, tmp_path, monkeypatch):
        monkeypatch.setenv("VCROPPER_API_KEY", "k")
        _patch_backends(monkeypatch, FakeBackend(POINT))
        clip = write_clip(tmp_path / "in.mp4", n_frames=40, w=320, h=180)
        out = tmp_path / "out.mp4"
        assert main([str(clip), "-o", str(out), "--sample-every", "10"]) == 0
        assert out.exists() and out.stat().st_size > 0

    def test_debug_writes_debug_file(self, tmp_path, monkeypatch):
        monkeypatch.setenv("VCROPPER_API_KEY", "k")
        _patch_backends(monkeypatch, FakeBackend(POINT))
        clip = write_clip(tmp_path / "in.mp4", n_frames=40, w=320, h=180)
        out = tmp_path / "out.mp4"
        assert main([str(clip), "-o", str(out), "--debug", "--sample-every", "10"]) == 0
        assert (tmp_path / "in_debug.mp4").exists()

    def test_gemini_key_back_compat(self, tmp_path, monkeypatch):
        # Only GEMINI_API_KEY set -> CLI still runs (back-compat preserved).
        monkeypatch.setenv("GEMINI_API_KEY", "g")
        _patch_backends(monkeypatch, FakeBackend(POINT))
        clip = write_clip(tmp_path / "in.mp4", n_frames=40, w=320, h=180)
        out = tmp_path / "out.mp4"
        assert main([str(clip), "-o", str(out), "--sample-every", "10"]) == 0
        assert out.exists()

    def test_default_prompt_is_football(self, tmp_path, monkeypatch):
        monkeypatch.setenv("VCROPPER_API_KEY", "k")
        be = FakeBackend(POINT)
        _patch_backends(monkeypatch, be)
        clip = write_clip(tmp_path / "in.mp4", n_frames=40, w=320, h=180)
        main([str(clip), "-o", str(tmp_path / "out.mp4"), "--sample-every", "10"])
        assert be.calls[0][0]["content"][1]["text"] == PRESETS["football"]

    def test_sport_flag_selects_preset(self, tmp_path, monkeypatch):
        monkeypatch.setenv("VCROPPER_API_KEY", "k")
        be = FakeBackend(POINT)
        _patch_backends(monkeypatch, be)
        clip = write_clip(tmp_path / "in.mp4", n_frames=40, w=320, h=180)
        main([str(clip), "-o", str(tmp_path / "out.mp4"), "--sample-every", "10", "--sport", "hockey"])
        assert be.calls[0][0]["content"][1]["text"] == PRESETS["hockey"]

    def test_inline_prompt_overrides(self, tmp_path, monkeypatch):
        monkeypatch.setenv("VCROPPER_API_KEY", "k")
        be = FakeBackend(POINT)
        _patch_backends(monkeypatch, be)
        clip = write_clip(tmp_path / "in.mp4", n_frames=40, w=320, h=180)
        main(
            [
                str(clip),
                "-o",
                str(tmp_path / "out.mp4"),
                "--sample-every",
                "10",
                "--prompt",
                "point at the ball x y json",
            ]
        )
        assert be.calls[0][0]["content"][1]["text"] == "point at the ball x y json"

    def test_scoreboard_seam_produces_output(self, tmp_path, monkeypatch):
        monkeypatch.setenv("VCROPPER_API_KEY", "k")
        focus_be = FakeBackend(POINT)
        sb_be = FakeBackend(
            json.dumps(
                {
                    "sport": "hockey",
                    "teams": [{"name": "BOS", "score": 3}, {"name": "MTL", "score": 1}],
                    "period": "2nd",
                    "clock": "10:00",
                }
            )
        )
        _patch_backends(monkeypatch, focus_be, sb_be)
        clip = write_clip(tmp_path / "in.mp4", n_frames=40, w=320, h=180)
        out = tmp_path / "out.mp4"
        rc = main([str(clip), "-o", str(out), "--scoreboard", "--scoreboard-sample-count", "3", "--sample-every", "10"])
        assert rc == 0 and out.exists()
        assert len(sb_be.calls) == 3

    def test_no_readable_scoreboard_skips_overlay(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setenv("VCROPPER_API_KEY", "k")
        _patch_backends(monkeypatch, FakeBackend(POINT), FakeBackend("{}"))
        clip = write_clip(tmp_path / "in.mp4", n_frames=40, w=320, h=180)
        out = tmp_path / "out.mp4"
        rc = main([str(clip), "-o", str(out), "--scoreboard", "--scoreboard-sample-count", "3", "--sample-every", "10"])
        assert rc == 0 and "No readable scoreboard" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
class TestMetrics:
    def test_metrics_json_written(self, tmp_path, monkeypatch):
        monkeypatch.setenv("VCROPPER_API_KEY", "k")
        _patch_backends(monkeypatch, FakeBackend(POINT))
        clip = write_clip(tmp_path / "in.mp4", n_frames=40, w=320, h=180)
        mpath = tmp_path / "metrics.json"
        main([str(clip), "-o", str(tmp_path / "out.mp4"), "--sample-every", "10", "--metrics-json", str(mpath)])
        m = json.loads(mpath.read_text())
        assert set(
            [
                "wall_time_sec",
                "rtf",
                "keyframes_ok",
                "keyframes_failed",
                "focus_usage",
                "est_cost_usd",
                "sport",
                "sample_every",
            ]
        ).issubset(m)
        assert m["keyframes_ok"] == 4  # idx 0/10/20/30
        assert m["sport"] == "football"
        assert m["focus_usage"]["api_calls"] == 4

    def test_summary_line_printed(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setenv("VCROPPER_API_KEY", "k")
        _patch_backends(monkeypatch, FakeBackend(POINT))
        clip = write_clip(tmp_path / "in.mp4", n_frames=40, w=320, h=180)
        main([str(clip), "-o", str(tmp_path / "out.mp4"), "--sample-every", "10"])
        out = capsys.readouterr().out
        assert "keyframes:" in out and "est. cost:" in out


# --------------------------------------------------------------------------- #
# Adversarial pass 1: bug hunt (flag/env precedence, sample-fps math)
# --------------------------------------------------------------------------- #
class TestAdv1Precedence:
    def test_env_sport_used_when_no_flag(self, tmp_path, monkeypatch):
        monkeypatch.setenv("VCROPPER_API_KEY", "k")
        monkeypatch.setenv("VCROPPER_SPORT", "soccer")
        be = FakeBackend(POINT)
        _patch_backends(monkeypatch, be)
        clip = write_clip(tmp_path / "in.mp4", n_frames=40, w=320, h=180)
        main([str(clip), "-o", str(tmp_path / "out.mp4"), "--sample-every", "10"])
        assert be.calls[0][0]["content"][1]["text"] == PRESETS["soccer"]

    def test_flag_sport_beats_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("VCROPPER_API_KEY", "k")
        monkeypatch.setenv("VCROPPER_SPORT", "soccer")
        be = FakeBackend(POINT)
        _patch_backends(monkeypatch, be)
        clip = write_clip(tmp_path / "in.mp4", n_frames=40, w=320, h=180)
        main([str(clip), "-o", str(tmp_path / "out.mp4"), "--sample-every", "10", "--sport", "hockey"])
        assert be.calls[0][0]["content"][1]["text"] == PRESETS["hockey"]


# --------------------------------------------------------------------------- #
# Adversarial pass 2: robustness (partial failures reflected in metrics)
# --------------------------------------------------------------------------- #
class TestAdv2Robustness:
    def test_partial_failure_metrics(self, tmp_path, monkeypatch):
        monkeypatch.setenv("VCROPPER_API_KEY", "k")
        be = FakeBackend([POINT, "garbage", "garbage", POINT])  # 2 ok, 2 fail
        _patch_backends(monkeypatch, be)
        clip = write_clip(tmp_path / "in.mp4", n_frames=40, w=320, h=180)
        mpath = tmp_path / "m.json"
        main(
            [
                str(clip),
                "-o",
                str(tmp_path / "out.mp4"),
                "--sample-every",
                "10",
                "--concurrency",
                "1",
                "--metrics-json",
                str(mpath),
            ]
        )
        m = json.loads(mpath.read_text())
        assert m["keyframes_ok"] == 2 and m["keyframes_failed"] == 2
        assert m["keyframe_fail_fraction"] == 0.5

    def test_high_failure_prints_warning(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setenv("VCROPPER_API_KEY", "k")
        be = FakeBackend(["garbage", "garbage", "garbage", POINT])
        _patch_backends(monkeypatch, be)
        clip = write_clip(tmp_path / "in.mp4", n_frames=40, w=320, h=180)
        rc = main([str(clip), "-o", str(tmp_path / "out.mp4"), "--sample-every", "10", "--concurrency", "1"])
        assert rc == 0 and "failed their VLM call" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# Adversarial pass 3: edge cases (send-width/spring-k plumbed; metrics accuracy)
# --------------------------------------------------------------------------- #
class TestAdv3EdgeCases:
    def test_send_width_and_spring_k_recorded(self, tmp_path, monkeypatch):
        monkeypatch.setenv("VCROPPER_API_KEY", "k")
        _patch_backends(monkeypatch, FakeBackend(POINT))
        clip = write_clip(tmp_path / "in.mp4", n_frames=40, w=320, h=180)
        mpath = tmp_path / "m.json"
        main(
            [
                str(clip),
                "-o",
                str(tmp_path / "out.mp4"),
                "--sample-every",
                "10",
                "--send-width",
                "512",
                "--spring-k",
                "0.08",
                "--metrics-json",
                str(mpath),
            ]
        )
        m = json.loads(mpath.read_text())
        assert m["send_width"] == 512 and m["spring_k"] == 0.08

    def test_env_send_width_and_spring_k(self, tmp_path, monkeypatch):
        monkeypatch.setenv("VCROPPER_API_KEY", "k")
        monkeypatch.setenv("VCROPPER_SEND_WIDTH", "640")
        monkeypatch.setenv("VCROPPER_SPRING_K", "0.03")
        _patch_backends(monkeypatch, FakeBackend(POINT))
        clip = write_clip(tmp_path / "in.mp4", n_frames=40, w=320, h=180)
        mpath = tmp_path / "m.json"
        main([str(clip), "-o", str(tmp_path / "out.mp4"), "--sample-every", "10", "--metrics-json", str(mpath)])
        m = json.loads(mpath.read_text())
        assert m["send_width"] == 640 and m["spring_k"] == 0.03
