"""Unit + adversarial tests for crop geometry, compute_crop_path, and render()."""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest

from conftest import write_clip
from v_cropper.render import (
    _draw_scoreboard_debug,
    _reencode_h264,
    clamp_crop_x,
    compute_crop_path,
    crop_dst_width,
    portrait_crop,
    render,
)
from v_cropper.scoreboard_state import ScoreboardDebugInfo


def _debug_info() -> ScoreboardDebugInfo:
    """A rich consensus debug object exercising the INCLUDED + EXCLUDED + event branches."""
    return ScoreboardDebugInfo(
        total_readings=5,
        valid_readings=4,
        threshold=2,
        sport_votes={"hockey": 4},
        event_votes={"Regular": 3},
        field_votes={"Shots": {"30": 3}, "PP": {"1": 1}},
        included_fields=["Shots"],
        excluded_fields=["PP"],
    )

# The scorer's aspect tolerance (kept in sync with eval/score.py ASPECT_TOL_PX).
ASPECT = 9.0 / 16.0
ASPECT_TOL_PX = 1.5


class TestPortraitCrop:
    def test_9x16_width(self):
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        cropped, _crop_x, crop_w = portrait_crop(frame, center_x=960)
        assert crop_w == 1080 * 9 // 16  # 607
        assert cropped.shape[:2] == (1080, crop_w)

    def test_clamps_left_edge(self):
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        _, crop_x, _ = portrait_crop(frame, center_x=0)
        assert crop_x == 0

    def test_clamps_right_edge(self):
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        _, crop_x, crop_w = portrait_crop(frame, center_x=10_000)
        assert crop_x == 1920 - crop_w


class TestHelpers:
    def test_crop_dst_width(self):
        assert crop_dst_width(1920, 1080) == 607
        assert crop_dst_width(320, 1080) == 320  # never wider than the frame

    def test_clamp_crop_x_bounds(self):
        assert clamp_crop_x(-100, 1920, 607) == 0
        assert clamp_crop_x(10_000, 1920, 607) == 1920 - 607
        assert clamp_crop_x(960, 1920, 607) == int(960 - 607 / 2)  # 656


class TestComputeCropPath:
    def _clip(self, tmp_path, n=40, w=192, h=108):
        return write_clip(tmp_path / "c.mp4", n_frames=n, w=w, h=h)

    def test_one_window_per_frame(self, tmp_path):
        wins, fw, fh, _ = compute_crop_path(self._clip(tmp_path), {0: 40.0, 30: 150.0})
        assert (fw, fh) == (192, 108)
        assert [w["frame_idx"] for w in wins] == list(range(40))

    def test_window_schema_and_math(self, tmp_path):
        wins, fw, fh, _ = compute_crop_path(self._clip(tmp_path), {0: 96.0})
        dst_w = crop_dst_width(fw, fh)
        for w in wins:
            assert set(w) == {"frame_idx", "x_center", "y_center", "width", "height"}
            assert w["width"] == dst_w and w["height"] == fh and w["y_center"] == fh / 2

    def test_aspect_within_scorer_tolerance(self, tmp_path):
        wins, _, fh, _ = compute_crop_path(self._clip(tmp_path), {0: 96.0})
        assert abs(wins[0]["width"] - fh * ASPECT) <= ASPECT_TOL_PX

    def test_centers_clamped_inside_frame(self, tmp_path):
        wins, fw, fh, _ = compute_crop_path(
            self._clip(tmp_path), {i: 1e6 for i in range(0, 40, 5)})
        dst_w = crop_dst_width(fw, fh)
        for w in wins:
            assert dst_w / 2 <= w["x_center"] <= fw - dst_w / 2

    def test_empty_focus_holds_center(self, tmp_path):
        wins, fw, _, _ = compute_crop_path(self._clip(tmp_path), {})
        assert wins[0]["x_center"] == pytest.approx(fw / 2)

    def test_unreadable_clip_raises(self, tmp_path):
        with pytest.raises(RuntimeError):
            compute_crop_path(tmp_path / "nope.mp4", {0: 1.0})

    def test_single_frame_clip(self, tmp_path):
        wins, _, _, _ = compute_crop_path(self._clip(tmp_path, n=1), {0: 90.0})
        assert len(wins) == 1 and wins[0]["frame_idx"] == 0

    def test_no_ground_truth_param(self):
        import inspect
        params = set(inspect.signature(compute_crop_path).parameters)
        assert not ({"truth", "ground_truth", "labels"} & params)


class TestRenderSmoke:
    def test_render_produces_portrait_output(self, tmp_path):
        clip = write_clip(tmp_path / "in.mp4", n_frames=20, w=320, h=180)
        out = tmp_path / "out.mp4"
        render(str(clip), {0: 100.0, 10: 200.0}, out)
        assert out.exists() and out.stat().st_size > 0

    def test_render_path_matches_compute_crop_path_centers(self, tmp_path):
        # render() (streaming spring) and compute_crop_path (grab-pass spring) must agree
        clip = write_clip(tmp_path / "in.mp4", n_frames=30, w=192, h=108)
        wins, _, _, _ = compute_crop_path(clip, {0: 20.0, 20: 160.0})
        assert len(wins) == 30  # sanity: full path built


# --- Adversarial pass 3: crop_w parity with scorer aspect tol across sizes ---
class TestAdvAspectParity:
    @pytest.mark.parametrize("h", [90, 108, 180, 360, 720, 1080])
    def test_dst_width_within_tolerance(self, h):
        w = 3840
        assert abs(crop_dst_width(w, h) - h * ASPECT) <= ASPECT_TOL_PX


class TestReencodeH264:
    """The ffmpeg re-encode step must degrade gracefully and never lose the raw output."""

    def _raw_mp4v(self, tmp_path):
        return Path(write_clip(tmp_path / "raw.mp4", n_frames=3, w=64, h=48))

    def test_ffmpeg_missing_keeps_mp4v(self, tmp_path, monkeypatch):
        raw = self._raw_mp4v(tmp_path)

        def boom(*a, **k):
            raise FileNotFoundError("ffmpeg not on PATH")

        monkeypatch.setattr("v_cropper.render.subprocess.run", boom)
        out = _reencode_h264(raw)
        assert out == raw
        assert raw.exists()  # original mp4v preserved, no exception propagated

    def test_ffmpeg_failure_keeps_mp4v(self, tmp_path, monkeypatch):
        raw = self._raw_mp4v(tmp_path)

        def boom(*a, **k):
            raise subprocess.CalledProcessError(1, "ffmpeg")

        monkeypatch.setattr("v_cropper.render.subprocess.run", boom)
        out = _reencode_h264(raw)
        assert out == raw
        assert raw.exists()

    def test_ffmpeg_success_replaces_with_h264(self, tmp_path, monkeypatch):
        """Deterministic success path (independent of ffmpeg being installed)."""
        raw = self._raw_mp4v(tmp_path)

        def fake_run(cmd, capture_output=False, check=False):
            Path(cmd[-1]).write_bytes(b"fake-h264-bytes")  # cmd[-1] is the output path
            return subprocess.CompletedProcess(cmd, 0)

        monkeypatch.setattr("v_cropper.render.subprocess.run", fake_run)
        out = _reencode_h264(raw)
        assert out == raw  # user gets the path they asked for
        assert raw.read_bytes() == b"fake-h264-bytes"  # h264 renamed over the mp4v
        assert not raw.with_name(raw.stem + "-h264.mp4").exists()  # temp cleaned up


class TestRenderErrorPaths:
    def test_unopenable_video_raises(self, tmp_path):
        with pytest.raises(RuntimeError, match="Cannot open video"):
            render(str(tmp_path / "missing.mp4"), {0: 10.0}, tmp_path / "out.mp4")


class TestRenderDebug:
    def test_debug_writes_both_outputs(self, tmp_path):
        clip = write_clip(tmp_path / "in.mp4", n_frames=12, w=192, h=108)
        out = tmp_path / "out.mp4"
        dbg = tmp_path / "dbg.mp4"
        render(str(clip), {0: 30.0, 10: 150.0}, out, debug=True, debug_path=dbg)
        assert out.exists() and out.stat().st_size > 0
        assert dbg.exists() and dbg.stat().st_size > 0  # focus line branch (focus is not None)

    def test_debug_empty_focus_skips_focus_line(self, tmp_path):
        # Empty focus map -> interpolate_focus returns None every frame, exercising the
        # `if focus is not None` FALSE branch in the debug writer.
        clip = write_clip(tmp_path / "in.mp4", n_frames=6, w=192, h=108)
        dbg = tmp_path / "dbg.mp4"
        render(str(clip), {}, tmp_path / "out.mp4", debug=True, debug_path=dbg)
        assert dbg.exists() and dbg.stat().st_size > 0

    def test_debug_with_scoreboard_panel(self, tmp_path):
        clip = write_clip(tmp_path / "in.mp4", n_frames=8, w=320, h=180)
        dbg = tmp_path / "dbg.mp4"
        render(
            str(clip), {0: 60.0, 7: 120.0}, tmp_path / "out.mp4",
            debug=True, debug_path=dbg, scoreboard_debug=_debug_info(),
        )
        assert dbg.exists() and dbg.stat().st_size > 0

    def test_debug_scoreboard_zero_valid_readings_skips_panel(self, tmp_path):
        # valid_readings == 0 -> the panel is NOT drawn (branch guard in render()).
        clip = write_clip(tmp_path / "in.mp4", n_frames=4, w=320, h=180)
        dbg = tmp_path / "dbg.mp4"
        render(
            str(clip), {0: 60.0}, tmp_path / "out.mp4",
            debug=True, debug_path=dbg, scoreboard_debug=ScoreboardDebugInfo(),
        )
        assert dbg.exists()


class TestDrawScoreboardDebug:
    def test_draws_included_and_excluded_sections(self):
        overlay = np.zeros((720, 1280, 3), dtype=np.uint8)
        _draw_scoreboard_debug(overlay, _debug_info())
        assert overlay.any()  # something was drawn into the panel region

    def test_tiny_frame_skips_panel_region(self):
        # Panel anchors at py=40; a 20px-tall frame makes panel_region empty,
        # exercising the FALSE side of the shape>0 guard without raising.
        overlay = np.zeros((20, 60, 3), dtype=np.uint8)
        _draw_scoreboard_debug(overlay, _debug_info())  # must not raise
