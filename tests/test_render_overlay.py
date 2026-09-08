"""Integration-ish test for render() with and without a scoreboard overlay.

Offline: builds a synthetic landscape clip and a fake GameState, runs the full
crop/render loop, and checks the output is a valid 9:16 video. No Gemini needed.
"""
from __future__ import annotations

import cv2
import numpy as np

from v_cropper.render import render
from v_cropper.scoreboard_render import OverlayStyle
from v_cropper.scoreboard_state import GameState


def _write_landscape_clip(path, n_frames=20, w=320, h=180):
    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 30.0, (w, h))
    for i in range(n_frames):
        frame = np.full((h, w, 3), 90, dtype=np.uint8)
        # a moving bright square to give the focus map something to track
        x = (i * 10) % (w - 20)
        frame[40:60, x:x + 20] = (255, 255, 255)
        writer.write(frame)
    writer.release()
    return path


def _probe_dims(path):
    cap = cv2.VideoCapture(str(path))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return w, h, n


def _assert_portrait(w, h, n):
    assert h == 180
    assert n > 0
    # 9:16 portrait; h264 rounds odd widths to even, so allow +/-1 px.
    assert abs(w - (180 * 9 // 16)) <= 1
    assert w % 2 == 0  # even for yuv420p h264


def test_render_without_overlay(tmp_path):
    clip = _write_landscape_clip(tmp_path / "in.mp4")
    out = tmp_path / "out.mp4"
    focus_map = {0: 50.0, 10: 200.0}
    render(str(clip), focus_map, out)
    assert out.exists()
    _assert_portrait(*_probe_dims(out))


def test_render_with_overlay(tmp_path):
    clip = _write_landscape_clip(tmp_path / "in.mp4")
    out = tmp_path / "out_sb.mp4"
    focus_map = {0: 50.0, 10: 200.0}
    state = GameState(
        sport="hockey", home_team="BOS", away_team="MTL",
        home_score=3, away_score=1, period="2nd", clock="14:32",
    )
    render(str(clip), focus_map, out, game_state=state, overlay_style=OverlayStyle())
    assert out.exists()
    _assert_portrait(*_probe_dims(out))


def test_overlay_concentrated_in_bottom_band(tmp_path):
    """The overlay change must be concentrated in the bottom band, not the top.

    h264 is lossy, so we compare *relative* difference magnitude (bottom band vs
    top region) rather than exact pixel equality.
    """
    clip = _write_landscape_clip(tmp_path / "in.mp4")
    focus_map = {0: 100.0, 10: 100.0}  # static focus so crops align frame-for-frame

    plain = tmp_path / "plain.mp4"
    sb = tmp_path / "sb.mp4"
    render(str(clip), focus_map, plain)
    state = GameState(
        sport="hockey", home_team="BOS", away_team="MTL",
        home_score=3, away_score=1, period="2nd", clock="14:32",
    )
    render(str(clip), focus_map, sb, game_state=state, overlay_style=OverlayStyle())

    cap_p = cv2.VideoCapture(str(plain))
    cap_s = cv2.VideoCapture(str(sb))
    okp, fp = cap_p.read()
    oks, fs = cap_s.read()
    cap_p.release()
    cap_s.release()
    assert okp and oks
    assert fp.shape == fs.shape

    band = int(fp.shape[0] * 0.16)
    diff = np.abs(fp.astype(np.int16) - fs.astype(np.int16))
    bottom_diff = diff[-band:].mean()
    top_diff = diff[: fp.shape[0] - band].mean()
    # Overlay band should be clearly changed and much more than codec noise up top.
    assert bottom_diff > 2.0
    assert bottom_diff > top_diff * 3
