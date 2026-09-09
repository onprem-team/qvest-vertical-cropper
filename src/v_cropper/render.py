# SPDX-FileCopyrightText: Copyright (c) 2026 Qvest.US, LLC
# SPDX-License-Identifier: Apache-2.0
"""Frame loop: interpolate focus → pan → 9:16 crop → write (+optional debug) → ffmpeg h264."""

import logging
import subprocess
from pathlib import Path

import cv2
import numpy as np

from .focus import interpolate_focus
from .scoreboard_render import compose_scoreboard
from .smoothing import DEFAULT_SPRING_K, SpringPanner

logger = logging.getLogger(__name__)


def _draw_scoreboard_debug(overlay, debug):
    """Render the scoreboard OCR consensus panel on the debug overlay frame."""
    h, w = overlay.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness, line_h = 0.40, 1, 16
    color_header = (0, 255, 255)
    color_included = (0, 220, 100)
    color_excluded = (100, 100, 255)
    color_info = (200, 200, 200)

    lines = debug.to_lines()
    panel_h = (len(lines) + 1) * line_h + 10
    panel_w = 420
    px, py = w - panel_w - 10, 40

    panel_region = overlay[py:py + panel_h, px:px + panel_w]
    if panel_region.shape[0] > 0 and panel_region.shape[1] > 0:
        dark = np.zeros_like(panel_region)
        cv2.addWeighted(dark, 0.75, panel_region, 0.25, 0, panel_region)
        overlay[py:py + panel_h, px:px + panel_w] = panel_region

    y = py + 14
    in_excluded_section = False
    for line in lines:
        if line.startswith("--- EXCLUDED"):
            in_excluded_section = True
            color = color_excluded
        elif line.startswith("--- INCLUDED"):
            in_excluded_section = False
            color = color_included
        elif line.startswith("SCOREBOARD"):
            color = color_header
        elif line.startswith("  "):
            color = color_excluded if in_excluded_section else color_included
        else:
            color = color_info
        cv2.putText(overlay, line[:60], (px + 6, y), font, scale, color, thickness, cv2.LINE_AA)
        y += line_h


def crop_dst_width(frame_w, frame_h):
    """Width of the 9:16 crop for a given frame size (never wider than the frame)."""
    return min(frame_h * 9 // 16, frame_w)


def clamp_crop_x(center_x, frame_w, dst_w):
    """Left edge of a dst_w-wide crop centered on center_x, clamped inside the frame."""
    return max(0, min(int(center_x - dst_w / 2), frame_w - dst_w))


def portrait_crop(frame, center_x):
    """Crop a 9:16 vertical slice centered on center_x (clamped to frame). Returns (crop, x, w)."""
    h, w = frame.shape[:2]
    dst_w = crop_dst_width(w, h)
    crop_x = clamp_crop_x(center_x, w, dst_w)
    return frame[0:h, crop_x:crop_x + dst_w], crop_x, dst_w


def compute_crop_path(video_path, focus_map_px, *, spring_k=DEFAULT_SPRING_K):
    """Per-frame 9:16 crop windows for a clip (interpolate -> spring -> clamp).

    Decodes only to count frames + read dims (pixels aren't needed for the path), so this
    is the single source of crop geometry shared by ``render`` semantics and the eval
    adapter. Returns (windows, frame_w, frame_h, fps) with exactly one window per decoded
    frame ``0..n-1``: ``{frame_idx, x_center, y_center, width, height}`` (scorer schema).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    dst_w = crop_dst_width(frame_w, frame_h)
    spring = SpringPanner(k=spring_k)
    windows = []
    idx = 0
    while cap.grab():  # grab-only: we need frame counts/timing, not pixels
        spring.update(interpolate_focus(focus_map_px, idx))
        crop_x = clamp_crop_x(spring.center_x(frame_w), frame_w, dst_w)
        windows.append({
            "frame_idx": idx,
            "x_center": crop_x + dst_w / 2.0,
            "y_center": frame_h / 2.0,
            "width": dst_w,
            "height": frame_h,
        })
        idx += 1
    cap.release()
    return windows, frame_w, frame_h, fps


def _reencode_h264(raw_path: Path) -> Path:
    """Re-encode an mp4v file to h264 in place (returns the h264 path). Keeps raw if ffmpeg missing."""
    h264 = raw_path.with_name(raw_path.stem + "-h264.mp4")
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(raw_path), "-c:v", "libx264", "-preset", "fast",
             "-crf", "23", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-pix_fmt", "yuv420p",
             "-movflags", "+faststart", str(h264)],
            capture_output=True, check=True,
        )
        raw_path.unlink()
        h264.rename(raw_path)  # present the user the path they asked for
        return raw_path
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        logger.warning("ffmpeg unavailable/failed (%s) — leaving mp4v output: %s", e, raw_path)
        return raw_path


def render(video_path, focus_map_px, out_path, *, debug=False, debug_path=None,
           spring_k=DEFAULT_SPRING_K, game_state=None, overlay_style=None,
           scoreboard_debug=None):
    """Render the vertical crop. focus_map_px is {keyframe_idx: focus_x_pixels}.

    The crop center is smoothed with a critically-damped spring (spring_k). If game_state
    is provided, a scoreboard overlay is composited onto each cropped frame using
    overlay_style (defaults applied when None). scoreboard_debug (a ScoreboardDebugInfo)
    is drawn on the debug output when debug is enabled.
    """
    out_path = Path(out_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    crop_w = min(frame_h * 9 // 16, frame_w)

    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    crop_writer = cv2.VideoWriter(str(out_path), fourcc, fps, (crop_w, frame_h))
    debug_writer = None
    if debug:
        debug_path = Path(debug_path)
        debug_writer = cv2.VideoWriter(str(debug_path), fourcc, fps, (frame_w, frame_h))

    panner = SpringPanner(k=spring_k)
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        focus = interpolate_focus(focus_map_px, frame_idx)
        panner.update(focus)
        center_x = panner.center_x(frame_w)
        cropped, crop_x, cw = portrait_crop(frame, center_x)
        if game_state is not None:
            cropped = compose_scoreboard(cropped, game_state, overlay_style)
        crop_writer.write(cropped)
        if debug_writer is not None:
            dbg = frame.copy()
            cv2.rectangle(dbg, (crop_x, 0), (crop_x + cw, frame_h), (0, 165, 255), 3)
            if focus is not None:
                fx = int(focus)
                cv2.line(dbg, (fx, 0), (fx, frame_h), (255, 0, 255), 2)
            if scoreboard_debug is not None and scoreboard_debug.valid_readings > 0:
                _draw_scoreboard_debug(dbg, scoreboard_debug)
            debug_writer.write(dbg)
        frame_idx += 1

    cap.release()
    crop_writer.release()
    if debug_writer is not None:
        debug_writer.release()

    _reencode_h264(out_path)
    if debug:
        _reencode_h264(Path(debug_path))
    logger.info("Rendered %d frames → %s", frame_idx, out_path)
    return out_path
