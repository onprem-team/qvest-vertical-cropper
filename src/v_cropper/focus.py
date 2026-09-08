# SPDX-FileCopyrightText: Copyright (c) 2026 Qvest Group GmbH
# SPDX-License-Identifier: Apache-2.0
"""Point-based VLM focus extraction (provider-neutral connector).

The vision model is shown one clean, downscaled keyframe at a time and asked to point at
the current focus of play, returning ``{"x": 0-1000, "y": 0-1000}`` normalized image
coordinates. This is the R&D v0013-winning approach: one small image per call (no
motion-context frames), so it is cheap and works even on single-image models. A
median-of-3 pass rejects single stray points before interpolation.
"""

import concurrent.futures
import json
import logging
import re
import time

import cv2

from .backend import image_part, make_backend, text_part
from .prompts import resolve_prompt

logger = logging.getLogger(__name__)

# Enough for {"x":..,"y":..}; Gemini thinking is disabled by the backend so this is not
# consumed by hidden reasoning (verified live: ~16 completion tokens for a point reply).
POINT_MAX_TOKENS = 200
DEFAULT_SEND_WIDTH = 768

# First balanced {...} object in the reply (tolerates markdown fences / extra prose).
_POINT_RE = re.compile(r"\{[^{}]*\}")
# Regex fallback for models that emit almost-JSON (e.g. qwen3-vl drops a quote:
# ``{"x": 624, y": 437}``). Grabs the first number after an x/y key, tolerating a
# missing quote and a leading ``[`` from a list-packed coordinate.
_NUM = r"(-?\d+(?:\.\d+)?)"
_X_RE = re.compile(r'x"?\s*:\s*\[?\s*' + _NUM)
_Y_RE = re.compile(r'y"?\s*:\s*\[?\s*' + _NUM)


def _coord(value):
    """Coerce a coordinate to a float.

    Some vision models pack the point into a list (e.g. qwen3-vl returns
    ``{"x": [x, y], ...}``); take the first element so we still recover the
    coordinate. Scalars (the well-formed ``{"x": N}`` case) pass through unchanged.
    """
    if isinstance(value, (list, tuple)):
        value = value[0] if value else None
    return float(value)


def _extract_xy(block):
    """Return ``(x, y)`` floats from a ``{...}`` block, or ``None``.

    Prefers strict JSON (with list-packed coordinate tolerance); on malformed JSON
    falls back to a regex that recovers the first x/y numbers.
    """
    try:
        d = json.loads(block)
        return _coord(d["x"]), _coord(d["y"])
    except (ValueError, TypeError, KeyError):
        xm, ym = _X_RE.search(block), _Y_RE.search(block)
        if xm and ym:
            return float(xm.group(1)), float(ym.group(1))
        return None


def parse_point_response(text):
    """Extract ``(x_norm, y_norm)`` in 0-1000 from a model reply, or ``None``.

    Tolerates ```-fences and surrounding prose (matches the first ``{...}``), list-packed
    coordinates, and near-JSON typos, and rejects out-of-range coordinates, mirroring the
    R&D ``vlm.parse_point``.
    """
    if not text:
        return None
    m = _POINT_RE.search(text)
    if not m:
        return None
    xy = _extract_xy(m.group(0))
    if xy is None:
        return None
    x, y = xy
    if not (0 <= x <= 1000 and 0 <= y <= 1000):
        return None
    return x, y


def median3(focus_map):
    """Median-of-3 outlier rejection over sorted keyframe x-values (endpoints untouched).

    Kills a single stray VLM point (e.g. one frame that snapped to a bystander) before
    interpolation. Returns a new dict; inputs with <3 samples are returned unchanged.
    """
    pts = sorted(focus_map.items())
    if len(pts) < 3:
        return dict(focus_map)
    out = dict(focus_map)
    for j in range(1, len(pts) - 1):
        idx = pts[j][0]
        out[idx] = sorted([pts[j - 1][1], pts[j][1], pts[j + 1][1]])[1]
    return out


def interpolate_focus(focus_map, frame_idx):
    """Linearly interpolate sparse keyframe focus (pixels) to any frame index."""
    if not focus_map:
        return None
    keys = sorted(focus_map.keys())
    if frame_idx <= keys[0]:
        return focus_map[keys[0]]
    if frame_idx >= keys[-1]:
        return focus_map[keys[-1]]
    for i in range(len(keys) - 1):
        if keys[i] <= frame_idx <= keys[i + 1]:
            t = (frame_idx - keys[i]) / (keys[i + 1] - keys[i])
            return focus_map[keys[i]] * (1 - t) + focus_map[keys[i + 1]] * t
    return None  # pragma: no cover - defensive: the bracketing loop always returns for interior frames


def _downscaled_jpeg(frame, send_width):
    """Downscale a BGR frame to ``send_width`` (aspect-preserving; never upscales)."""
    fh, fw = frame.shape[:2]
    if send_width and fw > send_width:
        new_h = max(1, round(fh * send_width / fw))
        frame = cv2.resize(frame, (send_width, new_h), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return buf.tobytes()


def extract_focus_points(
    video_path,
    api_key=None,
    *,
    model=None,
    backend=None,
    prompt=None,
    sport=None,
    send_width=DEFAULT_SEND_WIDTH,
    sample_every=30,
    concurrency=8,
    max_tokens=POINT_MAX_TOKENS,
):
    """Sample keyframes and ask the VLM to point at the action; return an x-focus map.

    The vision model is reached through a provider-neutral ``VisionBackend`` (default:
    the OpenAI-compatible connector built from env vars / args). Pass an explicit
    ``backend`` to inject a fake in tests. The pointing ``prompt`` is resolved via
    :func:`v_cropper.prompts.resolve_prompt` (inline ``prompt`` > ``sport`` preset >
    football default) unless supplied directly.

    Returns (focus_map_px, reasoning_map, n_failed):
      focus_map_px : {keyframe_idx: focus_x_in_pixels} after median-of-3 filtering
      reasoning_map: always {} (the point prompt returns no reasoning; kept for API compat)
      n_failed     : keyframes whose VLM call failed / produced no valid point
    """
    backend = backend or make_backend(api_key=api_key, model=model)
    resolved_prompt = prompt if prompt is not None else resolve_prompt(sport=sport)

    cap = cv2.VideoCapture(video_path)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    keyframes = []  # list[(idx, jpeg_bytes)]
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % sample_every == 0:
            keyframes.append((idx, _downscaled_jpeg(frame, send_width)))
        idx += 1
    cap.release()

    logger.info("Sampled %d keyframes (every %d), send_width=%d, concurrency=%d",
                len(keyframes), sample_every, send_width, concurrency)

    if not keyframes:
        logger.warning("No keyframes sampled — video unreadable or empty: %s", video_path)
        return {}, {}, 0

    def _call(current_idx, jpeg_bytes):
        content = [image_part(jpeg_bytes), text_part(resolved_prompt)]
        messages = [{"role": "user", "content": content}]
        try:
            text = backend.complete(messages, temperature=0.0, max_tokens=max_tokens)
            point = parse_point_response(text)
            if point is None:
                logger.warning("  frame %d — no/invalid point in reply", current_idx)
                return current_idx, None
            x_norm, _y_norm = point
            fx_px = x_norm / 1000.0 * frame_w
            logger.info("  frame %d → x=%.0f (%.0fpx)", current_idx, x_norm, fx_px)
            return current_idx, fx_px
        except Exception as e:
            logger.warning("  frame %d — VLM call failed: %s", current_idx, e)
            return current_idx, None

    focus_map = {}
    n_failed = 0
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(keyframes), concurrency)) as pool:
        futures = [pool.submit(_call, fi, jb) for fi, jb in keyframes]
        for fut in concurrent.futures.as_completed(futures):
            fidx, fx_px = fut.result()
            if fx_px is not None:
                focus_map[fidx] = fx_px
            else:
                n_failed += 1

    focus_map = median3(focus_map)
    logger.info("Focus pass: %d/%d keyframes ok, %d failed, %.1fs",
                len(focus_map), len(keyframes), n_failed, time.time() - t0)
    return focus_map, {}, n_failed
