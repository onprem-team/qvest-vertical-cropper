# SPDX-FileCopyrightText: Copyright (c) 2026 Qvest.US, LLC
# SPDX-License-Identifier: Apache-2.0
"""Dynamic scoreboard overlay renderer.

Composites a scoreboard overlay onto cropped 9:16 frames using Pillow for
high-quality typography. Supports two rendering modes:
  - Matchup: head-to-head team display (soccer, hockey, football, etc.)
  - Info Board: key-value grid for individual sports (track, golf, swimming, etc.)

Uses the Glass Minimal style from the design mockups.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
    _HAS_PILLOW = True
except ImportError:  # pragma: no cover - Pillow is a hard dependency; cv2 fallback kept for safety
    _HAS_PILLOW = False

from .scoreboard_state import GameState, is_matchup

logger = logging.getLogger("v_cropper.scoreboard_render")

_FONT_CACHE: dict[tuple[str, int], ImageFont.FreeTypeFont | ImageFont.ImageFont] = {}


@dataclass(frozen=True)
class OverlayStyle:
    bg_opacity: float = 0.85
    height_ratio: float = 0.16
    position: str = "bottom"
    font_family: str = ""
    primary_color: tuple[int, int, int] = (255, 255, 255)
    accent_color: tuple[int, int, int] = (99, 102, 241)
    dim_color: tuple[int, int, int] = (160, 160, 200)
    bg_color: tuple[int, int, int] = (10, 10, 20)


def _get_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load a TrueType font with caching, falling back to default.

    Searches macOS system fonts first (for local dev) then common Linux paths
    (matching the vertical-crop service container), so output is consistent
    across both environments without bundling binary font assets.
    """
    key = ("bold" if bold else "regular", size)
    if key in _FONT_CACHE:
        return _FONT_CACHE[key]

    font_paths = [
        # macOS
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold
        else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial Bold.ttf" if bold
        else "/Library/Fonts/Arial.ttf",
        # Linux (DejaVu / Liberation)
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf" if bold
        else "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    for path in font_paths:
        try:
            font = ImageFont.truetype(path, size)
            _FONT_CACHE[key] = font
            return font
        except OSError:
            continue

    font = ImageFont.load_default()
    _FONT_CACHE[key] = font
    return font


def _render_matchup_pillow(
    draw: ImageDraw.ImageDraw,
    state: GameState,
    band_w: int,
    band_h: int,
) -> None:
    """Render matchup-mode overlay (two teams, scores, period/clock, extras)."""
    score_size = max(14, band_h // 3)
    team_size = max(10, band_h // 5)
    info_size = max(8, band_h // 6)
    extra_size = max(7, band_h // 7)

    font_score = _get_font(score_size, bold=True)
    font_team = _get_font(team_size, bold=True)
    font_info = _get_font(info_size, bold=False)
    font_extra = _get_font(extra_size, bold=False)

    mid_x = band_w // 2
    pad_x = int(band_w * 0.04)

    # Primary row vertical center
    primary_y = int(band_h * 0.20)

    # Home team (left)
    home_score_text = str(state.home_score) if state.home_score is not None else "-"
    draw.text((pad_x, primary_y), state.home_team, font=font_team, fill=(255, 255, 255))
    home_team_bbox = draw.textbbox((pad_x, primary_y), state.home_team, font=font_team)
    score_x = home_team_bbox[2] + int(band_w * 0.02)
    draw.text((score_x, primary_y - 2), home_score_text, font=font_score, fill=(255, 255, 255))

    # Away team (right)
    away_score_text = str(state.away_score) if state.away_score is not None else "-"
    away_team_bbox = draw.textbbox((0, 0), state.away_team, font=font_team)
    away_team_w = away_team_bbox[2] - away_team_bbox[0]
    away_score_bbox = draw.textbbox((0, 0), away_score_text, font=font_score)
    away_score_w = away_score_bbox[2] - away_score_bbox[0]

    away_team_x = band_w - pad_x - away_team_w
    draw.text((away_team_x, primary_y), state.away_team, font=font_team, fill=(255, 255, 255))
    away_score_x = away_team_x - int(band_w * 0.02) - away_score_w
    draw.text((away_score_x, primary_y - 2), away_score_text, font=font_score, fill=(255, 255, 255))

    # Center: period + clock
    center_parts = []
    if state.period:
        center_parts.append(state.period)
    if state.clock:
        center_parts.append(state.clock)
    if center_parts:
        center_text = "  ".join(center_parts)
        center_bbox = draw.textbbox((0, 0), center_text, font=font_info)
        center_w = center_bbox[2] - center_bbox[0]
        draw.text(
            (mid_x - center_w // 2, primary_y + 2),
            center_text, font=font_info, fill=(200, 200, 230),
        )

    # Extra fields row (below primary)
    if state.extra_fields:
        extra_y = int(band_h * 0.62)
        separator = "   |   "
        extra_text = separator.join(
            f"{ef['label']}: {ef['value']}" for ef in state.extra_fields
        )
        extra_bbox = draw.textbbox((0, 0), extra_text, font=font_extra)
        extra_w = extra_bbox[2] - extra_bbox[0]

        if extra_w > band_w - 2 * pad_x:
            # Wrap into two lines
            items = [f"{ef['label']}: {ef['value']}" for ef in state.extra_fields]
            half = len(items) // 2
            line1 = "   |   ".join(items[:half])
            line2 = "   |   ".join(items[half:])
            l1_bbox = draw.textbbox((0, 0), line1, font=font_extra)
            l2_bbox = draw.textbbox((0, 0), line2, font=font_extra)
            draw.text(
                (mid_x - (l1_bbox[2] - l1_bbox[0]) // 2, extra_y),
                line1, font=font_extra, fill=(160, 160, 200),
            )
            draw.text(
                (mid_x - (l2_bbox[2] - l2_bbox[0]) // 2, extra_y + extra_size + 4),
                line2, font=font_extra, fill=(160, 160, 200),
            )
        else:
            draw.text(
                (mid_x - extra_w // 2, extra_y),
                extra_text, font=font_extra, fill=(160, 160, 200),
            )


def _is_attempt_value(value: str) -> bool:
    """Check if a value looks like field event attempt results (e.g. 'O', 'XO', 'XXX')."""
    return bool(value) and all(c in "OXox-" for c in value.strip())


def _draw_attempt_marks(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    value: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
) -> int:
    """Draw attempt O/X marks with color coding. Returns the x after drawing."""
    green = (34, 197, 94)
    red = (239, 68, 68)
    dim = (120, 120, 160)
    cursor_x = x
    for ch in value.strip():
        if ch.upper() == "O":
            char, color = "\u2713", green   # checkmark
        elif ch.upper() == "X":
            char, color = "\u2717", red     # X mark
        else:
            char, color = "-", dim
        draw.text((cursor_x, y), char, font=font, fill=color)
        bbox = draw.textbbox((cursor_x, y), char, font=font)
        cursor_x = bbox[2] + 3
    return cursor_x


def _render_info_board_pillow(
    draw: ImageDraw.ImageDraw,
    state: GameState,
    band_w: int,
    band_h: int,
) -> None:
    """Render info-board mode (key-value pairs for individual sports).

    Separates the athlete name as a header row, then lays out remaining
    fields. Attempt results (O/X marks) are color-coded green/red.
    """
    event_size = max(7, band_h // 7)
    header_size = max(10, band_h // 4)
    label_size = max(8, band_h // 6)
    value_size = max(9, band_h // 5)

    font_event = _get_font(event_size, bold=True)
    font_header = _get_font(header_size, bold=True)
    font_label = _get_font(label_size, bold=False)
    font_value = _get_font(value_size, bold=True)

    pad_x = int(band_w * 0.04)
    fields = list(state.extra_fields)
    if not fields and not state.event:
        return

    # Extract athlete name to display as header, separate from other fields
    athlete_name: str | None = None
    detail_fields: list[dict[str, str]] = []
    for ef in fields:
        if ef["label"].lower() in ("athlete", "name", "competitor") and not athlete_name:
            athlete_name = ef["value"]
        else:
            detail_fields.append(ef)

    y_cursor = int(band_h * 0.05)

    # Row 0: Event type label (e.g. "HIGH JUMP")
    event_label = state.event or state.sport
    if event_label:
        event_display = event_label.upper()
        draw.text((pad_x, y_cursor), event_display, font=font_event, fill=(160, 160, 200))
        y_cursor += event_size + 3

    # Row 1: Athlete name (left) + primary metric (right)
    if athlete_name:
        draw.text((pad_x, y_cursor), athlete_name, font=font_header, fill=(255, 255, 255))
        # Pull out the primary metric (mark/height/time) to show next to the name
        metric_field = None
        for ef in detail_fields:
            if ef["label"].lower() in ("mark", "height", "time", "result", "distance", "score",
                                        "best", "current height", "bar height"):
                metric_field = ef
                break
        if metric_field:
            metric_text = f"{metric_field['value']}"
            m_bbox = draw.textbbox((0, 0), metric_text, font=font_header)
            m_w = m_bbox[2] - m_bbox[0]
            draw.text(
                (band_w - pad_x - m_w, y_cursor),
                metric_text, font=font_header, fill=(99, 200, 130),
            )
            detail_fields = [ef for ef in detail_fields if ef is not metric_field]

        y_cursor += header_size + 4

    # Row 2+: remaining fields laid out horizontally with pre-measured wrapping
    if detail_fields:
        x_cursor = pad_x
        gap = int(band_w * 0.04)
        max_x = band_w - pad_x

        for ef in detail_fields:
            label_text = ef["label"]
            value_text = ef["value"]

            # Pre-measure total width of this field to decide if we need to wrap first
            lbl_w = draw.textbbox((0, 0), label_text, font=font_label)[2]
            if _is_attempt_value(value_text):
                # Each mark char is roughly one character width
                mark_w = sum(
                    draw.textbbox((0, 0), ch, font=font_value)[2] + 3
                    for ch in value_text.strip()
                )
                val_w = mark_w
            else:
                val_w = draw.textbbox((0, 0), value_text, font=font_value)[2]
            field_w = lbl_w + 5 + val_w

            # Wrap BEFORE drawing if this field won't fit on the current line
            if x_cursor + field_w > max_x and x_cursor > pad_x:
                x_cursor = pad_x
                y_cursor += value_size + 4

            draw.text((x_cursor, y_cursor), label_text, font=font_label, fill=(160, 160, 200))
            lbl_bbox = draw.textbbox((x_cursor, y_cursor), label_text, font=font_label)
            val_x = lbl_bbox[2] + 5

            if _is_attempt_value(value_text):
                end_x = _draw_attempt_marks(draw, val_x, y_cursor, value_text, font_value)
            else:
                is_key = label_text.lower() in ("mark", "score", "result", "time", "position")
                val_color = (99, 200, 130) if is_key else (255, 255, 255)
                draw.text((val_x, y_cursor), value_text, font=font_value, fill=val_color)
                val_bbox = draw.textbbox((val_x, y_cursor), value_text, font=font_value)
                end_x = val_bbox[2]

            x_cursor = end_x + gap


def _render_fallback_cv2(
    frame: np.ndarray,
    state: GameState,
    band_y: int,
    band_h: int,
) -> None:
    """Minimal OpenCV fallback when Pillow is unavailable."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    h, w = frame.shape[:2]
    scale = band_h / 80.0
    thickness = max(1, int(scale))

    if is_matchup(state):
        home_text = f"{state.home_team} {state.home_score or '-'}"
        away_text = f"{state.away_score or '-'} {state.away_team}"
        center_text = f"{state.period or ''} {state.clock or ''}".strip()

        cv2.putText(frame, home_text, (10, band_y + int(band_h * 0.55)),
                    font, scale * 0.6, (255, 255, 255), thickness, cv2.LINE_AA)
        cv2.putText(frame, away_text, (w - 150, band_y + int(band_h * 0.55)),
                    font, scale * 0.6, (255, 255, 255), thickness, cv2.LINE_AA)
        if center_text:
            (tw, _), _ = cv2.getTextSize(center_text, font, scale * 0.5, thickness)
            cv2.putText(frame, center_text, (w // 2 - tw // 2, band_y + int(band_h * 0.55)),
                        font, scale * 0.5, (200, 200, 230), thickness, cv2.LINE_AA)
    else:
        y_pos = band_y + int(band_h * 0.4)
        x_pos = 10
        for ef in state.extra_fields[:6]:
            text = f"{ef['label']}: {ef['value']}"
            cv2.putText(frame, text, (x_pos, y_pos),
                        font, scale * 0.4, (255, 255, 255), thickness, cv2.LINE_AA)
            x_pos += 120


def compose_scoreboard(
    frame: np.ndarray,
    game_state: GameState,
    style: OverlayStyle | None = None,
) -> np.ndarray:
    """Composite the scoreboard overlay onto a cropped frame.

    Args:
        frame: BGR cropped frame (9:16 portrait).
        game_state: Consensus game state to render.
        style: Overlay style configuration. Uses defaults if None.

    Returns:
        Modified frame with overlay composited at the configured position.
    """
    if style is None:
        style = OverlayStyle()

    h, w = frame.shape[:2]
    band_h = max(30, int(h * style.height_ratio))

    if style.position == "top":
        band_y = 0
    else:
        band_y = h - band_h

    # Draw semi-transparent background band
    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (0, band_y),
        (w, band_y + band_h),
        style.bg_color,
        -1,
    )
    cv2.addWeighted(overlay, style.bg_opacity, frame, 1.0 - style.bg_opacity, 0, frame)

    # Subtle top border
    border_y = band_y if style.position == "bottom" else band_y + band_h
    cv2.line(frame, (0, border_y), (w, border_y), (60, 60, 100), 1)

    if _HAS_PILLOW:
        # Render text with Pillow for better typography
        band_img = Image.fromarray(frame[band_y:band_y + band_h, :, ::-1])
        draw = ImageDraw.Draw(band_img)

        if is_matchup(game_state):
            _render_matchup_pillow(draw, game_state, w, band_h)
        else:
            _render_info_board_pillow(draw, game_state, w, band_h)

        frame[band_y:band_y + band_h] = np.array(band_img)[:, :, ::-1]
    else:
        _render_fallback_cv2(frame, game_state, band_y, band_h)

    return frame
