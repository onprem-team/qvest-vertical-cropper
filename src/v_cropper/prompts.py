# SPDX-FileCopyrightText: Copyright (c) 2026 Qvest Group GmbH
# SPDX-License-Identifier: Apache-2.0
"""Sport-configurable VLM pointing prompts.

The focus model is asked to point at the current focus of play and reply with
``{"x": <int 0-1000>, "y": <int 0-1000>}`` normalized image coordinates. Built-in
presets cover the sports we ship; users can override with an inline prompt or a prompt
file for any sport/content. Football is the default (the byte-identical R&D v0013 prompt
that won the benchmark).
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# Shared JSON-output instruction appended to every built-in preset. Keeping it in one
# place (DRY) guarantees all sports request the same machine-parseable point format.
_JSON_SUFFIX = (
    ' Answer with ONLY this JSON, nothing else: {"x": <int 0-1000>, "y": <int 0-1000>} '
    "\u2014 normalized image coordinates (x: 0=left edge, 1000=right edge; y: 0=top)."
)

# Football lead sentence is byte-identical to the v0013 winner (before the JSON suffix).
_FOOTBALL_LEAD = (
    "American football broadcast frame. Point at the football's CURRENT location: the "
    "player holding or carrying the ball (e.g. the quarterback in the pocket counts), or "
    "the ball itself if in flight or loose. Only if no ball is in play, point at the "
    "center of the player formation."
)

_BASKETBALL_LEAD = (
    "Basketball broadcast frame. Point at the current focus of play: the player with the "
    "ball, or the ball itself if it is in flight or loose. Only if no ball is in play, "
    "point at the center of the player formation."
)

_SOCCER_LEAD = (
    "Association football (soccer) broadcast frame. Point at the current focus of play: "
    "the player with the ball, or the ball itself if it is in flight or loose. Only if no "
    "ball is in play, point at the center of the player formation."
)

_HOCKEY_LEAD = (
    "Ice hockey broadcast frame. Point at the current focus of play: the player with the "
    "puck, or the puck itself if it is loose or in flight. Only if no puck is in play, "
    "point at the center of the player formation."
)

_GENERAL_LEAD = (
    "Sports broadcast frame. Point at the single most important subject or action a "
    "broadcast director would keep inside a vertical 9:16 crop \u2014 usually the athlete "
    "with the ball/puck, or the ball/puck itself; otherwise the center of the main action."
)

PRESETS: dict[str, str] = {
    "football": _FOOTBALL_LEAD + _JSON_SUFFIX,
    "basketball": _BASKETBALL_LEAD + _JSON_SUFFIX,
    "soccer": _SOCCER_LEAD + _JSON_SUFFIX,
    "hockey": _HOCKEY_LEAD + _JSON_SUFFIX,
    "general": _GENERAL_LEAD + _JSON_SUFFIX,
}

DEFAULT_SPORT = "football"


def available_sports() -> list[str]:
    """Sorted list of built-in sport preset names (for CLI help / validation)."""
    return sorted(PRESETS)


def _looks_like_point_prompt(text: str) -> bool:
    low = text.lower()
    return ('"x"' in low or "json" in low) and "y" in low


def resolve_prompt(*, inline: str | None = None, prompt_file: str | None = None,
                   sport: str | None = None) -> str:
    """Resolve the pointing prompt by precedence.

    inline (``--prompt`` / ``$VCROPPER_PROMPT``)
      > prompt file (``--prompt-file`` / ``$VCROPPER_PROMPT_FILE``)
      > sport preset (``--sport`` / ``$VCROPPER_SPORT``)
      > football default.

    Raises ValueError for an unknown sport or an empty prompt file, and the usual
    OSError (FileNotFoundError/PermissionError) if a given prompt file cannot be read.
    """
    inline = inline if inline is not None else os.environ.get("VCROPPER_PROMPT")
    if inline is not None and inline.strip():
        return _checked(inline)

    prompt_file = prompt_file if prompt_file is not None else os.environ.get("VCROPPER_PROMPT_FILE")
    if prompt_file:
        text = Path(prompt_file).read_text(encoding="utf-8")
        if not text.strip():
            raise ValueError(f"prompt file is empty: {prompt_file}")
        return _checked(text)

    sport = sport if sport is not None else os.environ.get("VCROPPER_SPORT")
    sport = (sport or DEFAULT_SPORT).strip().lower()
    if sport not in PRESETS:
        raise ValueError(
            f"unknown sport preset {sport!r}; choose one of {available_sports()} "
            f"or supply --prompt / --prompt-file")
    return PRESETS[sport]


def _checked(text: str) -> str:
    """Warn (do not fail) if a custom prompt omits the {x,y} JSON instruction."""
    if not _looks_like_point_prompt(text):
        logger.warning(
            "Custom prompt does not mention an {\"x\",\"y\"} JSON reply; the focus parser "
            "expects normalized 0-1000 point coordinates and may reject the model's output.")
    return text
