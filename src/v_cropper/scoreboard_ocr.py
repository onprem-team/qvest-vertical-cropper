# SPDX-FileCopyrightText: Copyright (c) 2026 Qvest Group GmbH
# SPDX-License-Identifier: Apache-2.0
"""VLM-based scoreboard OCR extraction via the provider-neutral connector.

Samples N frames from the original (pre-crop) video and asks the configured
vision model to extract structured scoreboard information from each frame.
Results are collected as ScoreboardReading objects for downstream consensus
merging.
"""
from __future__ import annotations

import concurrent.futures
import json
import logging
import time
from collections.abc import Callable

import cv2
import numpy as np

from .backend import image_part, make_backend, text_part
from .scoreboard_state import ScoreboardReading, TeamInfo

logger = logging.getLogger("v_cropper.scoreboard_ocr")

_SCOREBOARD_PROMPT = (
    "You are analyzing a sports broadcast frame. Extract ALL visible scoreboard / "
    "overlay information into a structured JSON object.\n\n"
    "Return ONLY a JSON object with this schema:\n"
    "{\n"
    '  "sport": "<detected sport>",\n'
    '  "event": "<specific event name if visible, e.g. High Jump, 100m, Pole Vault, or null>",\n'
    '  "teams": [\n'
    '    { "name": "<team name or abbreviation>", "score": <int or null> }\n'
    "  ],\n"
    '  "period": "<quarter, half, inning, period, set — whatever applies, or null>",\n'
    '  "clock": "<game clock string, or null>",\n'
    '  "athlete": "<athlete/competitor name if individual sport, or null>",\n'
    '  "extra_fields": [\n'
    '    { "label": "<field name>", "value": "<field value>" }\n'
    "  ]\n"
    "}\n\n"
    "Rules:\n"
    '- "teams" should have exactly 2 entries (home first, away second) if two teams '
    "are visible. If the sport is individual (track, golf, swimming, high jump, "
    "pole vault), return an empty teams array and use the \"athlete\" field instead.\n"
    '- "athlete" is for individual sports only. Do NOT also put the athlete name '
    "in extra_fields — that would be a duplicate.\n"
    '- "extra_fields" captures anything sport-specific: fouls, timeouts, possession '
    "indicator, balls/strikes/outs, down & distance, shot clock, event, mark/time, "
    "lane, position, lap, etc.\n"
    "- For field events (high jump, pole vault, long jump, etc.), look for attempt "
    "results at each height/distance. Use checkmarks and X marks as shown on the "
    "broadcast. Format each height as a single extra_field, e.g.:\n"
    '  {"label": "2.29", "value": "O"} for a successful clearance\n'
    '  {"label": "2.31", "value": "XO"} for a miss then clearance\n'
    '  {"label": "2.33", "value": "X"} for a miss (attempt in progress)\n'
    "  Use O for clear/pass/checkmark, X for fail/miss/X-mark, - for not yet attempted.\n"
    "- If a field is not visible or unreadable, use null.\n"
    "- Do NOT guess — only extract what is clearly visible.\n"
)


def _select_sample_frames(total_frames: int, sample_count: int) -> list[int]:
    """Evenly space sample_count frame indices across the clip."""
    if total_frames <= sample_count:
        return list(range(total_frames))
    step = total_frames / sample_count
    return [int(i * step) for i in range(sample_count)]


def _count_frames(video_path: str) -> int:
    """Count actually-decodable frames via a sequential pass.

    CAP_PROP_FRAME_COUNT reports the container's frame count, which can exceed
    the number OpenCV/ffmpeg actually decodes for some files, so we count for
    real to keep sampling indices valid.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0
    count = 0
    while cap.grab():
        count += 1
    cap.release()
    return count


def sample_frames(video_path: str, sample_count: int = 10) -> list[np.ndarray]:
    """Read sample_count evenly spaced full-resolution BGR frames (pre-crop).

    Two sequential passes: one to count the truly-decodable frames, one to
    collect the evenly-spaced target indices. Seeking (CAP_PROP_POS_FRAMES) is
    avoided because it is unreliable across codecs. Memory stays low since only
    the sampled frames are retained, not the whole clip.
    """
    total = _count_frames(video_path)
    if total <= 0:
        logger.warning("No decodable frames for scoreboard sampling: %s", video_path)
        return []

    wanted = set(_select_sample_frames(total, sample_count))
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    frames: list[np.ndarray] = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx in wanted:
            frames.append(frame)
            if len(frames) == len(wanted):
                break
        idx += 1
    cap.release()
    return frames


def _encode_frame_jpeg(frame: np.ndarray) -> bytes:
    """Encode a BGR frame to JPEG bytes."""
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes()


def _parse_reading(frame_idx: int, text: str) -> ScoreboardReading:
    """Parse a VLM JSON response into a ScoreboardReading."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        logger.warning("  frame %d — failed to parse JSON response", frame_idx)
        return ScoreboardReading(frame_idx=frame_idx, confidence=0.0)

    teams: list[TeamInfo] = []
    for t in obj.get("teams", []):
        name = t.get("name", "") or ""
        score = t.get("score")
        if isinstance(score, str):
            try:
                score = int(score)
            except ValueError:
                score = None
        teams.append(TeamInfo(name=name, score=score))

    # Extract athlete name (individual sports) and use it to dedup extra_fields
    athlete = obj.get("athlete")
    if athlete and str(athlete).lower() == "null":
        athlete = None
    athlete_lower = (athlete or "").strip().lower()

    extra_fields = []
    dedup_labels = {"athlete", "name", "competitor"}
    for ef in obj.get("extra_fields", []):
        label = ef.get("label", "")
        value = ef.get("value", "")
        if not label or not value or str(value).lower() == "null":
            continue
        # Skip fields that duplicate the top-level athlete name
        if athlete_lower and label.lower() in dedup_labels and value.strip().lower() == athlete_lower:
            continue
        extra_fields.append({"label": str(label), "value": str(value)})

    # Prepend athlete as first extra_field if present (single source of truth)
    if athlete:
        extra_fields.insert(0, {"label": "Athlete", "value": str(athlete).strip()})

    event_raw = obj.get("event")
    event = str(event_raw).strip() if event_raw and str(event_raw).lower() != "null" else None

    return ScoreboardReading(
        frame_idx=frame_idx,
        sport=obj.get("sport"),
        event=event,
        teams=teams,
        period=obj.get("period") if obj.get("period") != "null" else None,
        clock=obj.get("clock") if obj.get("clock") != "null" else None,
        extra_fields=extra_fields,
        confidence=1.0,
    )


def extract_scoreboard(
    frames: list[np.ndarray],
    api_key: str | None = None,
    model: str | None = None,
    on_progress: Callable[[int, int], None] | None = None,
    *,
    backend=None,
) -> list[ScoreboardReading]:
    """Extract scoreboard data from pre-sampled video frames via the VLM connector.

    Args:
        frames: Already-sampled BGR frames (pre-crop, full resolution). One VLM
            call is made per frame.
        api_key: Provider API key (else resolved from env by the connector).
        model: Model ID (else the provider default from the connector).
        on_progress: Optional callback(done, total) for progress reporting.
        backend: Optional VisionBackend to inject (used by tests).

    Returns:
        List of ScoreboardReading objects (one per frame).
    """
    if not frames:
        return []

    backend = backend or make_backend(api_key=api_key, model=model)
    logger.info(
        "Scoreboard OCR: extracting from %d sampled frames (model=%s)",
        len(frames), backend.model,
    )

    def _call_vlm(frame_idx: int) -> ScoreboardReading:
        jpeg_bytes = _encode_frame_jpeg(frames[frame_idx])
        messages = [{
            "role": "user",
            "content": [image_part(jpeg_bytes), text_part(_SCOREBOARD_PROMPT)],
        }]
        # Retry/backoff (incl. rate limits) is handled inside backend.complete.
        try:
            text = backend.complete(messages, temperature=0.1)
        except Exception as e:
            logger.warning("  frame %d — scoreboard VLM call failed: %s", frame_idx, e)
            return ScoreboardReading(frame_idx=frame_idx, confidence=0.0)

        reading = _parse_reading(frame_idx, text)
        logger.info(
            "  frame %d — sport=%s teams=%d extras=%d",
            frame_idx, reading.sport, len(reading.teams), len(reading.extra_fields),
        )
        return reading

    readings: list[ScoreboardReading] = []
    t0 = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(frames), 2)) as pool:
        futures = {pool.submit(_call_vlm, idx): idx for idx in range(len(frames))}
        done_count = 0
        for fut in concurrent.futures.as_completed(futures):
            reading = fut.result()
            readings.append(reading)
            done_count += 1
            if on_progress:
                on_progress(done_count, len(frames))

    elapsed = time.time() - t0
    valid_count = sum(1 for r in readings if r.confidence > 0)
    totals = backend.usage_totals
    logger.info(
        "Scoreboard OCR complete: %d/%d valid readings in %.1fs (%d API calls, %d tokens)",
        valid_count, len(readings), elapsed, totals["api_calls"], totals["total_tokens"],
    )
    return readings
