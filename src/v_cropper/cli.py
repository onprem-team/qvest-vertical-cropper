# SPDX-FileCopyrightText: Copyright (c) 2026 Qvest.US, LLC
# SPDX-License-Identifier: Apache-2.0
"""v-cropper CLI: point a vision model at a sports video, get a smooth 9:16 vertical crop."""

import argparse
import concurrent.futures
import json
import logging
import os
import sys
import time
from pathlib import Path

import cv2
from dotenv import load_dotenv

from .backend import make_backend
from .focus import DEFAULT_SEND_WIDTH, extract_focus_points
from .pipeline import resolve_stride
from .prompts import DEFAULT_SPORT, available_sports, resolve_prompt
from .render import render
from .smoothing import DEFAULT_SPRING_K

logger = logging.getLogger(__name__)

API_KEY_ENV_VARS = ("VCROPPER_API_KEY", "GEMINI_API_KEY")
# Rough flash-tier $/1M tokens for the informational cost estimate; override via env.
DEFAULT_PRICE_IN = 0.30
DEFAULT_PRICE_OUT = 2.50


def _has_api_key(cli_key):
    """True if a key is reachable via --api-key or a supported env var.

    We deliberately do NOT resolve the env key here — the backend resolves env + the
    GEMINI_API_KEY back-compat routing itself; passing an explicit key would suppress it.
    """
    return bool(cli_key or any(os.environ.get(v) for v in API_KEY_ENV_VARS))


def _provider_is_configured(cli_key):
    """Validate only configuration required before constructing the selected backend."""
    provider = os.environ.get("VCROPPER_PROVIDER", "openai").strip().lower()
    if provider == "bedrock":
        return bool(os.environ.get("BEDROCK_REGION") or os.environ.get("AWS_REGION"))
    return _has_api_key(cli_key)


def _env_truthy(name):
    return os.environ.get(name, "").strip().lower() in ("true", "1", "yes")


def _resolve(cli_value, env_name, default, cast=str):
    """--flag → $ENV → default (cast applied to the env string)."""
    if cli_value is not None:
        return cli_value
    raw = os.environ.get(env_name)
    if raw is not None and raw != "":
        try:
            return cast(raw)
        except (TypeError, ValueError):
            logger.warning("Invalid %s=%r; using default %r", env_name, raw, default)
    return default


def _probe_video(path):
    """Return (fps, n_frames) from container metadata (fast; header may be approximate)."""
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return fps, n_frames


def _sample_every(cli_sample_every, sample_fps, source_fps):
    """Resolve the keyframe stride. Delegates so the CLI, service, and eval agree."""
    return resolve_stride(cli_sample_every, sample_fps, source_fps)


def _estimate_cost(usage, price_in, price_out):
    return usage.get("prompt_tokens", 0) / 1e6 * price_in + usage.get("completion_tokens", 0) / 1e6 * price_out


def _run_scoreboard_ocr(video, backend, sample_count):
    """Pass 3: sample frames → VLM scoreboard OCR → consensus GameState (uses `backend`)."""
    from .scoreboard_ocr import extract_scoreboard, sample_frames
    from .scoreboard_state import merge_readings_with_debug

    frames = sample_frames(video, sample_count)
    readings = extract_scoreboard(frames, backend=backend)
    return merge_readings_with_debug(readings)


def _build_parser():
    parser = argparse.ArgumentParser(
        prog="v-cropper",
        description="Turn a landscape sports video into a smooth 9:16 vertical crop that "
        "follows the action, using OpenAI-compatible APIs or AWS Bedrock.",
    )
    parser.add_argument("video", help="Input landscape video file")
    parser.add_argument(
        "-o", "--output", default=None, help="Output crop path (default: <input>_vertical.mp4 beside input)"
    )
    parser.add_argument(
        "--debug", action="store_true", help="Also write <input>_debug.mp4 with the focus point + crop box"
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="VLM API key (else $VCROPPER_API_KEY / $GEMINI_API_KEY / .env). "
        "Endpoint + model come from $VCROPPER_BASE_URL / $VCROPPER_MODEL.",
    )
    parser.add_argument(
        "--base-url", default=None, help="OpenAI-compatible VLM endpoint (overrides $VCROPPER_BASE_URL)"
    )
    parser.add_argument("--model", default=None, help="Model ID (else $VCROPPER_MODEL, else the provider default)")
    parser.add_argument("--concurrency", type=int, default=8, help="Max concurrent VLM calls")

    sport = parser.add_argument_group("action tracking (sport / prompt)")
    sport.add_argument(
        "--sport",
        default=None,
        help=f"Sport preset for the pointing prompt (default: {DEFAULT_SPORT}). "
        f"Built-in: {', '.join(available_sports())}. Also $VCROPPER_SPORT.",
    )
    sport.add_argument(
        "--prompt", default=None, help="Inline pointing prompt (overrides --sport). Also $VCROPPER_PROMPT."
    )
    sport.add_argument(
        "--prompt-file", default=None, help="Read the pointing prompt from a file. Also $VCROPPER_PROMPT_FILE."
    )
    sport.add_argument("--sample-fps", type=float, default=2.0, help="Keyframes sampled per second (default 2.0)")
    sport.add_argument("--sample-every", type=int, default=None, help="Sample every Nth frame (overrides --sample-fps)")
    sport.add_argument(
        "--send-width",
        type=int,
        default=None,
        help=f"Downscale width sent to the VLM (default {DEFAULT_SEND_WIDTH}). Also $VCROPPER_SEND_WIDTH.",
    )
    sport.add_argument(
        "--spring-k",
        type=float,
        default=None,
        help=f"Spring smoothing stiffness, lower = smoother (default {DEFAULT_SPRING_K}). Also $VCROPPER_SPRING_K.",
    )

    metrics = parser.add_argument_group("metrics")
    metrics.add_argument(
        "--metrics-json", default=None, help="Write a run-metrics JSON (time, rtf, tokens, est. cost) to PATH"
    )

    sb = parser.add_argument_group("scoreboard overlay")
    sb.add_argument(
        "--scoreboard",
        action="store_true",
        help="Reconstruct the broadcast scoreboard and re-draw it on the crop (else $SCOREBOARD_ENABLED)",
    )
    sb.add_argument(
        "--scoreboard-sample-count",
        type=int,
        default=None,
        help="Frames sampled for scoreboard OCR (else $SCOREBOARD_SAMPLE_COUNT, default 10)",
    )
    sb.add_argument(
        "--scoreboard-model",
        default=None,
        help="Model for scoreboard OCR (else $SCOREBOARD_MODEL, else the main model)",
    )
    sb.add_argument(
        "--scoreboard-position",
        default=None,
        choices=["bottom", "top"],
        help="Overlay band position (else $SCOREBOARD_POSITION, default bottom)",
    )
    sb.add_argument(
        "--scoreboard-height-ratio",
        type=float,
        default=None,
        help="Overlay band height as a fraction of frame height (else $SCOREBOARD_HEIGHT_RATIO, default 0.16)",
    )
    sb.add_argument(
        "--scoreboard-opacity",
        type=float,
        default=None,
        help="Overlay band background opacity (else $SCOREBOARD_OPACITY, default 0.85)",
    )
    return parser


def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    load_dotenv()  # make .env values (API key + SCOREBOARD_*) visible to os.environ

    video = Path(args.video)
    if not video.exists():
        parser.error(f"Video not found: {video}")
    if not _provider_is_configured(args.api_key):
        if os.environ.get("VCROPPER_PROVIDER", "openai").strip().lower() == "bedrock":
            parser.error("AWS Bedrock requires $BEDROCK_REGION or $AWS_REGION.")
        parser.error(
            "No VLM API key. Pass --api-key, set $VCROPPER_API_KEY "
            "(with optional $VCROPPER_BASE_URL / $VCROPPER_MODEL) or $GEMINI_API_KEY, "
            "or add one to a .env file."
        )

    # Resolve the pointing prompt (inline > file/env > sport preset > football default).
    try:
        prompt = resolve_prompt(inline=args.prompt, prompt_file=args.prompt_file, sport=args.sport)
    except (ValueError, OSError) as e:
        parser.error(str(e))
    sport_label = (
        "custom"
        if (
            args.prompt
            or args.prompt_file
            or os.environ.get("VCROPPER_PROMPT")
            or os.environ.get("VCROPPER_PROMPT_FILE")
        )
        else (args.sport or os.environ.get("VCROPPER_SPORT") or DEFAULT_SPORT)
    )

    send_width = _resolve(args.send_width, "VCROPPER_SEND_WIDTH", DEFAULT_SEND_WIDTH, int)
    spring_k = _resolve(args.spring_k, "VCROPPER_SPRING_K", DEFAULT_SPRING_K, float)

    source_fps, source_frames = _probe_video(video)
    sample_every = _sample_every(args.sample_every, args.sample_fps, source_fps)

    out_path = Path(args.output) if args.output else video.with_name(f"{video.stem}_vertical.mp4")
    debug_path = video.with_name(f"{video.stem}_debug.mp4") if args.debug else None

    scoreboard_enabled = args.scoreboard or _env_truthy("SCOREBOARD_ENABLED")
    sb_sample_count = _resolve(args.scoreboard_sample_count, "SCOREBOARD_SAMPLE_COUNT", 10, int)
    sb_model = _resolve(args.scoreboard_model, "SCOREBOARD_MODEL", args.model)

    # Build the focus backend here (not inside focus) so we can report its token usage.
    focus_backend = make_backend(
        api_key=args.api_key, model=args.model, base_url=args.base_url,
    )

    t0 = time.time()
    game_state = None
    overlay_style = None
    sb_debug = None
    sb_backend = None

    if scoreboard_enabled:
        sb_backend = make_backend(
            api_key=args.api_key, model=sb_model, base_url=args.base_url,
        )
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            focus_fut = pool.submit(
                extract_focus_points,
                str(video),
                backend=focus_backend,
                prompt=prompt,
                sample_every=sample_every,
                concurrency=args.concurrency,
                send_width=send_width,
            )
            sb_fut = pool.submit(_run_scoreboard_ocr, str(video), sb_backend, sb_sample_count)
            focus_map, _reasoning, n_failed = focus_fut.result()
            game_state, sb_debug = sb_fut.result()

        from .scoreboard_render import OverlayStyle

        overlay_style = OverlayStyle(
            bg_opacity=_resolve(args.scoreboard_opacity, "SCOREBOARD_OPACITY", 0.85, float),
            height_ratio=_resolve(args.scoreboard_height_ratio, "SCOREBOARD_HEIGHT_RATIO", 0.16, float),
            position=_resolve(args.scoreboard_position, "SCOREBOARD_POSITION", "bottom"),
        )
        if game_state is None:
            print("No readable scoreboard found — skipping overlay.")
    else:
        focus_map, _reasoning, n_failed = extract_focus_points(
            str(video),
            backend=focus_backend,
            prompt=prompt,
            sample_every=sample_every,
            concurrency=args.concurrency,
            send_width=send_width,
        )

    n_total = len(focus_map) + n_failed
    if n_total == 0:
        parser.error("No keyframes were produced — is the video readable?")

    render(
        str(video),
        focus_map,
        out_path,
        debug=args.debug,
        debug_path=debug_path,
        spring_k=spring_k,
        game_state=game_state,
        overlay_style=overlay_style,
        scoreboard_debug=sb_debug if args.debug else None,
    )

    elapsed = time.time() - t0
    duration = source_frames / source_fps if (source_fps and source_frames) else 0.0
    fail_frac = n_failed / n_total
    price_in = _resolve(None, "VCROPPER_PRICE_IN", DEFAULT_PRICE_IN, float)
    price_out = _resolve(None, "VCROPPER_PRICE_OUT", DEFAULT_PRICE_OUT, float)
    est_cost = _estimate_cost(focus_backend.usage_totals, price_in, price_out)
    if sb_backend is not None:
        est_cost += _estimate_cost(sb_backend.usage_totals, price_in, price_out)

    metrics = {
        "wall_time_sec": round(elapsed, 3),
        "video_fps": round(source_fps, 4),
        "video_frames": source_frames,
        "video_duration_sec": round(duration, 4),
        "rtf": round(elapsed / duration, 4) if duration > 0 else None,
        "sport": sport_label,
        "sample_every": sample_every,
        "send_width": send_width,
        "spring_k": spring_k,
        "keyframes_ok": len(focus_map),
        "keyframes_failed": n_failed,
        "keyframe_fail_fraction": round(fail_frac, 4),
        "focus_model": focus_backend.model,
        "focus_usage": dict(focus_backend.usage_totals),
        "scoreboard_usage": dict(sb_backend.usage_totals) if sb_backend else None,
        "est_cost_usd": round(est_cost, 6),
    }
    if args.metrics_json:
        Path(args.metrics_json).write_text(json.dumps(metrics, indent=2))

    print(f"\nDone in {elapsed:.1f}s → {out_path}")
    if args.debug:
        print(f"Debug overlay → {debug_path}")
    if scoreboard_enabled and game_state is not None:
        print(
            f"Scoreboard: {game_state.sport} | "
            f"{game_state.home_team} {game_state.home_score} - "
            f"{game_state.away_score} {game_state.away_team}".rstrip()
        )
    rtf_str = f"{metrics['rtf']}" if metrics["rtf"] is not None else "n/a"
    print(
        f"Sport: {sport_label} | keyframes: {len(focus_map)} ok, {n_failed} failed "
        f"({fail_frac:.0%}) | rtf: {rtf_str} | tokens: {focus_backend.usage_totals['total_tokens']} "
        f"| est. cost: ${est_cost:.4f}"
    )
    if fail_frac > 0.20:
        print(
            f"WARNING: {fail_frac:.0%} of keyframes failed their VLM call — "
            f"the crop may be degraded (subject tracking relied on interpolation)."
        )
    if args.metrics_json:
        print(f"Metrics → {args.metrics_json}")
    return 0


if __name__ == "__main__":  # pragma: no cover - module entrypoint
    sys.exit(main())
