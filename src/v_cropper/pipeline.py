# SPDX-FileCopyrightText: Copyright (c) 2026 Qvest Group GmbH
# SPDX-License-Identifier: Apache-2.0
"""Reusable, cancellable orchestration for VLM-guided vertical cropping."""
from __future__ import annotations

import concurrent.futures
import json
import os
import subprocess
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2

from .backend import VisionBackend, make_backend
from .focus import DEFAULT_SEND_WIDTH, extract_focus_points
from .prompts import DEFAULT_SPORT, resolve_prompt
from .render import render
from .smoothing import DEFAULT_SPRING_K

ProgressCallback = Callable[[str, float, dict | None], None]
CancelCallback = Callable[[], bool]

# Used only when the source fps is unknown or unusable (~2 fps against a 60 fps source).
FALLBACK_STRIDE = 30


class PipelineCancelled(RuntimeError):
    """Raised at a safe phase boundary when cancellation was requested."""


@dataclass(frozen=True)
class CropOptions:
    sport: str = DEFAULT_SPORT
    prompt: str | None = None
    sample_fps: float = 2.0
    sample_every: int | None = None
    send_width: int = DEFAULT_SEND_WIDTH
    spring_k: float = DEFAULT_SPRING_K
    concurrency: int = 8
    model: str | None = None
    scoreboard: bool = False
    scoreboard_sample_count: int = 10
    scoreboard_model: str | None = None
    scoreboard_position: str = "bottom"
    scoreboard_height_ratio: float = 0.16
    scoreboard_opacity: float = 0.85


@dataclass(frozen=True)
class PipelineResult:
    output_path: Path
    metrics: dict
    media: dict


def probe_media(path: str | Path) -> dict:
    """Return normalized ffprobe metadata for a local media file."""
    proc = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries",
            "format=duration,size:stream=index,codec_type,codec_name,width,height,r_frame_rate",
            "-of", "json", str(path),
        ],
        capture_output=True,
        check=True,
        text=True,
    )
    raw = json.loads(proc.stdout)
    streams = raw.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), {})
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    fmt = raw.get("format", {})
    return {
        "duration_sec": round(float(fmt.get("duration", 0) or 0), 4),
        "size_bytes": int(fmt.get("size", 0) or 0),
        "video_codec": video.get("codec_name"),
        "width": video.get("width"),
        "height": video.get("height"),
        "audio_codec": audio.get("codec_name") if audio else None,
        "has_audio": audio is not None,
    }


def trim_source(source: str, output: Path, *, in_s: float | None = None, out_s: float | None = None) -> Path:
    """Use ffmpeg input seeking/range reads to materialize only the requested clip."""
    cmd = ["ffmpeg", "-y", "-v", "error"]
    if in_s is not None:
        cmd += ["-ss", str(in_s)]
    cmd += ["-i", source]
    if out_s is not None:
        duration = out_s - (in_s or 0)
        cmd += ["-t", str(duration)]
    cmd += [
        "-map", "0:v:0", "-map", "0:a?", "-c:v", "libx264", "-preset", "veryfast",
        "-crf", "18", "-c:a", "aac", "-movflags", "+faststart", str(output),
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    return output


def mux_source_audio(video_path: Path, source_path: Path, output_path: Path) -> Path:
    """Copy cropped H.264 video and mux the trimmed source audio when present."""
    subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error", "-i", str(video_path), "-i", str(source_path),
            "-map", "0:v:0", "-map", "1:a?", "-c:v", "copy", "-c:a", "aac", "-shortest",
            "-movflags", "+faststart", str(output_path),
        ],
        capture_output=True,
        check=True,
    )
    return output_path


def _probe_cv(path: Path) -> tuple[float, int]:
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return fps, frames


def resolve_stride(sample_every: int | None, sample_fps: float, source_fps: float) -> int:
    """Resolve the keyframe stride: explicit ``sample_every`` wins, else derive from fps.

    Single source of truth for the CLI, the service, and the eval harness. Each used to
    resolve this independently, so a benchmark could sample a clip at a different rate than
    the deployed service did and still report the number as representative.
    """
    if sample_every is not None:
        return max(1, sample_every)
    if source_fps and source_fps > 0 and sample_fps > 0:
        return max(1, round(source_fps / sample_fps))
    return FALLBACK_STRIDE


def _stride(options: CropOptions, fps: float) -> int:
    return resolve_stride(options.sample_every, options.sample_fps, fps)


def _cost(backend: VisionBackend, price_in: float, price_out: float) -> float:
    usage = backend.usage_totals
    return usage.get("prompt_tokens", 0) / 1e6 * price_in + usage.get("completion_tokens", 0) / 1e6 * price_out


def run_pipeline(
    source: str | Path,
    output_path: str | Path,
    *,
    options: CropOptions | None = None,
    in_s: float | None = None,
    out_s: float | None = None,
    work_dir: str | Path | None = None,
    progress: ProgressCallback | None = None,
    cancelled: CancelCallback | None = None,
    backend: VisionBackend | None = None,
    scoreboard_backend: VisionBackend | None = None,
    source_is_trimmed: bool = False,
) -> PipelineResult:
    """Run trim → VLM focus → crop → audio mux and return metrics/metadata."""
    options = options or CropOptions()
    output_path = Path(output_path)
    root = Path(work_dir or output_path.parent)
    root.mkdir(parents=True, exist_ok=True)
    notify = progress or (lambda _phase, _pct, _detail=None: None)
    is_cancelled = cancelled or (lambda: False)

    def checkpoint(phase: str, pct: float, detail: dict | None = None) -> None:
        if is_cancelled():
            raise PipelineCancelled("crop job cancelled")
        notify(phase, pct, detail)

    started = time.monotonic()
    trimmed = Path(source) if source_is_trimmed else root / "source-trimmed.mp4"
    checkpoint("trimming", 5)
    if not source_is_trimmed:
        trim_source(str(source), trimmed, in_s=in_s, out_s=out_s)
    source_media = probe_media(trimmed)
    checkpoint("analyzing", 20, {"duration_sec": source_media["duration_sec"]})

    fps, frames = _probe_cv(trimmed)
    if not fps or not frames:
        raise RuntimeError("No keyframes were produced — source video is unreadable")
    stride = _stride(options, fps)
    prompt = options.prompt or resolve_prompt(sport=options.sport)
    focus_backend = backend or make_backend(model=options.model)
    game_state = overlay_style = scoreboard_debug = None

    if options.scoreboard:
        from .scoreboard_ocr import extract_scoreboard, sample_frames
        from .scoreboard_render import OverlayStyle
        from .scoreboard_state import merge_readings_with_debug

        sb_backend = scoreboard_backend or make_backend(model=options.scoreboard_model or options.model)
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            focus_future = pool.submit(
                extract_focus_points, str(trimmed), backend=focus_backend, prompt=prompt,
                sample_every=stride, concurrency=options.concurrency, send_width=options.send_width,
            )
            score_future = pool.submit(
                lambda: merge_readings_with_debug(
                    extract_scoreboard(sample_frames(str(trimmed), options.scoreboard_sample_count), backend=sb_backend)
                )
            )
            focus_map, _, failed = focus_future.result()
            game_state, scoreboard_debug = score_future.result()
        overlay_style = OverlayStyle(
            bg_opacity=options.scoreboard_opacity,
            height_ratio=options.scoreboard_height_ratio,
            position=options.scoreboard_position,
        )
    else:
        sb_backend = None
        focus_map, _, failed = extract_focus_points(
            str(trimmed), backend=focus_backend, prompt=prompt, sample_every=stride,
            concurrency=options.concurrency, send_width=options.send_width,
        )

    total = len(focus_map) + failed
    if total == 0:
        raise RuntimeError("No keyframes were produced — source video is unreadable")
    if not focus_map:
        # Every keyframe failed, so the model contributed nothing and render() falls back to
        # a static centre crop. That looks like a valid result to a caller, so report it as
        # a failure rather than returning an untracked clip with a success status.
        raise RuntimeError(
            f"All {failed} keyframes failed — the VLM returned no usable focus points. "
            "Check the provider endpoint, credentials, and model id."
        )
    checkpoint("cropping", 65, {"keyframes_ok": len(focus_map), "keyframes_failed": failed})
    silent = root / "crop-silent.mp4"
    render(
        str(trimmed), focus_map, silent, spring_k=options.spring_k, game_state=game_state,
        overlay_style=overlay_style, scoreboard_debug=scoreboard_debug,
    )
    checkpoint("muxing_audio", 85)
    mux_source_audio(silent, trimmed, output_path)
    media = probe_media(output_path)
    elapsed = time.monotonic() - started
    duration = source_media["duration_sec"]
    price_in = float(os.getenv("VCROPPER_PRICE_IN", "0.30"))
    price_out = float(os.getenv("VCROPPER_PRICE_OUT", "2.50"))
    estimated_cost = _cost(focus_backend, price_in, price_out)
    if sb_backend:
        estimated_cost += _cost(sb_backend, price_in, price_out)
    metrics = {
        "wall_time_sec": round(elapsed, 3),
        "video_fps": round(fps, 4),
        "video_frames": frames,
        "video_duration_sec": duration,
        "rtf": round(elapsed / duration, 4) if duration else None,
        "sport": options.sport if not options.prompt else "custom",
        "sample_every": stride,
        "send_width": options.send_width,
        "spring_k": options.spring_k,
        "keyframes_ok": len(focus_map),
        "keyframes_failed": failed,
        "keyframe_fail_fraction": round(failed / total, 4),
        "focus_model": focus_backend.model,
        "focus_usage": dict(focus_backend.usage_totals),
        "scoreboard_usage": dict(sb_backend.usage_totals) if sb_backend else None,
        "est_cost_usd": round(estimated_cost, 6),
        "options": asdict(options),
    }
    checkpoint("complete", 100, {"media": media})
    return PipelineResult(output_path=output_path, metrics=metrics, media=media)
