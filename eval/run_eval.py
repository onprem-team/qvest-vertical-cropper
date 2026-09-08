#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 Qvest Group GmbH
# SPDX-License-Identifier: Apache-2.0
"""Run v-cropper on the labeled benchmark clips and score against the frozen scorer.

Emits predictions in the scorer schema (one crop window per DECODED frame 0..n-1) and
then scores them, so we can prove coverage parity with the R&D v0013 winner.

    VCROPPER_EVAL_DATA=/path/to/vertical-cropper/data \
      uv run python eval/run_eval.py --split val --out eval/results/pointing

Use the SAME prompt config you ship: a built-in preset (``--sport hockey``) or a custom
prompt for a novel sport (``--prompt "..."`` / ``--prompt-file my_prompt.txt``), resolved
with the same precedence as the CLI (inline > prompt-file > sport preset > football).

Dataset layout expected under --data / $VCROPPER_EVAL_DATA:
    raw/<item>.mp4
    splits.json                     {"val": [...], "test": [...]}
    val/ground_truth.jsonl          {item_id, frame_idx, x, y}
    test/ground_truth.jsonl

The raw clips are copyrighted and live OUTSIDE this repo; only the path is referenced.
The manifest is regenerated in THIS environment (decode-count must match) into --out.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# eval/ is a sibling of src/; import the installed package + local helpers.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import score as scorer  # noqa: E402
from cache import CachingBackend  # noqa: E402
from make_manifest import probe  # noqa: E402

from v_cropper.backend import make_backend  # noqa: E402
from v_cropper.cli import (  # noqa: E402
    DEFAULT_PRICE_IN,
    DEFAULT_PRICE_OUT,
    _estimate_cost,
    _resolve,
)
from v_cropper.focus import extract_focus_points  # noqa: E402
from v_cropper.pipeline import CropOptions, resolve_stride  # noqa: E402
from v_cropper.prompts import resolve_prompt  # noqa: E402
from v_cropper.render import compute_crop_path  # noqa: E402
from v_cropper.smoothing import DEFAULT_SPRING_K  # noqa: E402


def perf_summary(usage, wall_time_sec, rtf, cache_hits):
    """Roll VLM usage + wall time into a JSON-serializable perf/cost block.

    Cost uses the same $/1M-token envs as the CLI (VCROPPER_PRICE_IN/OUT); it is a rough
    flash-tier estimate, not a bill.
    """
    price_in = _resolve(None, "VCROPPER_PRICE_IN", DEFAULT_PRICE_IN, float)
    price_out = _resolve(None, "VCROPPER_PRICE_OUT", DEFAULT_PRICE_OUT, float)
    return {
        "vlm_calls": usage.get("api_calls", 0),
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0),
        "est_cost_usd": round(_estimate_cost(usage, price_in, price_out), 4),
        "price_in_per_mtok": price_in,
        "price_out_per_mtok": price_out,
        "wall_time_sec": round(wall_time_sec, 2),
        "cache_hits": cache_hits,
        "rtf": rtf,
    }


def build_manifest(raw_dir, item_ids, out_path):
    """Regenerate the manifest in THIS env (decode counts must match the scorer)."""
    manifest = {item: probe(str(Path(raw_dir) / f"{item}.mp4")) for item in item_ids}
    Path(out_path).write_text(json.dumps(manifest, indent=2))
    return manifest


def run(args):
    data_root = Path(args.data or os.environ.get("VCROPPER_EVAL_DATA", ""))
    if not data_root or not data_root.exists():
        sys.exit("Set --data or $VCROPPER_EVAL_DATA to the dataset root (raw/, splits.json, ...).")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = json.loads((data_root / "splits.json").read_text())
    if args.split not in splits:
        sys.exit(f"unknown split {args.split!r}; have {sorted(splits)}")
    items = splits[args.split]

    # Resolve the pointing prompt with the SAME precedence as the CLI so a benchmarked
    # config matches what you actually ship: inline > prompt-file > sport preset > football.
    try:
        prompt = resolve_prompt(inline=args.prompt, prompt_file=args.prompt_file, sport=args.sport)
    except (ValueError, OSError) as e:
        sys.exit(str(e))

    manifest = build_manifest(data_root / "raw", items, out_dir / "manifest.json")

    inner = make_backend(model=args.model, base_url=getattr(args, "base_url", None))
    backend = inner if args.no_cache else CachingBackend(inner, out_dir / ".cache")

    records = []
    wall_total = 0.0
    strides = {}
    for item in items:
        clip = data_root / "raw" / f"{item}.mp4"
        # Resolve the stride the way the shipped pipeline does, from this clip's own fps.
        # Hardcoding it here would benchmark a sampling rate the service never runs at.
        stride = resolve_stride(args.sample_every, args.sample_fps, manifest[item].get("fps", 0.0))
        strides[item] = stride
        t0 = time.perf_counter()
        try:
            focus_map, _reasoning, n_failed = extract_focus_points(
                str(clip), backend=backend,
                sample_every=stride, concurrency=args.concurrency,
                send_width=args.send_width, prompt=prompt,
            )
            windows, fw, fh, fps = compute_crop_path(clip, focus_map, spring_k=args.spring_k)
        except Exception as e:
            # A clip that cannot be processed is left out entirely; the scorer counts a
            # missing prediction as all-misses (never silently dropped from the denominator).
            print(f"[eval] {item}: FAILED ({e}) — recorded as missing prediction",
                  file=sys.stderr)
            continue
        elapsed = time.perf_counter() - t0
        wall_total += elapsed
        n_ok = len(focus_map)
        print(f"[eval] {item}: {len(windows)} frames, {n_ok} keyframes ok / {n_failed} failed, "
              f"stride={stride}, {elapsed:.1f}s", file=sys.stderr)
        records.append({
            "item_id": item,
            "output": {
                "crop_path": windows,
                "frame_width": fw,
                "frame_height": fh,
                "fps": fps,
                "processing_time_sec": round(elapsed, 4),
            },
            "reasoning": None,
        })

    pred_path = out_dir / f"{args.split}.jsonl"
    with open(pred_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    truth = scorer._load_jsonl(data_root / args.split / "ground_truth.jsonl")
    result = scorer.evaluate(records, truth, args.split, manifest=manifest)
    result.pop("_per_frame_rows", None)
    cache_hits = backend.cache_hits if isinstance(backend, CachingBackend) else 0
    result["perf"] = perf_summary(inner.usage_totals, wall_total, result.get("rtf"), cache_hits)
    # Record the sampling actually used so two runs are comparable without guessing.
    result["sampling"] = {
        "sample_fps": args.sample_fps,
        "sample_every": args.sample_every,
        "resolved_stride": strides,
    }
    (out_dir / f"metrics-{args.split}.json").write_text(json.dumps(result, indent=2))
    print(f"\n[eval] split={args.split} coverage={result['primary_metric']:.4f} "
          f"worst_clip={result['guardrail_metric']:.4f} max_jerk={result['max_jerk']} "
          f"rtf={result['rtf']} structural_ok={result['structural_ok']}")
    perf = result["perf"]
    print(f"[eval] perf: {perf['vlm_calls']} calls, {perf['total_tokens']} tokens, "
          f"~${perf['est_cost_usd']}, {perf['wall_time_sec']}s wall, cache_hits={cache_hits}")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--data", default=None, help="dataset root (else $VCROPPER_EVAL_DATA)")
    ap.add_argument("--out", default="eval/results/run")
    ap.add_argument("--model", default=None, help="VLM model id (else provider default)")
    ap.add_argument("--base-url", default=None,
                    help="OpenAI-compatible VLM endpoint (overrides $VCROPPER_BASE_URL)")
    ap.add_argument("--sample-fps", type=float, default=CropOptions.sample_fps,
                    help="keyframes sampled per second (default matches the shipped service)")
    ap.add_argument("--sample-every", type=int, default=None,
                    help="sample every Nth frame (overrides --sample-fps)")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--send-width", type=int, default=768, help="downscale width sent to the VLM")
    ap.add_argument("--sport", default=None, help="prompt preset (default: football)")
    ap.add_argument("--prompt", default=None,
                    help="inline pointing prompt for a custom sport (overrides --sport)")
    ap.add_argument("--prompt-file", default=None,
                    help="read the pointing prompt from a file (overrides --sport)")
    ap.add_argument("--spring-k", type=float, default=DEFAULT_SPRING_K,
                    help="critically-damped spring stiffness (lower = smoother)")
    ap.add_argument("--no-cache", action="store_true")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
