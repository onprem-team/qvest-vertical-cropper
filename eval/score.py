#!/usr/bin/env python3
"""Scorer for the vertical-cropper task (frozen; vendored from the R&D instance).

Compares predicted crop paths against human point-labels:

    python score.py --pred runs/val.jsonl \
                    --truth data/val/ground_truth.jsonl --split val --json \
                    --manifest data/manifest.json --outdir runs/evaluation

Prints a single JSON object on the LAST stdout line. Metrics:
  primary_metric    coverage — fraction of labeled frames (pooled over the split) whose
                    labeled action point lies inside a VALID predicted crop window
                    (a window violating the 9:16/bounds constraints scores as a MISS).
  guardrail_metric  worst-clip coverage.
  rtf               total processing time / total video duration (duration from the
                    independent manifest; time is pipeline-reported).
  structural_ok     False if ANY of: constraint violations, path-length mismatch vs the
                    manifest, jerk above threshold, missing rtf, identical paths across
                    items.
  per_item          [{item_id, coverage, n_labeled, rtf, jerk, path_x_std,
                      missing_prediction, path_length_mismatch}]
Missing predictions (whole clip or single frame) count as MISSES, never dropped.
The manifest (data/manifest.json: {item_id: {n_frames, fps, width, height}}) is generated
by probing the raw videos directly (make_manifest.py) — independent of the pipeline.
"""
import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

SCORER_VERSION = "1.0.0"

# Mean |third difference| of x_center (px/frame^3) above which a path is judged too
# jittery to watch. Calibrated 2026-06-10: smooth follow paths <=0.15, sub-pixel residual
# noise ~1.8, raw >=2px/frame jitter >=7. FROZEN.
JERK_THRESHOLD = 2.0

ASPECT = 9.0 / 16.0
ASPECT_TOL_PX = 1.5  # |width - height*9/16| tolerance for integer-pixel windows


def point_in_window(x, y, window):
    """Inclusive point-in-rect test against a crop window dict."""
    half_w = window["width"] / 2.0
    half_h = window["height"] / 2.0
    return (window["x_center"] - half_w <= x <= window["x_center"] + half_w
            and window["y_center"] - half_h <= y <= window["y_center"] + half_h)


def window_violations(window, frame_w, frame_h):
    """Count hard-constraint violations for one window (0 = valid)."""
    vals = [window["x_center"], window["y_center"], window["width"], window["height"]]
    if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in vals):
        return 1  # non-finite/absent coordinates: invalid, skip further checks
    v = 0
    half_w = window["width"] / 2.0
    half_h = window["height"] / 2.0
    if (window["x_center"] - half_w < -1e-6 or window["x_center"] + half_w > frame_w + 1e-6
            or window["y_center"] - half_h < -1e-6
            or window["y_center"] + half_h > frame_h + 1e-6):
        v += 1
    if abs(window["width"] - window["height"] * ASPECT) > ASPECT_TOL_PX:
        v += 1
    if window["height"] > frame_h + 1e-6:
        v += 1
    return v


def mean_abs_jerk(crop_path):
    """Mean |third difference| of x_center in px/frame^3 (0 if fewer than 4 frames)."""
    xs = [w["x_center"] for w in sorted(crop_path, key=lambda w: w["frame_idx"])]
    if len(xs) < 4:
        return 0.0
    d1 = [b - a for a, b in zip(xs, xs[1:])]
    d2 = [b - a for a, b in zip(d1, d1[1:])]
    d3 = [b - a for a, b in zip(d2, d2[1:])]
    return sum(abs(j) for j in d3) / len(d3)


def _std(values):
    if not values:
        return 0.0
    m = sum(values) / len(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / len(values))


def evaluate(preds, truths, split, manifest=None):
    """Pure scoring function. preds/truths = lists of dicts (see module docstring).

    manifest: {item_id: {"n_frames", "fps", "width", "height"}} probed independently
    from the raw videos. When given, every item must appear in it; the manifest's
    width/height are AUTHORITATIVE for all bounds checks; the path's frame indices must be
    exactly {0..n_frames-1}; rtf uses the manifest duration.
    """
    if not truths:
        raise ValueError("empty ground truth — refusing to score")

    pred_by_item = {}
    for p in preds:
        if p["item_id"] in pred_by_item:
            raise ValueError(f"duplicate prediction for item {p['item_id']!r}")
        pred_by_item[p["item_id"]] = p

    truth_by_item = defaultdict(list)
    for t in truths:
        truth_by_item[t["item_id"]].append(t)

    if manifest is not None:
        unknown = (set(truth_by_item) | set(pred_by_item)) - set(manifest)
        if unknown:
            raise ValueError(f"items not in manifest: {sorted(unknown)}")
        for item_id in set(manifest) & set(pred_by_item):
            m = manifest[item_id]
            if "width" not in m or "height" not in m:
                raise ValueError(f"manifest entry for {item_id!r} lacks width/height — "
                                 f"regenerate with make_manifest.py")

    per_item = []
    per_frame_rows = []
    hits = 0
    n_labeled = 0
    offsets = []
    violations = 0
    total_proc, total_dur = 0.0, 0.0
    path_signatures = []

    for item_id in sorted(truth_by_item):
        labels = truth_by_item[item_id]
        n_labeled += len(labels)
        p = pred_by_item.get(item_id)
        if p is None:
            per_item.append({"item_id": item_id, "coverage": 0.0,
                             "n_labeled": len(labels), "rtf": None, "jerk": None,
                             "path_x_std": None, "missing_prediction": True})
            for t in labels:
                per_frame_rows.append([item_id, t["frame_idx"], t["x"], t["y"],
                                       None, None, 0, None])
            continue

        out = p["output"]
        path = out["crop_path"]
        if manifest is not None:
            # the manifest's probed dimensions are the ruler — never the prediction's
            frame_w, frame_h = manifest[item_id]["width"], manifest[item_id]["height"]
            if (out.get("frame_width") != frame_w or out.get("frame_height") != frame_h):
                raise ValueError(
                    f"prediction for {item_id!r} reports frame dims "
                    f"{out.get('frame_width')}x{out.get('frame_height')} but the manifest "
                    f"probed {frame_w}x{frame_h} — pipeline bug or gaming attempt")
        else:
            frame_w, frame_h = out["frame_width"], out["frame_height"]
        win_by_frame = {}
        for w in path:
            if w["frame_idx"] in win_by_frame:
                raise ValueError(
                    f"duplicate frame_idx {w['frame_idx']} in crop_path of {item_id!r}")
            win_by_frame[w["frame_idx"]] = w
        item_violations = sum(window_violations(w, frame_w, frame_h) for w in path)

        length_mismatch = False
        if manifest is not None:
            expected_n = manifest[item_id]["n_frames"]
            if set(win_by_frame) != set(range(expected_n)):
                length_mismatch = True  # exactly one window per frame 0..n-1, no more
                item_violations += 1
        violations += item_violations

        item_hits = 0
        for t in labels:
            w = win_by_frame.get(t["frame_idx"])
            if w is None:
                per_frame_rows.append([item_id, t["frame_idx"], t["x"], t["y"],
                                       None, None, 0, None])
                continue
            # a window that violates the hard constraints scores as a MISS
            hit = window_violations(w, frame_w, frame_h) == 0 \
                and point_in_window(t["x"], t["y"], w)
            item_hits += hit
            off = abs(w["x_center"] - t["x"]) / frame_w
            offsets.append(off)
            per_frame_rows.append([item_id, t["frame_idx"], t["x"], t["y"],
                                   w["x_center"], w["width"], int(hit), round(off, 5)])
        hits += item_hits

        proc = out.get("processing_time_sec")
        if proc is not None and (not math.isfinite(proc) or proc <= 0):
            raise ValueError(f"invalid processing_time_sec for {item_id!r}: {proc}")
        if manifest is not None:
            m = manifest[item_id]
            dur = m["n_frames"] / m["fps"] if m.get("fps") else 0.0
        else:
            fps = out.get("fps") or 0.0
            dur = len(path) / fps if fps else 0.0
        rtf = (proc / dur) if (proc is not None and dur > 0) else None
        if rtf is not None:
            total_proc += proc
            total_dur += dur

        xs = [w["x_center"] for w in path]
        path_signatures.append(tuple(round(x, 3) for x in xs))
        per_item.append({"item_id": item_id,
                         "coverage": item_hits / len(labels),
                         "n_labeled": len(labels),
                         "rtf": round(rtf, 4) if rtf is not None else None,
                         "jerk": round(mean_abs_jerk(path), 4),
                         "path_x_std": round(_std(xs), 2),
                         "missing_prediction": False,
                         "path_length_mismatch": length_mismatch})

    jerks = [i["jerk"] for i in per_item if i["jerk"] is not None]
    coverages = [i["coverage"] for i in per_item]
    rtf = round(total_proc / total_dur, 4) if total_dur > 0 else None
    smoothness_ok = (max(jerks) <= JERK_THRESHOLD) if jerks else False
    identical = len(path_signatures) != len(set(path_signatures))
    missing_rtf = any(i["rtf"] is None and not i["missing_prediction"] for i in per_item)
    return {
        "split": split,
        "n_items": len(truth_by_item),
        "n_labeled_frames": n_labeled,
        "primary_metric": hits / n_labeled,
        "guardrail_metric": min(coverages),
        "rtf": rtf,
        "rtf_realtime_ok": rtf is not None and rtf <= 1.0,
        "max_jerk": max(jerks) if jerks else None,
        "smoothness_ok": smoothness_ok,
        "mean_norm_x_offset": (sum(offsets) / len(offsets)) if offsets else None,
        "constraint_violations": violations,
        "identical_paths_across_items": identical,
        "structural_ok": (violations == 0 and smoothness_ok and not identical
                          and not missing_rtf
                          and not any(i["missing_prediction"] for i in per_item)),
        "per_item": per_item,
        "scorer_version": SCORER_VERSION,
        "_per_frame_rows": per_frame_rows,  # stripped before printing; written to CSV
    }


def _load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--truth", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--manifest", required=True,
                    help="data/manifest.json probed from the raw videos (make_manifest.py)")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    with open(args.manifest) as f:
        manifest = json.load(f)
    result = evaluate(_load_jsonl(args.pred), _load_jsonl(args.truth), args.split,
                      manifest=manifest)
    per_frame_rows = result.pop("_per_frame_rows")

    if args.outdir:
        outdir = Path(args.outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        with open(outdir / "metrics.json", "w") as f:
            json.dump(result, f, indent=2)
        with open(outdir / "per_frame.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["item_id", "frame_idx", "x_label", "y_label",
                        "x_center", "width", "hit", "norm_offset"])
            w.writerows(per_frame_rows)

    print(f"[score] split={result['split']} items={result['n_items']} "
          f"labeled={result['n_labeled_frames']} coverage={result['primary_metric']:.4f} "
          f"worst_clip={result['guardrail_metric']:.4f} rtf={result['rtf']} "
          f"max_jerk={result['max_jerk']} violations={result['constraint_violations']} "
          f"structural_ok={result['structural_ok']}",
          file=sys.stderr)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
