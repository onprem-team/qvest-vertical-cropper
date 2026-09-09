#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 Qvest.US, LLC
# SPDX-License-Identifier: Apache-2.0
"""Render GT-vs-prediction debug videos over the full horizontal frame.

    python eval/render_debug.py --split val \
        --pred eval/results/pointing --data /path/to/dataset --out eval/results/pointing/debug

Green window = the 9:16 crop centered on the human label, linearly interpolated between
the ~1/sec labeled frames (edge-held outside) — "where it should look". Red window = the
crop path the app actually produced (from ``run_eval.py``'s prediction schema). At each
labeled frame (+/- a short flash) the exact GT point is drawn HIT/MISS per the frozen
scorer's point-in-window rule.

Predictions and ground truth are read from disk; the raw video is only needed to draw.
"""
import argparse
import json
import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import score as sc  # noqa: E402

from v_cropper.focus import interpolate_focus  # noqa: E402
from v_cropper.render import crop_dst_width  # noqa: E402

FLASH = 20  # frames on either side of a labeled frame during which the GT point is drawn


def _load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def compute_hits(labels, win_by_frame, frame_w, frame_h):
    """{labeled frame_idx -> bool} using the frozen scorer's rule (violations = MISS)."""
    hits = {}
    for t in labels:
        w = win_by_frame.get(t["frame_idx"])
        if w is None:
            continue
        hits[t["frame_idx"]] = (sc.window_violations(w, frame_w, frame_h) == 0
                                and sc.point_in_window(t["x"], t["y"], w))
    return hits


def gt_center_x(labels, frame_idx):
    """Interpolated ground-truth x for a frame (edge-held), or None if no labels."""
    return interpolate_focus({t["frame_idx"]: t["x"] for t in labels}, frame_idx)


def _render_item(item_id, output, labels, raw_path, out_path, out_height):
    win_by_frame = {w["frame_idx"]: w for w in output["crop_path"]}
    label_by_frame = {t["frame_idx"]: t for t in labels}
    cap = cv2.VideoCapture(str(raw_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {raw_path}")
    frame_w = output.get("frame_width") or int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = output.get("frame_height") or int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = output.get("fps") or cap.get(cv2.CAP_PROP_FPS) or 30.0
    dst_w = crop_dst_width(frame_w, frame_h)
    hits = compute_hits(labels, win_by_frame, frame_w, frame_h)

    out_h = out_height
    out_w = int(round(out_h * frame_w / frame_h / 2) * 2)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter.fourcc(*"mp4v"),
                             fps, (out_w, out_h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"VideoWriter failed: {out_path}")

    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        # green: GT ideal window (interpolated, edge-held), clamped inside the frame
        gx = gt_center_x(labels, i)
        if gx is not None:
            gx = min(max(gx, dst_w / 2), frame_w - dst_w / 2)
            g0 = int(round(gx - dst_w / 2))
            cv2.rectangle(frame, (g0, 0), (g0 + dst_w, frame_h - 1), (0, 220, 0), 5)
        # red: predicted window
        w = win_by_frame.get(i)
        if w is not None:
            p0 = int(round(w["x_center"] - w["width"] / 2))
            cv2.rectangle(frame, (p0, 2), (p0 + int(w["width"]), frame_h - 3), (0, 0, 255), 5)
        # labeled-point flash + HIT/MISS
        for li, t in label_by_frame.items():
            if abs(i - li) <= FLASH:
                col = (0, 220, 0) if hits.get(li) else (0, 0, 255)
                cv2.drawMarker(frame, (int(t["x"]), int(t["y"])), (255, 255, 255),
                               cv2.MARKER_CROSS, 70, 10)
                cv2.drawMarker(frame, (int(t["x"]), int(t["y"])), col, cv2.MARKER_CROSS, 60, 5)
                cv2.putText(frame, "HIT" if hits.get(li) else "MISS",
                            (int(t["x"]) + 40, max(60, int(t["y"]) - 40)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.4, col, 4)
        cv2.putText(frame, f"{item_id}  f{i}  t={i / fps:.1f}s   "
                    f"GREEN=ground-truth ideal crop  RED=app crop",
                    (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
        writer.write(cv2.resize(frame, (out_w, out_h)))
        i += 1
    cap.release()
    writer.release()
    n_hit = sum(1 for v in hits.values() if v)
    return i, n_hit, len(labels)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="val")
    ap.add_argument("--pred", required=True,
                    help="run dir containing <split>.jsonl (from run_eval.py)")
    ap.add_argument("--data", default=None, help="dataset root (else $VCROPPER_EVAL_DATA)")
    ap.add_argument("--out", default=None, help="debug output dir (default <pred>/debug)")
    ap.add_argument("--out-height", type=int, default=720)
    args = ap.parse_args(argv)

    data_root = args.data or os.environ.get("VCROPPER_EVAL_DATA")
    if not data_root or not os.path.isdir(data_root):
        sys.exit("Set --data or $VCROPPER_EVAL_DATA to the dataset root (raw/, <split>/...).")
    pred_file = os.path.join(args.pred, f"{args.split}.jsonl")
    preds = {r["item_id"]: r["output"] for r in _load_jsonl(pred_file)}
    truths = {}
    for t in _load_jsonl(os.path.join(data_root, args.split, "ground_truth.jsonl")):
        truths.setdefault(t["item_id"], []).append(t)

    out_dir = args.out or os.path.join(args.pred, "debug")
    os.makedirs(out_dir, exist_ok=True)
    for item_id in sorted(preds):
        labels = sorted(truths.get(item_id, []), key=lambda t: t["frame_idx"])
        raw_path = os.path.join(data_root, "raw", f"{item_id}.mp4")
        out_path = os.path.join(out_dir, f"{item_id}_debug.mp4")
        n, n_hit, n_lab = _render_item(item_id, preds[item_id], labels, raw_path,
                                       out_path, args.out_height)
        print(f"{item_id}: debug {n} frames -> {out_path} ({n_hit}/{n_lab} labeled frames hit)")


if __name__ == "__main__":
    main()
