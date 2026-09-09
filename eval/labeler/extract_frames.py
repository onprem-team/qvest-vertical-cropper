#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 Qvest.US, LLC
# SPDX-License-Identifier: Apache-2.0
"""Extract labeling frames: one per second (offset 0.5s) from every clip in the dataset.

Writes JPEGs to ``<out>/<item>__<frame_idx>.jpg`` and an ordered index to ``<index>``.
Frame indices are positions in the DECODED stream (the same indexing the pipeline,
manifest, and scorer use), so labels line up with predictions.

    python eval/labeler/extract_frames.py --data /path/to/dataset

Dataset layout (see eval/README.md):
    <data>/manifest.json        {item_id: {n_frames, fps, width, height}}  (make_manifest.py)
    <data>/raw/<item>.mp4
"""
import argparse
import json
import os
import sys

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA = os.environ.get("VCROPPER_EVAL_DATA") or os.path.join(
    os.path.dirname(HERE), "data"
)


def sample_targets(fps, n_frames):
    """Decoded-frame indices at ~1/sec (0.5s offset), bounded by the clip length."""
    targets, k = [], 0
    while True:
        idx = int(round((0.5 + k) * fps))
        if idx >= n_frames:
            break
        targets.append(idx)
        k += 1
    return targets


def extract(manifest_path, raw_dir, out_dir, index_path):
    with open(manifest_path) as f:
        manifest = json.load(f)
    os.makedirs(out_dir, exist_ok=True)
    index = []
    for item_id in sorted(manifest):
        m = manifest[item_id]
        targets = set(sample_targets(m["fps"], m["n_frames"]))
        cap = cv2.VideoCapture(os.path.join(raw_dir, f"{item_id}.mp4"))
        i, got = 0, 0
        while True:
            if not cap.grab():
                break
            if i in targets:
                ok, frame = cap.retrieve()
                if not ok:
                    raise RuntimeError(f"retrieve failed {item_id}@{i}")
                name = f"{item_id}__{i}.jpg"
                cv2.imwrite(os.path.join(out_dir, name), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                index.append({"item_id": item_id, "frame_idx": i, "file": name,
                              "width": m["width"], "height": m["height"]})
                got += 1
            i += 1
        cap.release()
        print(f"{item_id}: {got} frames (of {len(targets)} planned)", file=sys.stderr)
        if got != len(targets):
            raise RuntimeError(f"{item_id}: extracted {got} != planned {len(targets)}")
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)
    print(f"total {len(index)} labeling frames -> {index_path}", file=sys.stderr)
    return index


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=DEFAULT_DATA,
                    help="dataset root (else $VCROPPER_EVAL_DATA, else eval/data)")
    ap.add_argument("--manifest", default=None, help="manifest.json (default <data>/manifest.json)")
    ap.add_argument("--raw", default=None, help="raw clips dir (default <data>/raw)")
    ap.add_argument("--out", default=os.path.join(HERE, "static", "frames"),
                    help="output frames dir (default eval/labeler/static/frames)")
    ap.add_argument("--index", default=os.path.join(HERE, "frames_index.json"),
                    help="output frame index (default eval/labeler/frames_index.json)")
    args = ap.parse_args(argv)
    manifest = args.manifest or os.path.join(args.data, "manifest.json")
    raw = args.raw or os.path.join(args.data, "raw")
    extract(manifest, raw, args.out, args.index)


if __name__ == "__main__":
    main()
