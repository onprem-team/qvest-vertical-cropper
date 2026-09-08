#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 Qvest Group GmbH
# SPDX-License-Identifier: Apache-2.0
"""Generate data/manifest.json by probing the raw videos directly (vendored).

The manifest is the scorer's independent source of clip length/duration/dims (so a
pipeline cannot pad or thin its crop path). Frames are counted by actually decoding
(grab loop), not the container header, because CAP_PROP_FRAME_COUNT can be approximate.
Regenerate this in the environment that will run the eval, so decode counts match.

    python make_manifest.py [--raw data/raw] [--out data/manifest.json]
"""
import argparse
import glob
import json
import os

import cv2


def probe(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = 0
    while cap.grab():
        n += 1
    cap.release()
    if n == 0 or not fps or fps <= 0 or width <= 0 or height <= 0:
        raise RuntimeError(f"bad probe for {path}: n={n} fps={fps} {width}x{height}")
    return {"n_frames": n, "fps": round(fps, 4), "duration_sec": round(n / fps, 4),
            "width": width, "height": height}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/manifest.json")
    args = ap.parse_args()

    manifest = {}
    for p in sorted(glob.glob(os.path.join(args.raw, "*.mp4"))):
        item_id = os.path.splitext(os.path.basename(p))[0]
        manifest[item_id] = probe(p)
        print(f"{item_id}: {manifest[item_id]}")
    with open(args.out, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"wrote {args.out} ({len(manifest)} items)")


if __name__ == "__main__":
    main()
