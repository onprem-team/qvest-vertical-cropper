#!/usr/bin/env python3
"""Create a small non-sensitive landscape video for deployment smoke tests."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--frames", type=int, default=30)
    args = parser.parse_args(argv)
    if args.frames < 1:
        parser.error("--frames must be positive")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter.fourcc(*"mp4v"), 30.0, (160, 90),
    )
    if not writer.isOpened():
        raise RuntimeError(f"cannot create {output}")
    for index in range(args.frames):
        frame = np.full((90, 160, 3), 24, dtype=np.uint8)
        x = 20 + int((120 * index) / max(args.frames - 1, 1))
        cv2.circle(frame, (x, 45), 10, (0, 255, 255), -1)
        writer.write(frame)
    writer.release()


if __name__ == "__main__":
    main()
