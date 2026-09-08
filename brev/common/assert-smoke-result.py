#!/usr/bin/env python3
"""Fail a deployment smoke test unless v-cropper produced a valid live focus result."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-fail-fraction", type=float, default=0.2)
    args = parser.parse_args(argv)

    metrics = json.loads(Path(args.metrics).read_text())
    output = Path(args.output)
    if not output.is_file() or output.stat().st_size == 0:
        raise SystemExit("smoke output is missing or empty")
    if metrics.get("keyframes_ok", 0) < 1:
        raise SystemExit("smoke test received no valid focus points")
    if metrics.get("keyframe_fail_fraction", 1.0) > args.max_fail_fraction:
        raise SystemExit("smoke test keyframe failure rate exceeded the allowed threshold")


if __name__ == "__main__":
    main()
