#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 Qvest.US, LLC
# SPDX-License-Identifier: Apache-2.0
"""Convert click-labels into scorer-ready ground truth.

Reads ``labels.json`` (written by ``app.py``) and ``splits.json``, drops skipped
frames, rounds coordinates, routes each label to its split, and writes
``<data>/<split>/ground_truth.jsonl`` in the scorer schema
(``{item_id, frame_idx, x, y}``).

    python eval/labeler/ingest_labels.py --data /path/to/dataset

This closes the one manual step in the original R&D workflow (labels were hand-converted).
"""
import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA = os.environ.get("VCROPPER_EVAL_DATA") or os.path.join(
    os.path.dirname(HERE), "data"
)


def to_ground_truth(labels, splits, *, ndigits=1):
    """Pure conversion: labels dict + splits dict -> {split: [ground_truth rows]}.

    labels: {"<item_id>__<frame_idx>": {"x", "y", "skipped"}}
    splits: {"<split>": [item_id, ...]}
    Skipped frames and frames with no coordinate are dropped; the rest are rounded and
    routed to the split their item belongs to. Raises if a labeled item is in no split.
    """
    item_to_split = {}
    for split, items in splits.items():
        for item in items:
            item_to_split[item] = split
    out = {split: [] for split in splits}
    for key, lab in labels.items():
        if lab.get("skipped"):
            continue
        if lab.get("x") is None or lab.get("y") is None:
            continue
        item_id, sep, fidx = key.rpartition("__")
        if not sep:
            raise ValueError(f"malformed label key {key!r} (expected '<item>__<frame_idx>')")
        split = item_to_split.get(item_id)
        if split is None:
            raise ValueError(f"item {item_id!r} (label {key!r}) is not in any split")
        out[split].append({
            "item_id": item_id,
            "frame_idx": int(fidx),
            "x": round(float(lab["x"]), ndigits),
            "y": round(float(lab["y"]), ndigits),
        })
    for split in out:
        out[split].sort(key=lambda r: (r["item_id"], r["frame_idx"]))
    return out


def write_ground_truth(by_split, data_root):
    """Write each split's rows to ``<data_root>/<split>/ground_truth.jsonl``."""
    for split, rows in by_split.items():
        out_dir = os.path.join(data_root, split)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "ground_truth.jsonl")
        with open(out_path, "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        print(f"{split}: wrote {len(rows)} labels -> {out_path}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=DEFAULT_DATA,
                    help="dataset root (else $VCROPPER_EVAL_DATA, else eval/data)")
    ap.add_argument("--labels", default=os.path.join(HERE, "labels.json"),
                    help="labels.json from the labeler (default eval/labeler/labels.json)")
    ap.add_argument("--splits", default=None, help="splits.json (default <data>/splits.json)")
    ap.add_argument("--ndigits", type=int, default=1, help="coordinate rounding (default 1)")
    args = ap.parse_args(argv)
    with open(args.labels) as f:
        labels = json.load(f)
    with open(args.splits or os.path.join(args.data, "splits.json")) as f:
        splits = json.load(f)
    write_ground_truth(to_ground_truth(labels, splits, ndigits=args.ndigits), args.data)


if __name__ == "__main__":
    main()
