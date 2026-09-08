#!/usr/bin/env python3
"""Write non-secret provenance for a Brev workstation profile."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--source-revision", required=True)
    args = parser.parse_args(argv)

    manifest = {
        "base_url": args.base_url,
        "model": args.model,
        "profile": args.profile,
        "source_revision": args.source_revision,
    }
    Path(args.output).write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
