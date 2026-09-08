#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 Qvest Group GmbH
# SPDX-License-Identifier: Apache-2.0
"""End-to-end verification of a running v-cropper API.

Exercises the real deployed path rather than a stub: object storage, presigned GET/PUT,
bearer auth, the async job lifecycle, and the produced artifact. Writes a non-secret
evidence file and exits non-zero on any failure.

The presigned URLs are deliberately signed for the *service's* view of object storage
(``--storage-internal-endpoint``, e.g. ``http://minio:9000``) while this script uploads and
downloads over its own view (``--storage-endpoint``, e.g. ``http://127.0.0.1:9000``).
SigV4 signs the Host header, so a URL signed for localhost is rejected when the container
requests it by service name.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import boto3
import httpx
from botocore.client import Config
from botocore.exceptions import ClientError

TERMINAL = {"succeeded", "failed", "cancelled"}


def _client(endpoint: str, key: str, secret: str):
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=key,
        aws_secret_access_key=secret,
        config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
        region_name="us-east-1",
    )


def _wait_for_storage(client, timeout_sec: float) -> None:
    deadline = time.monotonic() + timeout_sec
    last: Exception | None = None
    while time.monotonic() < deadline:
        try:
            client.list_buckets()
            return
        except Exception as exc:  # botocore raises several unrelated types here
            last = exc
            time.sleep(1)
    raise SystemExit(f"object storage did not become reachable within {timeout_sec}s: {last}")


def _wait_for_service(base_url: str, timeout_sec: float) -> None:
    deadline = time.monotonic() + timeout_sec
    last = ""
    while time.monotonic() < deadline:
        try:
            response = httpx.get(f"{base_url}/readyz", timeout=5)
            if response.status_code == 200:
                return
            last = f"{response.status_code} {response.text[:200]}"
        except httpx.HTTPError as exc:
            last = str(exc)
        time.sleep(1)
    raise SystemExit(f"service did not become ready within {timeout_sec}s: {last}")


def _probe(path: Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,codec_name", "-show_entries", "format=duration",
         "-of", "json", str(path)],
        check=True, capture_output=True, text=True,
    )
    data = json.loads(out.stdout)
    stream = (data.get("streams") or [{}])[0]
    return {
        "width": stream.get("width"),
        "height": stream.get("height"),
        "codec": stream.get("codec_name"),
        "duration_sec": float(data.get("format", {}).get("duration", 0.0)),
    }


def check(condition: bool, message: str, failures: list[str]) -> bool:
    print(f"  [{'ok' if condition else 'FAIL'}] {message}")
    if not condition:
        failures.append(message)
    return condition


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True, help="service base URL, e.g. http://127.0.0.1:8090")
    ap.add_argument("--token", required=True, help="CROPPER_API_TOKEN for the service")
    ap.add_argument("--storage-endpoint", required=True, help="object storage as seen by this script")
    ap.add_argument("--storage-internal-endpoint", required=True, help="object storage as seen by the service")
    ap.add_argument("--storage-key", required=True)
    ap.add_argument("--storage-secret", required=True)
    ap.add_argument("--bucket", default="vcropper-verify")
    ap.add_argument("--video", required=True, help="local source clip to upload")
    ap.add_argument("--evidence", required=True, help="path for the non-secret evidence JSON")
    ap.add_argument("--timeout", type=float, default=900, help="seconds to wait for the job")
    args = ap.parse_args()

    base_url = args.base_url.rstrip("/")
    auth = {"Authorization": f"Bearer {args.token}"}
    failures: list[str] = []

    admin = _client(args.storage_endpoint, args.storage_key, args.storage_secret)
    signer = _client(args.storage_internal_endpoint, args.storage_key, args.storage_secret)

    print("[verify] waiting for dependencies")
    _wait_for_storage(admin, 120)
    _wait_for_service(base_url, 180)

    print("[verify] preparing object storage")
    try:
        admin.create_bucket(Bucket=args.bucket)
    except ClientError as exc:
        if exc.response["Error"]["Code"] not in {"BucketAlreadyOwnedByYou", "BucketAlreadyExists"}:
            raise
    admin.upload_file(args.video, args.bucket, "source.mp4")

    source_url = signer.generate_presigned_url(
        "get_object", Params={"Bucket": args.bucket, "Key": "source.mp4"}, ExpiresIn=3600)
    destination_url = signer.generate_presigned_url(
        "put_object",
        Params={"Bucket": args.bucket, "Key": "output.mp4", "ContentType": "video/mp4"},
        ExpiresIn=3600,
    )

    print("[verify] authentication")
    payload = {"source_url": source_url, "destination_url": destination_url,
               "sample_every": 30, "concurrency": 1}
    anon = httpx.post(f"{base_url}/v1/jobs", json=payload, timeout=30)
    check(anon.status_code == 401, "unauthenticated submit is rejected with 401", failures)
    check(args.token not in anon.text, "rejection does not echo the expected token", failures)
    bad = httpx.post(f"{base_url}/v1/jobs", json=payload,
                     headers={"Authorization": "Bearer wrong"}, timeout=30)
    check(bad.status_code == 401, "wrong bearer token is rejected with 401", failures)

    print("[verify] health endpoints stay open")
    check(httpx.get(f"{base_url}/healthz", timeout=10).status_code == 200,
          "/healthz is reachable without a credential", failures)

    print("[verify] submitting job")
    accepted = httpx.post(f"{base_url}/v1/jobs", json=payload, headers=auth, timeout=30)
    if accepted.status_code != 202:
        print(f"  [FAIL] submit returned {accepted.status_code}: {accepted.text[:500]}")
        return 1
    job_id = accepted.json()["job_id"]
    print(f"  job_id={job_id}")

    deadline = time.monotonic() + args.timeout
    view: dict = {}
    while time.monotonic() < deadline:
        view = httpx.get(f"{base_url}/v1/jobs/{job_id}", headers=auth, timeout=30).json()
        if view["status"] in TERMINAL:
            break
        time.sleep(2)
    else:
        print(f"  [FAIL] job did not finish within {args.timeout}s (last={view})")
        return 1

    print(f"[verify] job finished: status={view['status']} phase={view.get('phase')}")
    if not check(view["status"] == "succeeded", f"job succeeded (error={view.get('error')})", failures):
        return 1

    serialized = json.dumps(view)
    check("source_url" not in serialized and "destination_url" not in serialized,
          "job view never returns signed URLs", failures)
    check("Signature" not in serialized, "job view never returns a signature", failures)

    # A crop is produced even when every VLM call fails: render() falls back to a static
    # centre crop, so "succeeded" plus a 9:16 file does not prove the model was reached.
    metrics = view.get("metrics") or {}
    check(metrics.get("keyframes_ok", 0) > 0,
          f"the VLM produced usable focus points (keyframes_ok={metrics.get('keyframes_ok')})",
          failures)
    check((metrics.get("focus_usage") or {}).get("api_calls", 0) > 0,
          f"the VLM was actually called (api_calls={(metrics.get('focus_usage') or {}).get('api_calls')})",
          failures)
    check(metrics.get("keyframe_fail_fraction", 1.0) < 0.5,
          f"most keyframes succeeded (fail_fraction={metrics.get('keyframe_fail_fraction')})",
          failures)

    print("[verify] inspecting the produced artifact")
    local = Path(args.evidence).parent / "verified-output.mp4"
    local.parent.mkdir(parents=True, exist_ok=True)
    admin.download_file(args.bucket, "output.mp4", str(local))
    media = _probe(local)
    check(local.stat().st_size > 0, "output object is non-empty", failures)
    check(media["duration_sec"] > 0, f"output has a positive duration ({media['duration_sec']:.2f}s)", failures)
    check(media["width"] is not None and media["height"] is not None, "output has a decodable video stream", failures)
    if media["width"] and media["height"]:
        ratio = media["width"] / media["height"]
        check(abs(ratio - 9 / 16) < 0.02,
              f"output is 9:16 ({media['width']}x{media['height']}, ratio={ratio:.4f})", failures)

    Path(args.evidence).write_text(json.dumps({
        "job_id": job_id,
        "status": view["status"],
        "metrics": view.get("metrics"),
        "result": view.get("result"),
        "probed_output": media,
        "checks_failed": failures,
    }, indent=2))

    if failures:
        print(f"\n[verify] FAILED: {len(failures)} check(s)")
        for item in failures:
            print(f"  - {item}")
        return 1
    print(f"\n[verify] all checks passed; evidence={args.evidence}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
