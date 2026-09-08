"""Regression tests for the reviewed source archive used by disposable Brev VMs."""
from __future__ import annotations

import io
import subprocess
import tarfile


def _archive_files():
    archive = subprocess.check_output(["git", "archive", "--format=tar", "HEAD"])
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        return {member.name for member in tar.getmembers() if member.isfile()}


def test_git_archive_matches_tracked_head_and_contains_deployment_assets():
    archived = _archive_files()
    tracked = set(subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", "HEAD"],
        text=True,
    ).splitlines())
    assert archived == tracked
    assert "brev/cpu-remote/deploy-and-verify.sh" in archived
    # The service is built from the archive on the VM, so its build context must ship.
    assert {"Dockerfile", "compose.yaml"} <= archived


def test_git_archive_excludes_runtime_secrets_and_derived_artifacts():
    archived = _archive_files()
    assert ".env" not in archived
    assert not any(path.startswith("brev/") and path.endswith(".env") for path in archived)
    assert not any(path.startswith(prefix) for path in archived for prefix in (
        "eval/data/", "eval/results/", "renders/",
    ))
    assert not any(path.endswith(("_vertical.mp4", "_debug.mp4")) for path in archived)
