"""Offline static and simulated tests for non-secret Brev workstation assets."""
from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

import cv2
import pytest

ROOT = Path(__file__).resolve().parents[1]
BREV = ROOT / "brev"

SHELL_ASSETS = [
    "common/install-vcropper.sh",
    "common/paths.sh",
    "common/validate-runtime-config.sh",
    "common/validate-exposure.sh",
    "cpu-remote/deploy-and-verify.sh",
    "cpu-remote/setup.sh",
    "cpu-remote/vcropper-provider",
    "cpu-remote/vcropper-service",
    "cpu-remote/vcropper-tunnel",
    "cpu-remote/deploy-check.sh",
]


def _asset(path: str) -> Path:
    return BREV / path


def test_expected_brev_assets_exist():
    expected = [
        "README.md",
        "common/install-vcropper.sh",
        "common/paths.sh",
        "common/validate-runtime-config.sh",
        "common/validate-exposure.sh",
        "common/write-runtime-manifest.py",
        "common/create-smoke-video.py",
        "common/verify-service.py",
        "cpu-remote/deploy-and-verify.sh",
        "cpu-remote/setup.sh",
        "cpu-remote/vcropper-provider",
        "cpu-remote/vcropper-service",
        "cpu-remote/vcropper-tunnel",
        "cpu-remote/deploy-check.sh",
        "cpu-remote/DEPLOY.md",
        "cpu-remote/profile.env.example",
    ]
    assert [path for path in expected if not _asset(path).is_file()] == []


def test_no_local_model_runtime_profile_remains():
    """The GPU/NIM profile was removed: the model is always reached over an API."""
    assert not (BREV / "gpu-local-nim").exists()
    assert not (BREV / "provision-gpu-and-verify.sh").exists()


@pytest.mark.parametrize("path", SHELL_ASSETS)
def test_shell_assets_parse(path):
    subprocess.run(["bash", "-n", _asset(path)], check=True)


def test_cpu_assets_do_not_reference_local_model_runtime():
    """The profile runs the API in containers but never hosts the model itself.

    Docker Compose is expected here (it runs the API and Redis); what must stay absent is
    any local *model* runtime, which is the thing this profile deliberately does not do.
    """
    contents = "\n".join(_asset(path).read_text().lower() for path in [
        "cpu-remote/setup.sh",
        "cpu-remote/vcropper-service",
        "cpu-remote/vcropper-provider",
        "cpu-remote/profile.env.example",
    ])
    assert "nim_image" not in contents
    assert "nvidia-smi" not in contents
    assert "nvcr.io" not in contents
    assert "ngc_api_key" not in contents
    assert "runtime: nvidia" not in contents


def test_setup_verifies_docker_instead_of_installing_it():
    """Brev VM-Mode images preinstall Docker.

    Installing it here would need sudo and a docker-group membership the current
    non-interactive shell would not pick up until re-login.
    """
    setup = _asset("cpu-remote/setup.sh").read_text()
    assert "docker compose version" in setup
    assert "docker info" in setup
    assert "apt-get install" not in setup
    assert "get.docker.com" not in setup
    assert "usermod" not in setup


def test_setup_persists_launch_parameters_for_restart():
    """Brev supplies Launch parameters once; a restart must not lose the configuration."""
    setup = _asset("cpu-remote/setup.sh").read_text()
    assert "chmod 600" in setup
    assert "umask 077" in setup
    assert "CROPPER_API_TOKEN" in setup
    assert "Keeping existing" in setup
    # A generated token is unreachable without SSH, so the Launchable requires one.
    assert "CROPPER_API_TOKEN is required" in setup
    assert "token_urlsafe(32)" in setup, "the refusal should still tell the operator how to mint one"
    assert "CROPPER_ALLOWED_HOSTS=" in setup
    assert "CROPPER_ALLOW_PRIVATE_HOSTS=" in setup


def test_setup_derives_the_repo_root_instead_of_guessing_the_harness_path():
    """Brev clones a git source wherever it chooses; $HOME/workspace was the harness path."""
    setup = _asset("cpu-remote/setup.sh").read_text()
    assert "common/paths.sh" in setup
    assert "$HOME/workspace/v-cropper-cli" not in setup
    paths = _asset("common/paths.sh").read_text()
    assert 'dirname "${BASH_SOURCE[0]}"' in paths
    assert "$HOME/workspace/v-cropper-cli" not in paths


def test_setup_starts_the_service_and_waits_for_healthz():
    """A Launchable's setup script is the only thing that runs; the API must be listening."""
    setup = _asset("cpu-remote/setup.sh").read_text()
    assert "vcropper-service" in setup
    assert "/healthz" in setup
    assert "VCROPPER_SKIP_START" in setup


def test_service_wrapper_refuses_to_run_without_a_strong_token():
    wrapper = _asset("cpu-remote/vcropper-service").read_text()
    assert "CROPPER_API_TOKEN" in wrapper
    assert "-lt 24" in wrapper or "< 24" in wrapper


def test_compose_binds_loopback_and_requires_a_token():
    """A published 0.0.0.0 port on a Brev VM would expose the API to the internet."""
    compose = (ROOT / "compose.yaml").read_text()
    assert "${CROPPER_BIND_ADDRESS:-127.0.0.1}" in compose
    assert "CROPPER_API_TOKEN: ${CROPPER_API_TOKEN:?" in compose
    assert '"8090:8080"' not in compose
    assert re.search(r"^\s+-\s+\"0\.0\.0\.0:", compose, re.MULTILINE) is None


def test_minio_is_confined_to_the_verify_profile():
    """Object storage is a test fixture and must never start in a normal deployment."""
    compose = (ROOT / "compose.yaml").read_text()
    minio = compose.split("minio:", 1)[1]
    assert 'profiles: ["verify"]' in minio.split("volumes:")[0]
    # No committed credential: deploy-and-verify.sh generates one per run.
    assert "MINIO_ROOT_PASSWORD: ${MINIO_ROOT_PASSWORD:-}" in compose


def test_shared_installer_bootstraps_uv_in_an_isolated_venv():
    installer = _asset("common/install-vcropper.sh").read_text()
    assert 'if ! python3 -m venv "$bootstrap_dir"; then' in installer
    assert "sudo apt-get install -y python3-venv" in installer
    assert 'rm -rf "$bootstrap_dir"' in installer
    assert '"$bootstrap_dir/bin/pip" install --upgrade pip uv' in installer


@pytest.mark.parametrize("path", [
    "cpu-remote/setup.sh",
    "cpu-remote/vcropper-provider",
    "cpu-remote/deploy-and-verify.sh",
])
def test_remote_assets_add_user_local_bin_to_path(path):
    assert 'export PATH="$HOME/.local/bin:$PATH"' in _asset(path).read_text()


@pytest.mark.parametrize("path", ["cpu-remote/setup.sh"])
def test_setup_assets_invoke_shared_installer_through_bash(path):
    assert 'bash "$script_dir/../common/install-vcropper.sh"' in _asset(path).read_text()


def test_assets_do_not_enable_shell_trace_or_echo_secret_values():
    contents = "\n".join(_asset(path).read_text() for path in SHELL_ASSETS)
    assert "set -x" not in contents
    assert "echo \"$VCROPPER_API_KEY\"" not in contents
    assert "printenv" not in contents


def test_runtime_brev_env_profiles_are_gitignored():
    assert "brev/**/*.env" in (ROOT / ".gitignore").read_text()


def test_validate_runtime_config_never_echoes_secret(tmp_path):
    script = _asset("common/validate-runtime-config.sh")
    secret = "secret-that-must-not-appear"
    env = os.environ | {
        "VCROPPER_BASE_URL": "http://127.0.0.1:8000/v1",
        "VCROPPER_API_KEY": secret,
        "VCROPPER_MODEL": "local-test",
    }
    result = subprocess.run(["bash", script], env=env, capture_output=True, text=True, check=True)
    assert secret not in result.stdout
    assert secret not in result.stderr
    assert "Runtime config valid" in result.stdout


def test_validate_runtime_config_rejects_non_v1_endpoint_without_secret():
    script = _asset("common/validate-runtime-config.sh")
    secret = "secret-that-must-not-appear"
    env = os.environ | {
        "VCROPPER_BASE_URL": "http://127.0.0.1:8000",
        "VCROPPER_API_KEY": secret,
        "VCROPPER_MODEL": "local-test",
    }
    result = subprocess.run(["bash", script], env=env, capture_output=True, text=True)
    assert result.returncode != 0
    assert secret not in result.stdout
    assert secret not in result.stderr


class TestExposureGuard:
    """CROPPER_BIND_ADDRESS is the host publish address, so this layer knows the truth.

    The service itself cannot tell: inside a container it always binds 0.0.0.0 because
    Docker routes to it, which is why the authoritative check lives in the shell.
    """

    script = "common/validate-exposure.sh"

    def _run(self, **env_overrides):
        env = os.environ | {"CROPPER_ALLOW_PLAINTEXT": ""} | env_overrides
        return subprocess.run(
            ["bash", _asset(self.script)], env=env, capture_output=True, text=True)

    @pytest.mark.parametrize("bind", ["127.0.0.1", "::1", "localhost"])
    def test_loopback_publishing_is_allowed(self, bind):
        assert self._run(CROPPER_BIND_ADDRESS=bind).returncode == 0

    def test_the_default_is_loopback_when_unset(self):
        env = os.environ.copy()
        env.pop("CROPPER_BIND_ADDRESS", None)
        env["CROPPER_ALLOW_PLAINTEXT"] = ""
        result = subprocess.run(
            ["bash", _asset(self.script)], env=env, capture_output=True, text=True)
        assert result.returncode == 0

    @pytest.mark.parametrize("bind", ["0.0.0.0", "10.0.0.5", "::"])
    def test_routable_publishing_is_refused_without_the_opt_out(self, bind):
        result = self._run(CROPPER_BIND_ADDRESS=bind)
        assert result.returncode == 1
        assert "CROPPER_ALLOW_PLAINTEXT=1" in result.stderr
        assert "port-forward" in result.stderr, "the refusal should name the supported path"

    def test_the_opt_out_permits_publishing_but_warns(self):
        result = self._run(CROPPER_BIND_ADDRESS="0.0.0.0", CROPPER_ALLOW_PLAINTEXT="1")
        assert result.returncode == 0
        assert "Warning" in result.stderr

    def test_a_non_exact_opt_out_value_does_not_disable_the_guard(self):
        """Only '1' opts out, so a stray 'false' or 'no' cannot silently open the port."""
        for value in ["0", "false", "no", "true", "yes"]:
            result = self._run(CROPPER_BIND_ADDRESS="0.0.0.0", CROPPER_ALLOW_PLAINTEXT=value)
            assert result.returncode == (0 if value == "1" else 1), value

    def test_the_guard_never_echoes_the_api_token(self):
        secret = "token-that-must-not-appear"
        result = self._run(CROPPER_BIND_ADDRESS="0.0.0.0", CROPPER_API_TOKEN=secret)
        assert secret not in result.stdout
        assert secret not in result.stderr

    @pytest.mark.parametrize("wrapper", ["cpu-remote/vcropper-service", "cpu-remote/deploy-check.sh"])
    def test_both_service_wrappers_invoke_the_guard(self, wrapper):
        assert "validate-exposure.sh" in _asset(wrapper).read_text()

    def test_the_cli_wrapper_does_not_invoke_the_guard(self):
        """The CLI profile publishes no port, so the check would be noise there."""
        assert "validate-exposure.sh" not in _asset("cpu-remote/vcropper-provider").read_text()

    def test_compose_carries_the_exposure_declaration_into_the_container(self):
        """Without these the in-service guard is unreachable in a real deployment."""
        compose = (ROOT / "compose.yaml").read_text()
        assert "CROPPER_PUBLIC_ORIGIN: ${CROPPER_PUBLIC_ORIGIN:-}" in compose
        assert "CROPPER_ALLOW_PLAINTEXT: ${CROPPER_ALLOW_PLAINTEXT:-}" in compose


def test_runtime_manifest_is_non_secret_and_records_requested_metadata(tmp_path):
    script = _asset("common/write-runtime-manifest.py")
    out = tmp_path / "manifest.json"
    subprocess.run([
        "python", script, "--output", out, "--profile", "cpu-remote",
        "--base-url", "https://provider.example/v1", "--model", "vision",
        "--source-revision", "abc123",
    ], check=True)
    manifest = json.loads(out.read_text())
    assert manifest == {
        "base_url": "https://provider.example/v1",
        "model": "vision",
        "profile": "cpu-remote",
        "source_revision": "abc123",
    }


def test_smoke_video_generator(tmp_path):
    video = tmp_path / "smoke.mp4"
    subprocess.run([
        "python", _asset("common/create-smoke-video.py"), "--output", video,
    ], check=True)
    capture = cv2.VideoCapture(str(video))
    assert capture.isOpened()
    assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 30
    capture.release()


def _fake_command(bin_dir, name, body):
    command = bin_dir / name
    command.write_text(f"#!/usr/bin/env bash\nset -euo pipefail\n{body}\n")
    command.chmod(0o755)


class TestTunnel:
    """The API binds to the instance's loopback, so the tunnel is the only way in."""

    def test_requires_an_instance_name(self):
        result = subprocess.run(
            ["bash", _asset("cpu-remote/vcropper-tunnel")], capture_output=True, text=True)
        assert result.returncode == 2
        assert "usage:" in result.stderr

    def test_fails_fast_when_brev_is_unauthenticated(self, tmp_path):
        """An expired session must not become an infinite reconnect loop."""
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        _fake_command(bin_dir, "brev", 'if [[ "$1" == "ls" ]]; then exit 1; fi\nexit 0')
        env = os.environ | {
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "VCROPPER_BREV_BIN": str(bin_dir / "brev"),
        }
        result = subprocess.run(
            ["bash", _asset("cpu-remote/vcropper-tunnel"), "inst", "18099"],
            env=env, capture_output=True, text=True, timeout=30)
        assert result.returncode == 1
        assert "brev login" in result.stderr

    def test_reconnects_after_a_dropped_forward(self, tmp_path):
        """brev port-forward exits on any blip; a client mid-run must not be stranded."""
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        count = tmp_path / "attempts"
        _fake_command(bin_dir, "brev", f"""
if [[ "$1" == "ls" ]]; then exit 0; fi
if [[ "$1" == "port-forward" ]]; then
  n=$(cat "{count}" 2>/dev/null || echo 0)
  n=$((n + 1))
  echo "$n" > "{count}"
  if (( n < 3 )); then exit 7; fi
  exit 0
fi
exit 0
""")
        env = os.environ | {
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "VCROPPER_BREV_BIN": str(bin_dir / "brev"),
            "VCROPPER_TUNNEL_MAX_BACKOFF": "1",
        }
        result = subprocess.run(
            ["bash", _asset("cpu-remote/vcropper-tunnel"), "inst", "18098"],
            env=env, capture_output=True, text=True, timeout=60)
        assert result.returncode == 0
        assert count.read_text().strip() == "3", "should retry until the forward holds"
        assert "reconnecting" in result.stderr

    def test_does_not_log_the_api_token(self):
        script = _asset("cpu-remote/vcropper-tunnel").read_text()
        assert "echo \"$CROPPER_API_TOKEN\"" not in script
        assert "$CROPPER_API_TOKEN" in script, "it should still tell the operator to send one"


def test_cpu_wrapper_loads_private_profile_and_invokes_uv(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    profile = tmp_path / "provider.env"
    profile.write_text(
        "VCROPPER_BASE_URL=https://provider.example/v1\n"
        "VCROPPER_API_KEY=provider-key\n"
        "VCROPPER_MODEL=provider-model\n"
    )
    profile.chmod(0o600)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    capture = tmp_path / "capture.txt"
    _fake_command(
        bin_dir,
        "uv",
        'printf "%s|%s|%s|%s\\n" "$VCROPPER_BASE_URL" "$VCROPPER_MODEL" "$1" "$2" > "$CAPTURE"',
    )
    env = os.environ | {
        "VCROPPER_REPO_DIR": str(repo),
        "VCROPPER_PROFILE_FILE": str(profile),
        "CAPTURE": str(capture),
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
    }

    subprocess.run(["bash", _asset("cpu-remote/vcropper-provider"), "input.mp4"], env=env, check=True)
    assert capture.read_text().strip() == "https://provider.example/v1|provider-model|run|v-cropper"


class TestSetupLaunchable:
    """setup.sh is the Launchable entry point: persist params, start the API, stay silent."""

    token = "launchable-token-that-must-not-appear"
    api_key = "provider-key-that-must-not-appear"

    def _env(self, tmp_path, bin_dir, **overrides):
        repo = tmp_path / "repo"
        repo.mkdir(exist_ok=True)
        (repo / "compose.yaml").write_text("services: {}\n")
        profile = tmp_path / "provider.env"
        env = os.environ | {
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "VCROPPER_REPO_DIR": str(repo),
            "VCROPPER_PROFILE_FILE": str(profile),
            "XDG_CONFIG_HOME": str(tmp_path / "config"),
            "VCROPPER_SKIP_INSTALL": "1",
            "VCROPPER_SKIP_BUILD": "1",
            "VCROPPER_PROVIDER": "openai",
            "VCROPPER_BASE_URL": "https://integrate.api.nvidia.com/v1",
            "VCROPPER_API_KEY": self.api_key,
            "VCROPPER_MODEL": "nemotron-3-nano-omni-30b-a3b-reasoning",
            "CROPPER_API_TOKEN": self.token,
            "CROPPER_BIND_ADDRESS": "127.0.0.1",
            "CROPPER_PUBLISHED_PORT": "18090",
            "VCROPPER_READY_ATTEMPTS": "3",
        }
        env.update(overrides)
        return env, profile

    def _bin(self, tmp_path, *, healthz_ok=True):
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        calls = tmp_path / "calls.log"
        _fake_command(bin_dir, "docker", f"""
printf 'docker %s\\n' "$*" >> "{calls}"
case "$1" in
  compose)
    case "$2" in
      version) echo "Docker Compose version 2.29.0" ;;
      *) ;;
    esac
    ;;
  info) ;;
  *) ;;
esac
""")
        health = "echo '{{\"status\":\"ok\"}}'; exit 0" if healthz_ok else "exit 1"
        _fake_command(bin_dir, "curl", f"""
printf 'curl %s\\n' "$*" >> "{calls}"
{health}
""")
        return bin_dir, calls

    def test_writes_a_mode_600_profile_and_starts_the_service(self, tmp_path):
        bin_dir, calls = self._bin(tmp_path)
        env, profile = self._env(tmp_path, bin_dir)
        result = subprocess.run(
            ["bash", _asset("cpu-remote/setup.sh")],
            env=env, capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, result.stderr
        assert profile.is_file()
        assert (profile.stat().st_mode & 0o777) == 0o600
        text = profile.read_text()
        assert f"CROPPER_API_TOKEN={self.token}" in text
        assert f"VCROPPER_API_KEY={self.api_key}" in text
        assert "CROPPER_ALLOWED_HOSTS=" not in text
        assert "CROPPER_ALLOW_PRIVATE_HOSTS=" not in text
        assert "docker compose up" in calls.read_text()
        assert "/healthz" in calls.read_text()
        assert "setup complete" in result.stdout
        assert self.token not in result.stdout
        assert self.token not in result.stderr
        assert self.api_key not in result.stdout
        assert self.api_key not in result.stderr

    def test_persists_object_store_allowlist_when_supplied(self, tmp_path):
        """An S3 allowlist set as a Launch parameter must survive stop/start."""
        bin_dir, _ = self._bin(tmp_path)
        env, profile = self._env(
            tmp_path, bin_dir,
            CROPPER_ALLOWED_HOSTS="*.s3.us-west-2.amazonaws.com,*.amazonaws.com",
            CROPPER_ALLOW_PRIVATE_HOSTS="false",
            VCROPPER_SKIP_START="1",
        )
        result = subprocess.run(
            ["bash", _asset("cpu-remote/setup.sh")],
            env=env, capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, result.stderr
        text = profile.read_text()
        assert "CROPPER_ALLOWED_HOSTS=*.s3.us-west-2.amazonaws.com,*.amazonaws.com" in text
        assert "CROPPER_ALLOW_PRIVATE_HOSTS=false" in text

    def test_token_only_is_no_longer_a_silent_skip(self, tmp_path):
        """The old write was gated on a provider key, which left a VM with nothing listening."""
        bin_dir, _ = self._bin(tmp_path)
        env, profile = self._env(tmp_path, bin_dir, VCROPPER_API_KEY="")
        result = subprocess.run(
            ["bash", _asset("cpu-remote/setup.sh")],
            env=env, capture_output=True, text=True, timeout=30)
        assert result.returncode == 1
        assert "VCROPPER_API_KEY" in result.stderr
        assert not profile.exists(), "an incomplete profile must not be left behind"
        assert self.token not in result.stdout
        assert self.token not in result.stderr

    def test_missing_token_fails_without_generating_one(self, tmp_path):
        bin_dir, _ = self._bin(tmp_path)
        env, profile = self._env(tmp_path, bin_dir)
        env["CROPPER_API_TOKEN"] = ""
        result = subprocess.run(
            ["bash", _asset("cpu-remote/setup.sh")],
            env=env, capture_output=True, text=True, timeout=30)
        assert result.returncode == 1
        assert "CROPPER_API_TOKEN is required" in result.stderr
        assert not profile.exists()

    def test_a_second_run_does_not_clobber_the_profile(self, tmp_path):
        bin_dir, _ = self._bin(tmp_path)
        env, profile = self._env(tmp_path, bin_dir)
        first = subprocess.run(
            ["bash", _asset("cpu-remote/setup.sh")],
            env=env, capture_output=True, text=True, timeout=30)
        assert first.returncode == 0, first.stderr
        original = profile.read_text()
        env["CROPPER_API_TOKEN"] = "a-different-token-that-must-not-overwrite"
        env["VCROPPER_API_KEY"] = "a-different-key-that-must-not-overwrite"
        second = subprocess.run(
            ["bash", _asset("cpu-remote/setup.sh")],
            env=env, capture_output=True, text=True, timeout=30)
        assert second.returncode == 0, second.stderr
        assert "Keeping existing" in second.stdout
        assert profile.read_text() == original
        assert "a-different-token-that-must-not-overwrite" not in second.stdout
        assert "a-different-token-that-must-not-overwrite" not in second.stderr

    def test_derives_the_repo_root_when_the_harness_override_is_absent(self, tmp_path):
        bin_dir, _ = self._bin(tmp_path)
        env, _ = self._env(tmp_path, bin_dir)
        env.pop("VCROPPER_REPO_DIR")
        env["VCROPPER_SKIP_START"] = "1"
        env["VCROPPER_SKIP_BUILD"] = "1"
        result = subprocess.run(
            ["bash", _asset("cpu-remote/setup.sh")],
            env=env, capture_output=True, text=True, timeout=30)
        # Without the override, paths.sh points at this checkout. Skip start so we do
        # not actually compose-up against the real tree from the test process.
        assert result.returncode == 0, result.stderr

    def test_refuses_to_claim_ready_when_healthz_never_answers(self, tmp_path):
        bin_dir, _ = self._bin(tmp_path, healthz_ok=False)
        env, _ = self._env(tmp_path, bin_dir)
        result = subprocess.run(
            ["bash", _asset("cpu-remote/setup.sh")],
            env=env, capture_output=True, text=True, timeout=90)
        assert result.returncode == 1
        assert "did not become ready" in result.stderr
        assert self.token not in result.stdout
        assert self.token not in result.stderr


