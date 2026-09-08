#!/usr/bin/env bash
# CPU Remote-VLM setup for a Brev VM-Mode instance.
#
# This profile never runs a model locally: it starts the v-cropper API container plus Redis
# and reaches the VLM over an API. Brev VM-Mode images ship Docker and the Compose plugin
# already, so this script verifies them rather than installing them — installing Docker here
# would need sudo, a docker-group membership that the current non-interactive shell would
# not pick up until re-login, and would risk fighting the preinstalled daemon.
#
# On a Launchable this is the only command that runs. It must leave a listening, authenticated
# API behind; building the image and exiting used to, which is why the service is started here.
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common/paths.sh
source "$script_dir/../common/paths.sh"
repo_dir="$VCROPPER_REPO_DIR"
config_dir="${XDG_CONFIG_HOME:-$HOME/.config}/v-cropper"
profile_file="${VCROPPER_PROFILE_FILE:-$config_dir/provider.env}"
export VCROPPER_PROFILE_FILE="$profile_file"

fail() {
  echo "$1" >&2
  exit 1
}

# --- container runtime -------------------------------------------------------------
command -v docker >/dev/null 2>&1 \
  || fail "docker not found. Use a Brev VM-Mode container image that preinstalls Docker."
docker compose version >/dev/null 2>&1 \
  || fail "the docker compose plugin is not available. Use a Brev image that preinstalls it."
docker info >/dev/null 2>&1 \
  || fail "the Docker daemon is not reachable by $(id -un). Check the instance is fully booted."

# --- CLI (used by the eval harness and the smoke video generator) --------------------
if [[ "${VCROPPER_SKIP_INSTALL:-0}" != "1" ]]; then
  bash "$script_dir/../common/install-vcropper.sh"
fi

mkdir -p "$config_dir"
chmod 700 "$config_dir"

# --- profile ------------------------------------------------------------------------
# Brev Launch parameters arrive as environment variables on the first boot only; they are
# not re-supplied to later container runs or restarts. Persist them once, with restrictive
# permissions, so `vcropper-service up` keeps working after a stop/start cycle.
#
# CROPPER_API_TOKEN is required: generating one here would leave a credential the deployer
# cannot retrieve without SSH, which breaks the one-click Launchable. A missing provider
# key used to skip this write entirely and leave a silent, unconfigured VM.
if [[ ! -f "$profile_file" ]]; then
  if [[ -z "${CROPPER_API_TOKEN:-}" ]]; then
    fail "CROPPER_API_TOKEN is required as a Launch parameter (prefer a saved secret)."
  fi
  if (( ${#CROPPER_API_TOKEN} < 24 )); then
    fail "CROPPER_API_TOKEN is shorter than 24 characters; generate one with: python3 -c 'import secrets; print(secrets.token_urlsafe(32))'"
  fi

  provider="${VCROPPER_PROVIDER:-openai}"
  case "$provider" in
    bedrock)
      if [[ -z "${BEDROCK_REGION:-}" && -z "${AWS_REGION:-}" ]]; then
        fail "AWS Bedrock requires BEDROCK_REGION or AWS_REGION as a Launch parameter."
      fi
      ;;
    openai)
      if [[ -z "${VCROPPER_API_KEY:-}" ]]; then
        fail "VCROPPER_API_KEY is required as a Launch parameter (prefer a saved secret)."
      fi
      ;;
    *)
      fail "VCROPPER_PROVIDER must be openai or bedrock."
      ;;
  esac

  umask 077
  {
    echo "# Written by brev/cpu-remote/setup.sh from Brev Launch parameters."
    echo "# Values are secret. Do not commit this file or copy it off the instance."
    echo "VCROPPER_PROVIDER=${provider}"
    echo "VCROPPER_BASE_URL=${VCROPPER_BASE_URL:-https://integrate.api.nvidia.com/v1}"
    echo "VCROPPER_API_KEY=${VCROPPER_API_KEY:-}"
    echo "VCROPPER_MODEL=${VCROPPER_MODEL:-}"
    echo "BEDROCK_REGION=${BEDROCK_REGION:-}"
    echo "AWS_REGION=${AWS_REGION:-}"
    echo "BEDROCK_VLM_MODEL_ID=${BEDROCK_VLM_MODEL_ID:-}"
    echo "CROPPER_API_TOKEN=${CROPPER_API_TOKEN}"
    # Optional object-store allowlist. Omitted unless supplied: Compose then uses its
    # defaults (minio,host.docker.internal / private hosts true). An S3 Launch parameter
    # must be written here or stop/start drops it and jobs 422.
    if [[ -n "${CROPPER_ALLOWED_HOSTS:-}" ]]; then
      echo "CROPPER_ALLOWED_HOSTS=${CROPPER_ALLOWED_HOSTS}"
    fi
    if [[ -n "${CROPPER_ALLOW_PRIVATE_HOSTS:-}" ]]; then
      echo "CROPPER_ALLOW_PRIVATE_HOSTS=${CROPPER_ALLOW_PRIVATE_HOSTS}"
    fi
  } >"$profile_file"
  chmod 600 "$profile_file"
  echo "Wrote $profile_file from launch parameters (mode 600)"
else
  echo "Keeping existing $profile_file"
fi

# --- image --------------------------------------------------------------------------
# Build now so the first `vcropper-service up` is fast and any build failure surfaces
# during provisioning rather than at first use.
if [[ -f "$profile_file" && "${VCROPPER_SKIP_BUILD:-0}" != "1" ]]; then
  ( cd "$repo_dir" && set -a && . "$profile_file" && set +a && docker compose build cropper )
fi

# --- service ------------------------------------------------------------------------
# A Launchable's setup script is the only thing that runs. Leaving the image built and
# the process not started used to give the deployer a VM with nothing listening.
if [[ -f "$profile_file" && "${VCROPPER_SKIP_START:-0}" != "1" ]]; then
  bash "$script_dir/vcropper-service" up
  bind="${CROPPER_BIND_ADDRESS:-127.0.0.1}"
  port="${CROPPER_PUBLISHED_PORT:-8090}"
  ready=0
  attempts="${VCROPPER_READY_ATTEMPTS:-60}"
  for _ in $(seq 1 "$attempts"); do
    if curl -fsS "http://${bind}:${port}/healthz" >/dev/null 2>&1; then
      ready=1
      break
    fi
    sleep 1
  done
  if (( ready != 1 )); then
    fail "v-cropper API did not become ready on ${bind}:${port} within ${attempts}s"
  fi
  echo "v-cropper API is ready on ${bind}:${port}"
fi

echo "CPU Remote-VLM workstation setup complete"
