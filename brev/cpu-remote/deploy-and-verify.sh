#!/usr/bin/env bash
# End-to-end deployment smoke for the CPU Remote-VLM profile.
#
# Brings up the API, Redis, and a throwaway MinIO, then drives a real job through the
# deployed HTTP surface: presigned GET in, presigned PUT out, bearer auth enforced, and the
# resulting object probed. This exercises the container the operator actually runs rather
# than the library in-process.
#
# Requires a configured provider profile: the job makes real VLM calls.
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common/paths.sh
source "$script_dir/../common/paths.sh"
repo_dir="$VCROPPER_REPO_DIR"
work_dir="${VCROPPER_DEPLOY_TEST_DIR:-/tmp/vcropper-cpu-deploy-test}"
profile_file="${VCROPPER_PROFILE_FILE:-${XDG_CONFIG_HOME:-$HOME/.config}/v-cropper/provider.env}"
port="${CROPPER_PUBLISHED_PORT:-8090}"
minio_port="${MINIO_PUBLISHED_PORT:-9000}"
keep_up="${VCROPPER_KEEP_STACK:-0}"

if [[ ! -f "$profile_file" ]]; then
  echo "Provider profile file is missing: $profile_file" >&2
  exit 1
fi

set -a
# shellcheck disable=SC1090
source "$profile_file"
set +a

: "${CROPPER_API_TOKEN:?CROPPER_API_TOKEN must be set in the provider profile}"

# Generated per run so no MinIO credential is ever committed or reused.
export MINIO_ROOT_USER="${MINIO_ROOT_USER:-vcropper}"
export MINIO_ROOT_PASSWORD="${MINIO_ROOT_PASSWORD:-$(python3 -c 'import secrets; print(secrets.token_urlsafe(24))')}"
export CROPPER_PUBLISHED_PORT="$port"
export MINIO_PUBLISHED_PORT="$minio_port"
# The service resolves the source/destination hosts itself; allow the compose-internal name.
export CROPPER_ALLOWED_HOSTS="${CROPPER_ALLOWED_HOSTS:-minio,host.docker.internal}"
export CROPPER_ALLOW_PRIVATE_HOSTS="${CROPPER_ALLOW_PRIVATE_HOSTS:-true}"

rm -rf "$work_dir"
mkdir -p "$work_dir"

cd "$repo_dir"

cleanup() {
  if [[ "$keep_up" == "1" ]]; then
    echo "VCROPPER_KEEP_STACK=1; leaving the stack running"
    return
  fi
  docker compose --profile verify down --volumes --remove-orphans >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "[deploy] building and starting the verify stack"
docker compose --profile verify up -d --build --wait

echo "[deploy] generating a smoke clip"
uv run python "$script_dir/../common/create-smoke-video.py" --output "$work_dir/smoke.mp4"

echo "[deploy] driving a job through the deployed API"
uv run python "$script_dir/../common/verify-service.py" \
  --base-url "http://127.0.0.1:$port" \
  --token "$CROPPER_API_TOKEN" \
  --storage-endpoint "http://127.0.0.1:$minio_port" \
  --storage-internal-endpoint "http://minio:9000" \
  --storage-key "$MINIO_ROOT_USER" \
  --storage-secret "$MINIO_ROOT_PASSWORD" \
  --video "$work_dir/smoke.mp4" \
  --evidence "$work_dir/verify-evidence.json"

uv run python "$script_dir/../common/write-runtime-manifest.py" \
  --output "$work_dir/runtime-manifest.json" \
  --profile "cpu-remote" \
  --base-url "${VCROPPER_BASE_URL:-bedrock:${BEDROCK_REGION:-${AWS_REGION:-unset}}}" \
  --model "${VCROPPER_MODEL:-provider-default}" \
  --source-revision "$(cat "$repo_dir/.vcropper-source-revision" 2>/dev/null || echo unknown)"

echo "CPU Remote-VLM deploy-and-verify passed: evidence=$work_dir"
