#!/usr/bin/env bash
# Operator-run configuration check for a CPU Remote-VLM instance.
# Validates the profile and container runtime without starting the stack or spending
# anything at the VLM provider.
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common/paths.sh
source "$script_dir/../common/paths.sh"
repo_dir="$VCROPPER_REPO_DIR"
profile_file="${VCROPPER_PROFILE_FILE:-${XDG_CONFIG_HOME:-$HOME/.config}/v-cropper/provider.env}"

if [[ ! -f "$profile_file" ]]; then
  echo "Provider profile file is missing: $profile_file" >&2
  exit 1
fi

if mode="$(stat -c '%a' "$profile_file" 2>/dev/null)"; then
  :
else
  mode="$(stat -f '%Lp' "$profile_file")"
fi
if (( (8#$mode & 077) != 0 )); then
  echo "Provider profile file must not be readable by group or other users" >&2
  exit 1
fi

set -a
# shellcheck disable=SC1090
source "$profile_file"
set +a

bash "$script_dir/../common/validate-runtime-config.sh"
bash "$script_dir/../common/validate-exposure.sh"

: "${CROPPER_API_TOKEN:?CROPPER_API_TOKEN is required}"

docker compose version >/dev/null
( cd "$repo_dir" && docker compose config --quiet )

echo "CPU Remote-VLM deployment configuration check passed"
