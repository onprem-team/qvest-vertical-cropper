#!/usr/bin/env bash
# Install the checked-out project on a Brev workstation. Never pass or print secrets here.
set -euo pipefail

# shellcheck source=paths.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/paths.sh"
repo_dir="$VCROPPER_REPO_DIR"

if [[ ! -f "$repo_dir/pyproject.toml" ]]; then
  echo "v-cropper checkout not found at the configured VCROPPER_REPO_DIR" >&2
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  bootstrap_dir="$HOME/.local/share/vcropper-bootstrap"
  if ! python3 -m venv "$bootstrap_dir"; then
    rm -rf "$bootstrap_dir"
    sudo apt-get update
    sudo apt-get install -y python3-venv
    python3 -m venv "$bootstrap_dir"
  fi
  "$bootstrap_dir/bin/pip" install --upgrade pip uv
  mkdir -p "$HOME/.local/bin"
  ln -sf "$bootstrap_dir/bin/uv" "$HOME/.local/bin/uv"
  export PATH="$HOME/.local/bin:$PATH"
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
  sudo apt-get update
  sudo apt-get install -y ffmpeg
fi

(
  cd "$repo_dir"
  uv sync --frozen
)

echo "v-cropper dependencies installed"
