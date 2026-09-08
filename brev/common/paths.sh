# Shared path resolution for Brev scripts. Source this; do not execute it.
#
# Brev clones a Launchable's git source wherever it chooses. Derive the repo
# root from this file: brev/common/ is two levels below the repo root.
# VCROPPER_REPO_DIR remains an explicit override.
if [[ -z "${VCROPPER_REPO_DIR:-}" ]]; then
  VCROPPER_REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
export VCROPPER_REPO_DIR
