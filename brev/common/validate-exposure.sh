#!/usr/bin/env bash
# Refuse to publish the API on a non-loopback address without TLS in front of it.
#
# Callers authenticate with a bearer token, so publishing plain HTTP on a routable address
# hands that token to anything on the network path. This check lives here rather than in
# the service because the service cannot see its own exposure: inside a container it always
# binds 0.0.0.0 (Docker requires it), while CROPPER_BIND_ADDRESS below is the host publish
# address that actually governs reachability. This is the layer that knows the truth.
#
# Only used by the service wrappers. The CLI profile publishes no port and does not call it.
set -euo pipefail

bind="${CROPPER_BIND_ADDRESS:-127.0.0.1}"

case "$bind" in
  127.0.0.1|::1|localhost) exit 0 ;;
esac

if [[ "${CROPPER_ALLOW_PLAINTEXT:-}" == "1" ]]; then
  echo "Warning: publishing on $bind with CROPPER_ALLOW_PLAINTEXT=1." >&2
  echo "         Caller credentials are readable by anything on the network path." >&2
  exit 0
fi

cat >&2 <<EOF
CROPPER_BIND_ADDRESS is $bind, which publishes the API beyond this machine.

The API authenticates with a bearer token sent in a header. Published over plain HTTP,
that token is readable by anything on the network path.

Choose one:
  - Leave CROPPER_BIND_ADDRESS unset (or 127.0.0.1) and reach the API over
    'brev port-forward' or 'ssh -N -L', which tunnels it over SSH. This is the shipped default.
  - Put a TLS-terminating reverse proxy in front and publish to loopback only.
  - Set CROPPER_ALLOW_PLAINTEXT=1 to accept the risk deliberately.
EOF
exit 1
