# Security Policy

## Reporting a Vulnerability

Please **do not** open a public issue for security vulnerabilities.

Instead, report them privately with GitHub **private vulnerability reporting**:
[Security → Report a vulnerability](https://github.com/onprem-team/qvest-vertical-cropper/security/advisories/new).
Include a description, reproduction steps, and the affected version/commit if known.

We will acknowledge your report, investigate, and keep you informed of the resolution. Please
allow a reasonable time for a fix before any public disclosure.

## Supported Versions

This project is distributed from source/git (no released versions). Security fixes are applied
to the `main` branch.

## Scope notes

- The tool calls a **remotely hosted** vision model via an API key. Treat your API key as a
  secret: prefer `.env` (gitignored) or environment variables, never commit keys.
- Frames are sent to whichever provider you configure; review that provider's data-handling
  policy for sensitive footage.
- The HTTP job API requires `CROPPER_API_TOKEN` (or a minted key) on every `/v1` route.
  Treat those like secrets. `/healthz`, `/readyz`, and `/version` are open by design.
- Job `source_url` / `destination_url` are fetched and uploaded by the service. Keep
  `CROPPER_ALLOWED_HOSTS` tight; an open allowlist turns the service into an SSRF proxy.
- The default bind is loopback. Do not publish a TCP port or a Brev Secure Link for the
  API. Reach a Brev VM over SSH (`ssh -N -L` or `brev port-forward`).
