# Contributing to v-cropper-cli

Thanks for your interest in contributing! This project is a CPU-only, embeddable sports-cropper
library, CLI, and optional FastAPI job service built on any OpenAI-compatible vision model
(or native AWS Bedrock).

## Development setup

Requires [uv](https://docs.astral.sh/uv/) and `ffmpeg` on PATH (for H.264 output).

```bash
uv sync                 # install deps (incl. dev group)
uv run v-cropper --help # run the CLI against the working tree
```

## Tests & linting

All changes must keep the suite green and the tree lint-clean:

```bash
uv run pytest           # unit + integration tests (coverage is measured, not gated)
uv run ruff check .     # lint
```

- `live`-marked tests hit real provider APIs and are opt-in (`-m live`, needs a key); they do
  not run by default.
- Please add or update tests for any behavior change. The project targets high line + branch
  coverage.

## Pull requests

1. Fork and branch from `main`.
2. Keep changes focused; update `README.md`/docstrings when behavior changes.
3. Ensure `uv run pytest` and `uv run ruff check .` pass locally.
4. Open a PR and fill out the template.

## Developer Certificate of Origin (DCO)

This project uses the [DCO](https://developercertificate.org/). By contributing, you certify
that you wrote the change (or have the right to submit it) under the project's Apache-2.0
license. **Sign off every commit** with `-s`:

```bash
git commit -s -m "your message"
```

This appends a `Signed-off-by: Your Name <you@example.com>` line using your `git config`
`user.name` / `user.email`. Unsigned commits will be asked to amend before merge.

## Code style

- Python ≥ 3.10, formatted to the `ruff` config in `pyproject.toml` (line length 120).
- Prefer clear, modular, DRY code; docstrings explain intent, not the obvious.

## License

By contributing, you agree that your contributions are licensed under the
[Apache License 2.0](LICENSE).
