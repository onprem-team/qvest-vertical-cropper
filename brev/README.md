# Brev workstation assets

These assets deploy the v-cropper HTTP API on a NVIDIA Brev CPU instance. The instance
runs no model: it calls a hosted VLM over an API. Launch from this public repository:

<https://github.com/onprem-team/qvest-vertical-cropper>

| Profile | Model path | Default network posture |
|---|---|---|
| `cpu-remote` | Hosted or custom OpenAI-compatible VLM, or AWS Bedrock | Loopback only; SSH tunnel (`ssh -N -L` or `brev port-forward`) |

There is no local-model profile. v-cropper always reaches its model over an API, whether
that is NVIDIA-hosted, a gateway/router, Bedrock, or any OpenAI-compatible endpoint.

## Prerequisites before use

- A Brev organization and a Launchable that clones this public git URL (VM Mode).
  Paste a shebang setup script that `cd`s into the clone and runs
  `brev/cpu-remote/setup.sh` (see [cpu-remote/DEPLOY.md](cpu-remote/DEPLOY.md)).
- A scoped provider credential with an explicit spend limit.
- Launch parameters for `CROPPER_API_TOKEN` (required) and the VLM key (`VCROPPER_API_KEY`
  for OpenAI-compatible providers). Bedrock needs `VCROPPER_PROVIDER=bedrock` and
  `BEDROCK_REGION`/`AWS_REGION` instead of `VCROPPER_API_KEY`. Prefer Brev secrets; the
  secret *name* does not have to match the parameter name. Leave `VCROPPER_MODEL` /
  `VCROPPER_BASE_URL` / `CROPPER_ALLOWED_HOSTS` optional.

## CPU Remote-VLM

On first boot, `cpu-remote/setup.sh` writes `~/.config/v-cropper/provider.env` from Brev
Launch parameters and starts the API. The profile file is local and gitignored. Launch
parameters are supplied once at launch and are **not** re-supplied to later Compose runs.
**Only values written into `provider.env` survive a stop/start** (provider fields, the
bootstrap token, and `CROPPER_ALLOWED_HOSTS` / `CROPPER_ALLOW_PRIVATE_HOSTS` when supplied).

To restart by hand after the first boot:

```bash
chmod 600 ~/.config/v-cropper/provider.env
brev/cpu-remote/vcropper-service up
```

See [cpu-remote/DEPLOY.md](cpu-remote/DEPLOY.md) for Launchable setup, verification, and
the caller-integration walkthrough.

## Runtime manifest

`common/write-runtime-manifest.py` records only non-secret provenance: profile, endpoint,
model, and source revision. Never add provider credentials to a runtime manifest.
