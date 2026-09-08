#!/usr/bin/env bash
# Validate a loaded v-cropper provider profile without echoing credential values.
set -euo pipefail

provider="${VCROPPER_PROVIDER:-openai}"

case "$provider" in
  bedrock)
    # boto3 resolves credentials from its standard chain; only the region is our concern.
    if [[ -z "${BEDROCK_REGION:-}" && -z "${AWS_REGION:-}" ]]; then
      echo "AWS Bedrock requires BEDROCK_REGION or AWS_REGION" >&2
      exit 1
    fi
    ;;
  openai)
    : "${VCROPPER_BASE_URL:?VCROPPER_BASE_URL is required}"
    : "${VCROPPER_API_KEY:?VCROPPER_API_KEY is required}"

    case "$VCROPPER_BASE_URL" in
      http://*|https://*) ;;
      *)
        echo "VCROPPER_BASE_URL must be an absolute http(s) URL" >&2
        exit 1
        ;;
    esac

    if [[ "$VCROPPER_BASE_URL" != *"/v1"* ]]; then
      echo "VCROPPER_BASE_URL must contain /v1 for an OpenAI-compatible endpoint" >&2
      exit 1
    fi

    case "$VCROPPER_BASE_URL" in
      *@*)
        echo "VCROPPER_BASE_URL must not embed credentials; use VCROPPER_API_KEY" >&2
        exit 1
        ;;
    esac

    if [[ -z "${VCROPPER_MODEL:-}" ]]; then
      echo "Note: VCROPPER_MODEL is unset; the provider default will be used"
    fi
    ;;
  *)
    echo "VCROPPER_PROVIDER must be 'openai' or 'bedrock'" >&2
    exit 1
    ;;
esac

echo "Runtime config valid"
