#!/usr/bin/env bash
# Push local main to https://github.com/NarayanSiddhi/EchoNet_EF_Prediction
# Requires a Personal Access Token (classic) with "repo" scope:
#   https://github.com/settings/tokens
#
# Usage (recommended — token not stored in repo):
#   export GITHUB_TOKEN='ghp_xxxxxxxx'
#   bash scripts/push_echonet_ef.sh

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

TOKEN="${GITHUB_TOKEN:-${GH_TOKEN:-}}"
if [[ -z "$TOKEN" ]]; then
  echo "ERROR: Set GITHUB_TOKEN (or GH_TOKEN) to a GitHub PAT with 'repo' scope."
  echo "  export GITHUB_TOKEN='ghp_...'"
  echo "  bash scripts/push_echonet_ef.sh"
  exit 1
fi

REPO_URL="https://x-access-token:${TOKEN}@github.com/NarayanSiddhi/EchoNet_EF_Prediction.git"
export GIT_TERMINAL_PROMPT=0

echo "Pushing main → NarayanSiddhi/EchoNet_EF_Prediction ..."
git push -u "$REPO_URL" main:main

echo "Updating remote echonet_ef to plain HTTPS (no token in saved URL)..."
git remote set-url echonet_ef https://github.com/NarayanSiddhi/EchoNet_EF_Prediction.git
echo "Done. Next pushes: git push -u echonet_ef main   (after gh auth login or SSH)"
