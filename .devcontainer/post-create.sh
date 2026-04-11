#!/usr/bin/env bash
set -euo pipefail

cd /workspaces/diarizen-service

if [[ -f requirements-dev.txt ]]; then
  uv pip install --system -r requirements-dev.txt
fi
