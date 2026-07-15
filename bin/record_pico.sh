#!/usr/bin/env bash
# Convenience wrapper; all options are owned by the unified recorder.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source .venv/bin/activate
fi

exec handumi-record --device pico "$@"
