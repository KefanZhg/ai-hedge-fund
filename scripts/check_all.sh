#!/usr/bin/env bash
set -euo pipefail

echo "[check] Running repo checks"

if [[ -x ./scripts/lint.sh ]]; then
  ./scripts/lint.sh
else
  echo "[check] No scripts/lint.sh; skipping"
fi

if [[ -x ./scripts/test.sh ]]; then
  ./scripts/test.sh
else
  echo "[check] No scripts/test.sh; skipping"
fi

echo "[check] Complete"
