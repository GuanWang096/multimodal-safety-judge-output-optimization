#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for adapter in qwen safework internvl minicpm; do
  echo "===== Starting ${adapter} soft-score qualification ====="
  "${SCRIPT_DIR}/autodl_run_soft_scores.sh" "$adapter"
  echo "===== ${adapter} qualification passed ====="
done

echo "All four soft-score qualifications completed successfully."
