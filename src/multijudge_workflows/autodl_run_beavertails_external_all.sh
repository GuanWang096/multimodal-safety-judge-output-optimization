#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || ( "$1" != "smoke" && "$1" != "formal" ) ]]; then
  echo "Usage: $0 smoke|formal" >&2
  exit 2
fi

PHASE="$1"
cd /root/src
for judge in qwen safework minicpm; do
  echo "===== Starting BeaverTails-V ${judge} ${PHASE} ====="
  ./src/multijudge_workflows/autodl_run_beavertails_external.sh "$judge" "$PHASE"
done
echo "BeaverTails-V external ${PHASE} runs completed successfully."
