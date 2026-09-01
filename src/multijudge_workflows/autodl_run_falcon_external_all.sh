#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || ( "$1" != "smoke" && "$1" != "formal" ) ]]; then
  echo "Usage: $0 smoke|formal" >&2
  exit 2
fi

PHASE="$1"
cd /root/src
for judge in qwen safework minicpm; do
  echo "===== Starting Falcon-test ${judge} ${PHASE} ====="
  ./src/multijudge_workflows/autodl_run_falcon_external.sh "$judge" "$PHASE"
done
echo "Falcon-test external ${PHASE} runs completed successfully."
