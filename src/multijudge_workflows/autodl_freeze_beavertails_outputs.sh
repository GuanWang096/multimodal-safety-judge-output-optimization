#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="/root/src${PYTHONPATH:+:${PYTHONPATH}}"
SRC_ROOT="/root/src"
EXPERIMENT_ROOT="${SRC_ROOT}/src/multijudge_workflows"
BASE="/root/result/external/beavertails_v"

cd "$SRC_ROOT"
python -u src/multijudge_workflows/freeze_beavertails_outputs.py \
  --manifest "${EXPERIMENT_ROOT}/configs/beavertails_v_external_unlabeled.jsonl" \
  --result-root "${BASE}/raw" \
  --analysis-plan "${BASE}/analysis_plan.json" \
  --output "${BASE}/raw_output_freeze.json"

ARCHIVE="/root/beavertails-v-external-raw-20260802.tar.gz"
rm -f "$ARCHIVE"
tar -czf "$ARCHIVE" \
  -C /root/result/external \
  beavertails_v
ls -lh "$ARCHIVE"
