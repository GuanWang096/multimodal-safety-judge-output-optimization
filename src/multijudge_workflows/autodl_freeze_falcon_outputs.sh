#!/usr/bin/env bash
set -euo pipefail

cd /root/src
export PYTHONPATH="/root/src${PYTHONPATH:+:${PYTHONPATH}}"

python -u src/multijudge_workflows/freeze_falcon_outputs.py \
  --result-root /root/result/external/falcon_test/raw \
  --output /root/result/external/falcon_test/raw_output_freeze.json

echo "Falcon-test raw outputs frozen before adaptation-label access."
