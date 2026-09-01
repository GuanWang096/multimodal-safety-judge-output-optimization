#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 qwen|safework|minicpm" >&2
  exit 2
fi

export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="/root/src${PYTHONPATH:+:${PYTHONPATH}}"

SRC_ROOT="/root/src"
DATASET_ROOT="/root/datasets"
RESULT_ROOT="/root/result/mmds/qualification"
COMMITTEE="${RESULT_ROOT}/committee_v2.json"
POLICY="${SRC_ROOT}/src/multijudge_workflows/configs/canonical_policy_v1.json"

case "$1" in
  qwen)
    ADAPTER="qwen"
    ADAPTER_VERSION="qwen3-vl-soft-score-v1"
    MODEL_PATH="/root/autodl-tmp/model/Qwen3-VL-8B-Instruct"
    REVISION="5d854aab08710c16b980ec6d603d863b3821b915"
    QUALIFICATION_CONFIG="${SRC_ROOT}/src/multijudge_workflows/configs/soft_score_qualification_v1.json"
    ;;
  safework)
    ADAPTER="safework"
    ADAPTER_VERSION="safework-rm-safety-native-sequence-sum-v3"
    MODEL_PATH="/root/autodl-tmp/model/SafeWork-RM-Safety-7B"
    REVISION="be345f29425fe94586c0598785a143703bbbc4fc"
    QUALIFICATION_CONFIG="${SRC_ROOT}/src/multijudge_workflows/configs/soft_score_qualification_v3.json"
    ;;
  minicpm)
    ADAPTER="minicpm"
    ADAPTER_VERSION="minicpm-v4.5-soft-score-v1"
    MODEL_PATH="/root/autodl-tmp/model/MiniCPM-V-4_5"
    REVISION="2626e837a54905aab70fae9325153ef3454387ab"
    QUALIFICATION_CONFIG="${SRC_ROOT}/src/multijudge_workflows/configs/soft_score_qualification_v2.json"
    ;;
  *)
    echo "Unknown or unqualified soft adapter: $1" >&2
    exit 2
    ;;
esac

JUDGE_ROOT="${RESULT_ROOT}/${ADAPTER}"
FINGERPRINT="${JUDGE_ROOT}/audit/model_fingerprint.json"

for path in \
  "${SRC_ROOT}/src/multijudge_workflows/run_soft_scores.py" \
  "${SRC_ROOT}/src/multijudge_workflows/audit_soft_development.py" \
  "${SRC_ROOT}/src/multijudge_workflows/configs/softfusion_fit.jsonl" \
  "${SRC_ROOT}/src/multijudge_workflows/configs/softfusion_select.jsonl" \
  "$POLICY" \
  "$QUALIFICATION_CONFIG" \
  "$FINGERPRINT" \
  "$COMMITTEE"; do
  if [[ ! -f "$path" ]]; then
    echo "Missing required file: $path" >&2
    exit 1
  fi
done

python -c "import json; p='${COMMITTEE}'; d=json.load(open(p)); allowed={x['judge']:x['adapter_version'] for x in d['qualified_soft_judges']}; assert allowed.get('${ADAPTER}')=='${ADAPTER_VERSION}', allowed"

cd "$SRC_ROOT"
for split in fit select; do
  MANIFEST="${SRC_ROOT}/src/multijudge_workflows/configs/softfusion_${split}.jsonl"
  RUN_DIR="${JUDGE_ROOT}/${split}"
  python -u src/multijudge_workflows/run_soft_scores.py \
    --adapter "$ADAPTER" \
    --dataset-root "$DATASET_ROOT" \
    --manifest "$MANIFEST" \
    --policy "$POLICY" \
    --model-path "$MODEL_PATH" \
    --model-revision "$REVISION" \
    --model-fingerprint "$FINGERPRINT" \
    --qualification-config "$QUALIFICATION_CONFIG" \
    --run-dir "$RUN_DIR" \
    --attn-implementation sdpa \
    --seed 20260729

  python -u src/multijudge_workflows/audit_soft_development.py \
    --manifest "$MANIFEST" \
    --run-dir "$RUN_DIR" \
    --judge "$ADAPTER" \
    --adapter-version "$ADAPTER_VERSION" \
    --output "${RUN_DIR}/coverage_audit.json"
done

echo "${ADAPTER} fit+select soft scores completed successfully."
