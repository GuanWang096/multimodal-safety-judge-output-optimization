#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 qwen|safework|internvl|minicpm" >&2
  exit 2
fi

export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="/root/src${PYTHONPATH:+:${PYTHONPATH}}"

SRC_ROOT="/root/src"
DATASET_ROOT="/root/datasets"
MANIFEST="${SRC_ROOT}/src/multijudge_workflows/configs/mmds_smoke_100.json"
POLICY="${SRC_ROOT}/src/multijudge_workflows/configs/canonical_policy_v1.json"
case "$1" in
  qwen)
    ADAPTER="qwen"
    MODEL_PATH="/root/autodl-tmp/model/Qwen3-VL-8B-Instruct"
    REVISION="5d854aab08710c16b980ec6d603d863b3821b915"
    ADAPTER_FILE="${SRC_ROOT}/multijudge/qwen3_vl.py"
    HARD_MAX_NEW_TOKENS=96
    QUALIFICATION_CONFIG="${SRC_ROOT}/src/multijudge_workflows/configs/soft_score_qualification_v1.json"
    ;;
  safework)
    ADAPTER="safework"
    MODEL_PATH="/root/autodl-tmp/model/SafeWork-RM-Safety-7B"
    REVISION="be345f29425fe94586c0598785a143703bbbc4fc"
    ADAPTER_FILE="${SRC_ROOT}/multijudge/safework.py"
    HARD_MAX_NEW_TOKENS=32
    QUALIFICATION_CONFIG="${SRC_ROOT}/src/multijudge_workflows/configs/soft_score_qualification_v3.json"
    ;;
  internvl)
    ADAPTER="internvl"
    MODEL_PATH="/root/autodl-tmp/model/InternVL3_5-8B-Instruct"
    REVISION="6c2034f6f3d22bbbff919b11b91c5721bba84f8d"
    ADAPTER_FILE="${SRC_ROOT}/multijudge/internvl.py"
    HARD_MAX_NEW_TOKENS=32
    QUALIFICATION_CONFIG="${SRC_ROOT}/src/multijudge_workflows/configs/soft_score_qualification_internvl_v2.json"
    ;;
  minicpm)
    ADAPTER="minicpm"
    MODEL_PATH="/root/autodl-tmp/model/MiniCPM-V-4_5"
    REVISION="2626e837a54905aab70fae9325153ef3454387ab"
    ADAPTER_FILE="${SRC_ROOT}/multijudge/minicpm.py"
    HARD_MAX_NEW_TOKENS=32
    QUALIFICATION_CONFIG="${SRC_ROOT}/src/multijudge_workflows/configs/soft_score_qualification_v2.json"
    ;;
  *)
    echo "Unknown adapter: $1" >&2
    exit 2
    ;;
esac

RESULT_ROOT="/root/result/mmds/qualification/${ADAPTER}"
AUDIT_DIR="${RESULT_ROOT}/audit"
FINGERPRINT="${AUDIT_DIR}/model_fingerprint.json"

for path in \
  "${SRC_ROOT}/src/multijudge_workflows/run_soft_scores.py" \
  "${SRC_ROOT}/src/multijudge_workflows/audit_soft_qualification.py" \
  "${SRC_ROOT}/multijudge/soft_scores.py" \
  "$ADAPTER_FILE" \
  "${DATASET_ROOT}/MMDS/mmds.jsonl" \
  "${MANIFEST}" \
  "${POLICY}" \
  "${QUALIFICATION_CONFIG}" \
  "${MODEL_PATH}/config.json"; do
  if [[ ! -f "$path" ]]; then
    echo "Missing required file: $path" >&2
    exit 1
  fi
done

cd "$SRC_ROOT"
python -c "import torch, transformers, PIL, tqdm; assert torch.cuda.is_available(); print('torch='+torch.__version__); print('transformers='+transformers.__version__); print('gpu='+torch.cuda.get_device_name(0))"

mkdir -p "$AUDIT_DIR"
if [[ ! -f "$FINGERPRINT" ]]; then
  python -u src/multijudge_workflows/capture_runtime.py \
    --model-path "$MODEL_PATH" \
    --output-dir "$AUDIT_DIR"
fi

COMMON=(
  --adapter "$ADAPTER"
  --dataset-root "$DATASET_ROOT"
  --manifest "$MANIFEST"
  --policy "$POLICY"
  --model-path "$MODEL_PATH"
  --model-revision "$REVISION"
  --model-fingerprint "$FINGERPRINT"
  --qualification-config "$QUALIFICATION_CONFIG"
  --hard-regression
  --hard-max-new-tokens "$HARD_MAX_NEW_TOKENS"
  --attn-implementation sdpa
  --seed 20260729
)

python -u src/multijudge_workflows/run_soft_scores.py \
  "${COMMON[@]}" \
  --limit 3 \
  --minimum-hard-agreement 0.0 \
  --run-dir "${RESULT_ROOT}/qual3-a"

python -u src/multijudge_workflows/run_soft_scores.py \
  "${COMMON[@]}" \
  --limit 3 \
  --minimum-hard-agreement 0.0 \
  --run-dir "${RESULT_ROOT}/qual3-b"

python -u src/multijudge_workflows/audit_soft_qualification.py \
  --first "${RESULT_ROOT}/qual3-a/judgments.jsonl" \
  --second "${RESULT_ROOT}/qual3-b/judgments.jsonl" \
  --qualification-config "$QUALIFICATION_CONFIG" \
  --minimum-hard-agreement 0.0 \
  --output "${RESULT_ROOT}/qual3-audit.json"

python -u src/multijudge_workflows/run_soft_scores.py \
  "${COMMON[@]}" \
  --run-dir "${RESULT_ROOT}/qual100-a"

python -u src/multijudge_workflows/run_soft_scores.py \
  "${COMMON[@]}" \
  --run-dir "${RESULT_ROOT}/qual100-b"

python -u src/multijudge_workflows/audit_soft_qualification.py \
  --first "${RESULT_ROOT}/qual100-a/judgments.jsonl" \
  --second "${RESULT_ROOT}/qual100-b/judgments.jsonl" \
  --qualification-config "$QUALIFICATION_CONFIG" \
  --output "${RESULT_ROOT}/qualification_audit.json" \
  --enforce-codebook-gate

echo "${ADAPTER} soft-score qualification completed successfully."
