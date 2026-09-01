#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="/root/src${PYTHONPATH:+:${PYTHONPATH}}"

SOURCE="/root/autodl-tmp/Falcon_test.csv"
URL="${FALCON_URL:-https://huggingface.co/datasets/zhangrjjj/Falcon/resolve/53341884f7760e434481d88da02d50668b19d566/Falcon_test.csv}"
SHA256="cc40bc880fb948f96494daac5e4b72f50984d3541e2f75a101aeca2a8d2ec0f4"

mkdir -p /root/autodl-tmp /root/datasets/Falcon

if command -v curl >/dev/null 2>&1; then
  curl --fail --location --continue-at - --retry 5 --retry-delay 5 \
    --output "$SOURCE" "$URL"
elif command -v wget >/dev/null 2>&1; then
  wget --continue --output-document="$SOURCE" "$URL"
else
  echo "Neither curl nor wget is available." >&2
  exit 1
fi

echo "$SHA256  $SOURCE" | sha256sum -c -

cd /root/src
python -u src/multijudge_workflows/materialize_falcon_images.py \
  --source "$SOURCE" \
  --manifest /root/src/src/multijudge_workflows/configs/falcon_external_unlabeled.jsonl \
  --dataset-root /root/datasets \
  --output /root/result/external/falcon_test/image_materialization_audit.json

rm -f -- "$SOURCE"
test ! -e "$SOURCE"

python -c "import json; p='/root/result/external/falcon_test/image_materialization_audit.json'; d=json.load(open(p)); assert d['passed'] is True and d['source_label_columns_accessed'] is False and d['runtime_manifest_contains_labels'] is False; print({'falcon_images_ready':d['materialized_record_count'],'labels_loaded':False})"

echo "Falcon images materialized and the temporary labeled CSV was removed."
