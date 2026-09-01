#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="Qwen/Qwen3-VL-8B-Instruct"
MODEL_REVISION="5d854aab08710c16b980ec6d603d863b3821b915"
MODEL_DIR="/root/autodl-tmp/model/Qwen3-VL-8B-Instruct"

python -m pip install --upgrade "modelscope-hub==0.1.6"
mkdir -p "${MODEL_DIR}"

modelscope download "${MODEL_ID}" \
  --revision "${MODEL_REVISION}" \
  --local-dir "${MODEL_DIR}" \
  --max-workers 8

python - "${MODEL_DIR}" "${MODEL_REVISION}" <<'PY'
import json
import sys
from pathlib import Path

model_dir = Path(sys.argv[1])
revision = sys.argv[2]
config_path = model_dir / "config.json"
if not config_path.is_file():
    raise SystemExit(f"Missing {config_path}")

config = json.loads(config_path.read_text(encoding="utf-8"))
architectures = config.get("architectures") or []
if "Qwen3VLForConditionalGeneration" not in architectures:
    raise SystemExit(f"Unexpected architecture: {architectures}")

weight_files = sorted(model_dir.glob("*.safetensors"))
if not weight_files:
    raise SystemExit("No safetensors weight files were downloaded")

size_gib = sum(path.stat().st_size for path in model_dir.rglob("*") if path.is_file())
size_gib /= 1024 ** 3
print(f"Model ready: {model_dir}")
print(f"Revision: {revision}")
print(f"Architecture: {architectures[0]}")
print(f"Weight files: {len(weight_files)}")
print(f"Directory size: {size_gib:.2f} GiB")
PY
