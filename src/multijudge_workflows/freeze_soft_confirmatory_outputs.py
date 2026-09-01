from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping


JUDGES = ("qwen", "safework", "minicpm")
FORBIDDEN_LABEL_FIELDS = {
    "assistant_label",
    "assistant_dimension",
    "expected_label",
    "expected_dimension",
    "label",
    "ground_truth",
    "target",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def freeze(
    *,
    result_root: Path,
    manifest_path: Path,
    execution_manifest_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    manifest = load_jsonl(manifest_path)
    expected_ids = [str(value["sample_id"]) for value in manifest]
    if len(expected_ids) != 513 or len(set(expected_ids)) != 513:
        raise RuntimeError("Unexpected confirmatory manifest")
    if any(
        key.casefold() in FORBIDDEN_LABEL_FIELDS
        for value in manifest
        for key in value
    ):
        raise RuntimeError("Inference manifest contains label fields")
    execution = load_json(execution_manifest_path)
    if execution.get("labels_included") is not False:
        raise RuntimeError("Execution manifest is not label-blind")
    execution_content_hash = str(execution.get("content_sha256"))
    execution_unhashed = dict(execution)
    execution_unhashed.pop("content_sha256", None)
    if stable_json_sha256(execution_unhashed) != execution_content_hash:
        raise RuntimeError("Execution manifest content hash mismatch")

    judges: dict[str, Any] = {}
    for judge in JUDGES:
        formal = result_root / judge
        paths = {
            "judgments": formal / "judgments.jsonl",
            "config": formal / "config.json",
            "status": formal / "status.json",
            "coverage_audit": formal / "coverage_audit.json",
            "model_fingerprint": formal / "model_fingerprint.json",
            "qualification_config": formal / "qualification_config.json",
            "runtime": formal / "runtime.json",
        }
        missing = [str(path) for path in paths.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing {judge} artifacts: {missing}")
        audit = load_json(paths["coverage_audit"])
        if audit.get("passed") is not True or audit.get("labels_accessed") is not False:
            raise RuntimeError(f"{judge} coverage audit did not pass label-blindly")
        records = load_jsonl(paths["judgments"])
        observed_ids = [str(value["sample_id"]) for value in records]
        if observed_ids != expected_ids:
            raise RuntimeError(f"{judge} output order or membership differs")
        forbidden = {
            key
            for record in records
            for key in record
            if key.casefold() in FORBIDDEN_LABEL_FIELDS
        }
        if forbidden:
            raise RuntimeError(f"{judge} output contains label fields: {sorted(forbidden)}")
        if sha256_file(paths["judgments"]) != audit.get("judgments_sha256"):
            raise RuntimeError(f"{judge} judgments changed after coverage audit")
        judges[judge] = {
            "adapter_version": audit["adapter_version"],
            "record_count": len(records),
            "files": {
                name: {
                    "sha256": sha256_file(path),
                    "bytes": path.stat().st_size,
                }
                for name, path in paths.items()
            },
        }

    freeze_record: dict[str, Any] = {
        "schema": "softfusion-confirmatory-raw-output-freeze-v1",
        "status": "frozen",
        "labels_accessed": False,
        "confirmatory_manifest_sha256": sha256_file(manifest_path),
        "confirmatory_record_count": len(expected_ids),
        "execution_manifest_content_sha256": execution_content_hash,
        "judge_order": list(JUDGES),
        "judges": judges,
    }
    freeze_record["content_sha256"] = stable_json_sha256(freeze_record)
    write_json(output_path, freeze_record)
    return freeze_record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--execution-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = freeze(
        result_root=args.result_root.resolve(),
        manifest_path=args.manifest.resolve(),
        execution_manifest_path=args.execution_manifest.resolve(),
        output_path=args.output.resolve(),
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "record_count": result["confirmatory_record_count"],
                "content_sha256": result["content_sha256"],
                "labels_accessed": result["labels_accessed"],
            }
        )
    )


if __name__ == "__main__":
    main()
