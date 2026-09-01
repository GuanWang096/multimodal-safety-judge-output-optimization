from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path
from statistics import pstdev
from typing import Any


V15_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = (
    V15_ROOT / "src"
    if (V15_ROOT / "src" / "multijudge").is_dir()
    else V15_ROOT
)
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from multijudge.dataset import load_dataset_manifest  # noqa: E402
from multijudge.consistency import (  # noqa: E402
    IMAGE_TRANSFORMS,
    materialize_sample_view,
)
from multijudge.internvl import InternVLJudgeAdapter  # noqa: E402
from multijudge.minicpm import MiniCPMJudgeAdapter  # noqa: E402
from multijudge.policy import load_policy  # noqa: E402
from multijudge.qwen3_vl import Qwen3VLJudgeAdapter  # noqa: E402
from multijudge.runtime import collect_environment  # noqa: E402
from multijudge.safework import SafeWorkJudgeAdapter  # noqa: E402
from multijudge.schema import sha256_file  # noqa: E402
from multijudge.soft_scores import validate_soft_record  # noqa: E402


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def completed_sample_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    result: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                validate_soft_record(record)
                result.add(str(record["sample_id"]))
            except (json.JSONDecodeError, KeyError, ValueError) as exc:
                raise RuntimeError(
                    f"Invalid soft-score record at line {line_number}: {path}"
                ) from exc
    return result


def progress_iterator(values: list[Any]) -> Any:
    try:
        from tqdm import tqdm
    except ImportError:
        return values
    return tqdm(values, unit="sample", dynamic_ncols=True)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed % (2**32))
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def build_adapter(args: argparse.Namespace, policy: Any) -> Any:
    common = {
        "model_path": args.model_path,
        "model_revision": args.model_revision,
        "policy": policy,
        "max_new_tokens": args.hard_max_new_tokens,
        "attn_implementation": args.attn_implementation,
    }
    if args.adapter == "qwen":
        return Qwen3VLJudgeAdapter(**common)
    if args.adapter == "safework":
        return SafeWorkJudgeAdapter(**common)
    if args.adapter == "internvl":
        return InternVLJudgeAdapter(**common)
    if args.adapter == "minicpm":
        return MiniCPMJudgeAdapter(**common)
    raise ValueError(f"Unsupported soft-score adapter: {args.adapter}")


def _classification_counts(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    labeled = [record for record in records if record.get("expected_label")]
    if len(labeled) != len(records) or not labeled:
        return None
    tp = sum(
        record["expected_label"] == "unsafe"
        and record["soft_evidence"]["predicted_label"] == "unsafe"
        for record in labeled
    )
    tn = sum(
        record["expected_label"] == "safe"
        and record["soft_evidence"]["predicted_label"] == "safe"
        for record in labeled
    )
    fp = sum(
        record["expected_label"] == "safe"
        and record["soft_evidence"]["predicted_label"] == "unsafe"
        for record in labeled
    )
    fn = sum(
        record["expected_label"] == "unsafe"
        and record["soft_evidence"]["predicted_label"] == "safe"
        for record in labeled
    )
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def summarize_run(
    records: list[dict[str, Any]],
    *,
    require_hard_regression: bool,
    minimum_hard_agreement: float = 0.95,
) -> dict[str, Any]:
    margins = [
        float(record["soft_evidence"]["margin_unsafe_minus_safe"])
        for record in records
    ]
    probabilities = [
        float(record["soft_evidence"]["p_unsafe"]) for record in records
    ]
    finite = all(
        math.isfinite(value)
        for value in margins + probabilities
    )
    in_range = all(0.0 <= value <= 1.0 for value in probabilities)
    unique_probability_count = len({round(value, 12) for value in probabilities})
    nondegenerate = (
        len(records) < 20
        or (
            unique_probability_count >= 5
            and pstdev(probabilities) > 1e-8
        )
    )

    hard_records = [
        record for record in records if "hard_regression" in record
    ]
    hard_valid = sum(
        record["hard_regression"]["status"] == "ok"
        for record in hard_records
    )
    hard_agreements = sum(
        record["hard_regression"].get("agrees_with_soft_argmax") is True
        for record in hard_records
    )
    hard_agreement_rate = (
        hard_agreements / hard_valid if hard_valid else None
    )
    hard_gate = (
        not require_hard_regression
        or (
            len(hard_records) == len(records)
            and hard_valid == len(records)
            and hard_agreement_rate is not None
            and hard_agreement_rate >= minimum_hard_agreement
        )
    )
    passed = bool(records) and finite and in_range and nondegenerate and hard_gate
    return {
        "schema": "soft-score-run-summary-v1",
        "passed": passed,
        "sample_count": len(records),
        "finite_scores": finite,
        "probabilities_in_range": in_range,
        "unique_probability_count": unique_probability_count,
        "probability_population_std": pstdev(probabilities) if probabilities else None,
        "nondegenerate": nondegenerate,
        "hard_regression_required": require_hard_regression,
        "hard_regression_record_count": len(hard_records),
        "hard_regression_valid_count": hard_valid,
        "hard_argmax_agreement_rate": hard_agreement_rate,
        "minimum_hard_argmax_agreement": minimum_hard_agreement,
        "classification": _classification_counts(records),
    }


def run(args: argparse.Namespace) -> None:
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    result_path = run_dir / "judgments.jsonl"
    status_path = run_dir / "status.json"

    samples = load_dataset_manifest(args.dataset_root, args.manifest)
    if args.limit is not None:
        samples = samples[: args.limit]
    if not samples:
        raise ValueError("No samples selected")
    existing = completed_sample_ids(result_path)
    pending = [sample for sample in samples if sample.sample_id not in existing]
    policy = load_policy(args.policy)
    qualification = json.loads(
        args.qualification_config.read_text(encoding="utf-8")
    )
    if qualification.get("schema") not in {
        "soft-score-qualification-config-v1",
        "soft-score-qualification-config-v2",
        "soft-score-qualification-config-v3",
    }:
        raise ValueError("Unexpected soft-score qualification config")
    seed_everything(args.seed)
    adapter = build_adapter(args, policy)

    fingerprint = json.loads(
        args.model_fingerprint.read_text(encoding="utf-8")
    )
    if Path(fingerprint["model_path"]).resolve() != args.model_path.resolve():
        raise RuntimeError("Model fingerprint belongs to a different model path")

    config = {
        "schema": "soft-score-run-config-v1",
        "adapter": args.adapter,
        "adapter_version": adapter.soft_identity.adapter_version,
        "score_schema_version": "soft-evidence-v1",
        "dataset_root": str(args.dataset_root.resolve()),
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256_file(args.manifest.resolve()),
        "policy": str(args.policy.resolve()),
        "policy_sha256": policy.sha256,
        "model_path": str(args.model_path.resolve()),
        "model_revision": args.model_revision,
        "model_manifest_sha256": fingerprint["manifest_sha256"],
        "seed": args.seed,
        "attn_implementation": args.attn_implementation,
        "hard_regression": args.hard_regression,
        "hard_max_new_tokens": args.hard_max_new_tokens,
        "minimum_hard_argmax_agreement": (
            float(args.minimum_hard_agreement)
            if args.minimum_hard_agreement is not None
            else float(qualification["minimum_hard_argmax_agreement"])
        ),
        "qualification_config": str(args.qualification_config.resolve()),
        "qualification_config_sha256": sha256_file(
            args.qualification_config.resolve()
        ),
        "selected_sample_count": len(samples),
        "consistency_view_id": args.consistency_view_id,
        "policy_render_variant": policy.render_variant,
        "policy_semantic_sha256": policy.semantic_sha256,
        "image_transform": args.image_transform,
    }
    config_path = run_dir / "config.json"
    if config_path.is_file():
        prior = json.loads(config_path.read_text(encoding="utf-8"))
        if prior != config:
            raise RuntimeError(
                "Run directory contains a different config; choose a new directory"
            )
    else:
        write_json(config_path, config)
    fingerprint_copy = run_dir / "model_fingerprint.json"
    if not fingerprint_copy.is_file():
        shutil.copy2(args.model_fingerprint, fingerprint_copy)
    qualification_copy = run_dir / "qualification_config.json"
    if not qualification_copy.is_file():
        shutil.copy2(args.qualification_config, qualification_copy)
    if not (run_dir / "runtime.json").is_file():
        write_json(run_dir / "runtime.json", collect_environment())

    started = time.time()
    write_json(
        status_path,
        {
            "state": "running",
            "started_at_unix": started,
            "selected": len(samples),
            "already_completed": len(existing),
            "pending": len(pending),
        },
    )
    try:
        with result_path.open("a", encoding="utf-8", buffering=1) as output:
            for index, sample in enumerate(progress_iterator(pending), start=1):
                view_sample, view_metadata = materialize_sample_view(
                    sample,
                    image_transform=args.image_transform,
                    cache_dir=run_dir / "view-images",
                )
                record = adapter.soft_judge(
                    view_sample,
                    input_mode="native",
                    panel_dir=run_dir / "unused-panels",
                )
                image_audit = view_metadata["image_audit"]
                record["consistency_view"] = {
                    "view_id": args.consistency_view_id,
                    "policy_render_variant": policy.render_variant,
                    "policy_render_sha256": policy.sha256,
                    "policy_semantic_sha256": policy.semantic_sha256,
                    "image_transform": args.image_transform,
                    "image_order_preserved": view_metadata[
                        "image_order_preserved"
                    ],
                    "source_image_sha256": [
                        value["source_sha256"] for value in image_audit
                    ],
                    "transformed_image_sha256": [
                        value["transformed_sha256"] for value in image_audit
                    ],
                }
                if args.hard_regression:
                    hard = adapter.judge(
                        view_sample,
                        input_mode="native",
                        panel_dir=run_dir / "unused-panels",
                    )
                    hard_decision = hard["decision"]
                    record["hard_regression"] = {
                        "adapter_version": adapter.identity.adapter_version,
                        "status": hard_decision["status"],
                        "label": hard_decision["label"],
                        "agrees_with_soft_argmax": (
                            hard_decision["status"] == "ok"
                            and hard_decision["label"]
                            == record["soft_evidence"]["predicted_label"]
                        ),
                    }
                validate_soft_record(record)
                output.write(
                    json.dumps(record, ensure_ascii=False, separators=(",", ":"))
                    + "\n"
                )
                write_json(
                    status_path,
                    {
                        "state": "running",
                        "started_at_unix": started,
                        "selected": len(samples),
                        "completed": len(existing) + index,
                        "pending": len(pending) - index,
                        "last_sample_id": sample.sample_id,
                    },
                )

        selected_ids = {sample.sample_id for sample in samples}
        records = [
            json.loads(line)
            for line in result_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        selected_records = [
            record for record in records if record["sample_id"] in selected_ids
        ]
        minimum_hard_agreement = (
            float(args.minimum_hard_agreement)
            if args.minimum_hard_agreement is not None
            else float(qualification["minimum_hard_argmax_agreement"])
        )
        summary = summarize_run(
            selected_records,
            require_hard_regression=args.hard_regression,
            minimum_hard_agreement=minimum_hard_agreement,
        )
        write_json(run_dir / "qualification_summary.json", summary)
        final_state = (
            "completed"
            if len(selected_records) == len(samples) and summary["passed"]
            else "quality_gate_failed"
        )
        write_json(
            status_path,
            {
                "state": final_state,
                "started_at_unix": started,
                "ended_at_unix": time.time(),
                "selected": len(samples),
                "completed": len(selected_records),
                "summary_passed": summary["passed"],
            },
        )
        if final_state != "completed":
            raise RuntimeError("Soft-score run failed its single-run quality gate")
    except BaseException as exc:
        current = (
            json.loads(status_path.read_text(encoding="utf-8"))
            if status_path.is_file()
            else {}
        )
        if current.get("state") not in {"completed", "quality_gate_failed"}:
            current.update(
                {
                    "state": "failed",
                    "ended_at_unix": time.time(),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            write_json(status_path, current)
        raise
    finally:
        adapter.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--adapter",
        choices=("qwen", "safework", "internvl", "minicpm"),
        required=True,
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--model-fingerprint", type=Path, required=True)
    parser.add_argument("--qualification-config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--hard-regression", action="store_true")
    parser.add_argument("--minimum-hard-agreement", type=float)
    parser.add_argument("--hard-max-new-tokens", type=int, default=96)
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument(
        "--consistency-view-id",
        default="canonical",
    )
    parser.add_argument(
        "--image-transform",
        choices=IMAGE_TRANSFORMS,
        default="original",
    )
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
