from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
V15_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_CONFIG = SCRIPT_DIR / "configs" / "model_data_overlap_evidence.json"
DEFAULT_ACCEPTANCE = (
    V15_ROOT / "results" / "reputation" / "test" / "acceptance.json"
)
DEFAULT_OUTPUT_DIR = (
    V15_ROOT
    / "results"
    / "mmds"
    / "data"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def term_line_matches(text: str, terms: list[str]) -> dict[str, list[int]]:
    lines = text.splitlines()
    result: dict[str, list[int]] = {}
    for term in terms:
        pattern = re.compile(rf"(?<![A-Za-z0-9_]){re.escape(term)}(?![A-Za-z0-9_])", re.I)
        result[term] = [
            index
            for index, line in enumerate(lines, start=1)
            if pattern.search(line)
        ]
    return result


def _dataset_status(
    dataset_key: str,
    term_found: bool,
    declared_training: set[str],
    declared_evaluation: set[str],
) -> str:
    if dataset_key in declared_training:
        return "declared_training_overlap"
    if dataset_key in declared_evaluation:
        return "declared_evaluation_exposure"
    if term_found:
        return "named_in_model_card_but_unclassified"
    return "no_direct_overlap_disclosed"


def build_report(
    config: dict[str, Any],
    acceptance: dict[str, Any],
    config_path: Path,
    acceptance_path: Path,
) -> dict[str, Any]:
    datasets = config["audited_datasets"]
    accepted_models = acceptance["models"]
    model_reports: dict[str, Any] = {}
    gate_errors: list[str] = []

    for key, expected in config["models"].items():
        actual = accepted_models.get(key)
        if actual is None:
            gate_errors.append(f"missing_acceptance_model:{key}")
            continue

        identity = actual.get("service_identity", {})
        identity_ok = (
            identity.get("model_id") == expected["model_id"]
            and identity.get("model_revision") == expected["model_revision"]
        )
        if not identity_ok:
            gate_errors.append(f"identity_or_revision_mismatch:{key}")

        readme_path = (SCRIPT_DIR / expected["metadata_readme"]).resolve()
        if not readme_path.is_file():
            gate_errors.append(f"missing_model_card:{key}")
            continue

        text = readme_path.read_text(encoding="utf-8", errors="replace")
        declared_training = set(expected["declared_training_overlap"])
        declared_evaluation = set(expected["declared_evaluation_overlap"])
        dataset_reports: dict[str, Any] = {}

        for dataset_key, dataset in datasets.items():
            matches = term_line_matches(text, dataset["search_terms"])
            term_found = any(matches.values())
            status = _dataset_status(
                dataset_key,
                term_found,
                declared_training,
                declared_evaluation,
            )
            if status == "named_in_model_card_but_unclassified":
                gate_errors.append(
                    f"unclassified_named_dataset_mention:{key}:{dataset_key}"
                )
            if (
                dataset_key in declared_training | declared_evaluation
                and not term_found
            ):
                gate_errors.append(
                    f"declared_overlap_not_found_in_frozen_card:{key}:{dataset_key}"
                )
            dataset_reports[dataset_key] = {
                "display_name": dataset["display_name"],
                "status": status,
                "term_line_matches": matches,
            }

        model_reports[key] = {
            "model_id": expected["model_id"],
            "model_revision": expected["model_revision"],
            "identity_verified": identity_ok,
            "adapter_version": identity.get("adapter_version"),
            "metadata_readme": str(readme_path),
            "metadata_readme_sha256": sha256_file(readme_path),
            "official_source": expected["official_source"],
            "training_disclosure": expected["training_disclosure"],
            "training_corpus_completeness": "incomplete",
            "datasets": dataset_reports,
            "required_sensitivity": expected["required_sensitivity"],
        }

    safe_work_siuo = (
        model_reports.get("safework", {})
        .get("datasets", {})
        .get("siuo", {})
        .get("status")
        == "declared_evaluation_exposure"
    )
    if not safe_work_siuo:
        gate_errors.append("safework_siuo_exposure_not_verified")

    return {
        "schema": "model-data-overlap-audit-v1",
        "status": "pass_with_declared_limitations" if not gate_errors else "failed",
        "passed": not gate_errors,
        "config": str(config_path.resolve()),
        "config_sha256": sha256_file(config_path),
        "acceptance": str(acceptance_path.resolve()),
        "acceptance_sha256": sha256_file(acceptance_path),
        "models": model_reports,
        "gate_errors": gate_errors,
        "frozen_interpretation": {
            "mmds": (
                "No audited model card directly names MMDS. Because the full training "
                "corpora are not disclosed, this is not proof of zero training overlap."
            ),
            "siuo": (
                "SafeWork's official card reports SIUO evaluation performance. SIUO "
                "results involving SafeWork are benchmark-exposed and require a "
                "leave-SafeWork-out sensitivity analysis."
            ),
            "claim_boundary": (
                "The audit supports transparent disclosure and sensitivity analysis; "
                "it does not establish sample-level training-data independence."
            ),
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# 四裁判模型与数据重叠审计",
        "",
        f"- 状态：`{report['status']}`",
        f"- 停止门：`{'PASS' if report['passed'] else 'FAIL'}`",
        f"- 正式 test acceptance SHA-256：`{report['acceptance_sha256']}`",
        f"- 冻结证据配置 SHA-256：`{report['config_sha256']}`",
        "",
        "## 审计口径",
        "",
        "本审计核对实际运行的模型身份与 revision，并扫描随实验冻结的官方模型卡副本。"
        "“模型卡未点名数据集”只表示未发现直接披露，不等于证明训练样本无重叠。",
        "",
        "| 裁判 | MMDS | SIUO | 训练语料披露 | 处理 |",
        "|---|---|---|---|---|",
    ]
    for model in report["models"].values():
        actions = "; ".join(model["required_sensitivity"]) or "常规披露"
        lines.append(
            "| {model} | {mmds} | {siuo} | {completeness} | {actions} |".format(
                model=model["model_id"],
                mmds=model["datasets"]["mmds"]["status"],
                siuo=model["datasets"]["siuo"]["status"],
                completeness=model["training_corpus_completeness"],
                actions=actions,
            )
        )

    lines.extend(
        [
            "",
            "## 冻结解释",
            "",
            "1. **MMDS：** 四份官方模型卡均未直接点名 MMDS；由于完整训练语料成员关系未公开，"
            "正文只能写“未发现公开披露的直接重叠”，不能写“保证无训练污染”。",
            "2. **SIUO：** SafeWork 官方模型卡公开报告 SIUO 准确率，因此 SafeWork 对 SIUO "
            "属于评测集暴露。所有包含 SafeWork 的 SIUO 结果必须标注为 benchmark-exposed，"
            "并同时报告去除 SafeWork 的敏感性结果。",
            "3. **主张边界：** MMDS 确认集仍可作为本研究冻结流程中的主要确认样本；SIUO "
            "只提供跨数据集排序或独立人工二分类诊断，不能单独证明概率校准迁移。",
            "",
            "## 可复核证据",
            "",
        ]
    )
    for model in report["models"].values():
        lines.extend(
            [
                f"### {model['model_id']}",
                "",
                f"- revision：`{model['model_revision']}`",
                f"- adapter：`{model['adapter_version']}`",
                f"- 本地模型卡 SHA-256：`{model['metadata_readme_sha256']}`",
                f"- 官方来源：{model['official_source']}",
                f"- 披露摘要：{model['training_disclosure']}",
                f"- MMDS 命中行：`{model['datasets']['mmds']['term_line_matches']}`",
                f"- SIUO 命中行：`{model['datasets']['siuo']['term_line_matches']}`",
                "",
            ]
        )
    if report["gate_errors"]:
        lines.extend(["## 停止门错误", ""])
        lines.extend(f"- `{error}`" for error in report["gate_errors"])
        lines.append("")
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, Any]:
    config_path = args.config.resolve()
    acceptance_path = args.acceptance.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    report = build_report(
        load_json(config_path),
        load_json(acceptance_path),
        config_path,
        acceptance_path,
    )
    json_path = output_dir / "model_data_overlap_audit.json"
    markdown_path = output_dir / "MODEL_DATA_OVERLAP_AUDIT.md"
    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(render_markdown(report) + "\n", encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "status": report["status"]}))
    if not report["passed"]:
        raise RuntimeError("Model-data overlap audit failed")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--acceptance", type=Path, default=DEFAULT_ACCEPTANCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
