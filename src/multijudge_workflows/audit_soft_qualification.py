from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def load_records(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            sample_id = str(record["sample_id"])
            if sample_id in records:
                raise ValueError(f"Duplicate sample ID: {sample_id}")
            records[sample_id] = record
    return records


def _rank(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    result = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + end - 1) / 2.0
        for offset in range(start, end):
            result[order[offset]] = rank
        start = end
    return result


def _pearson(first: list[float], second: list[float]) -> float | None:
    if len(first) != len(second) or len(first) < 2:
        return None
    mean_first = sum(first) / len(first)
    mean_second = sum(second) / len(second)
    numerator = sum(
        (left - mean_first) * (right - mean_second)
        for left, right in zip(first, second)
    )
    denominator = math.sqrt(
        sum((value - mean_first) ** 2 for value in first)
        * sum((value - mean_second) ** 2 for value in second)
    )
    return numerator / denominator if denominator else None


def audit(
    first_path: Path,
    second_path: Path,
    *,
    enforce_codebook_gate: bool = False,
    min_codebook_spearman: float = 0.5,
    max_codebook_sign_flip_rate: float = 0.25,
    max_repeat_margin_difference: float = 1e-4,
    minimum_hard_agreement: float = 0.95,
    allow_native_sequence_fallback: bool = True,
) -> dict[str, Any]:
    first = load_records(first_path)
    second = load_records(second_path)
    if set(first) != set(second):
        raise ValueError("Repeat runs contain different sample IDs")
    sample_ids = sorted(first)
    repeat_differences = []
    direction_mismatches = []
    margin_a: list[float] = []
    margin_b: list[float] = []
    probability_a: list[float] = []
    probability_b: list[float] = []
    codebook_sign_flips = 0
    codebook_mode: str | None = None
    hard_valid = 0
    hard_agree = 0

    for sample_id in sample_ids:
        left = first[sample_id]
        right = second[sample_id]
        left_margin = float(left["soft_evidence"]["margin_unsafe_minus_safe"])
        right_margin = float(right["soft_evidence"]["margin_unsafe_minus_safe"])
        repeat_differences.append(abs(left_margin - right_margin))
        if (
            left["soft_evidence"]["predicted_label"]
            != right["soft_evidence"]["predicted_label"]
        ):
            direction_mismatches.append(sample_id)

        books = left["soft_evidence"].get("codebooks", [])
        score_method = left["soft_evidence"].get("score_method")
        current_mode = (
            "paired_codebooks"
            if len(books) >= 2
            else (
                "native_label_sequence_fallback"
                if score_method == "native_label_sequence_likelihood"
                and not books
                else "invalid"
            )
        )
        if current_mode == "invalid":
            raise ValueError(
                f"Unsupported codebook evidence for sample {sample_id}"
            )
        if codebook_mode is None:
            codebook_mode = current_mode
        elif codebook_mode != current_mode:
            raise ValueError("Qualification mixes incompatible score methods")
        if current_mode == "paired_codebooks":
            book_a_margin = float(books[0]["margin_unsafe_minus_safe"])
            book_b_margin = float(books[1]["margin_unsafe_minus_safe"])
            margin_a.append(book_a_margin)
            margin_b.append(book_b_margin)
            probability_a.append(float(books[0]["p_unsafe"]))
            probability_b.append(float(books[1]["p_unsafe"]))
            book_a_prediction = books[0].get(
                "predicted_label",
                "unsafe" if book_a_margin >= 0 else "safe",
            )
            book_b_prediction = books[1].get(
                "predicted_label",
                "unsafe" if book_b_margin >= 0 else "safe",
            )
            if book_a_prediction != book_b_prediction:
                codebook_sign_flips += 1

        hard = left.get("hard_regression")
        if hard and hard.get("status") == "ok":
            hard_valid += 1
            hard_agree += hard.get("agrees_with_soft_argmax") is True

    first_order = sorted(
        sample_ids,
        key=lambda sample_id: (
            float(first[sample_id]["soft_evidence"]["margin_unsafe_minus_safe"]),
            sample_id,
        ),
    )
    second_order = sorted(
        sample_ids,
        key=lambda sample_id: (
            float(second[sample_id]["soft_evidence"]["margin_unsafe_minus_safe"]),
            sample_id,
        ),
    )
    max_repeat_difference = max(repeat_differences, default=float("inf"))
    hard_agreement_rate = hard_agree / hard_valid if hard_valid else None
    codebook_margin_spearman = (
        _pearson(_rank(margin_a), _rank(margin_b))
        if codebook_mode == "paired_codebooks"
        else None
    )
    codebook_probability_spearman = (
        _pearson(_rank(probability_a), _rank(probability_b))
        if codebook_mode == "paired_codebooks"
        else None
    )
    codebook_sign_flip_rate = (
        codebook_sign_flips / len(sample_ids)
        if sample_ids and codebook_mode == "paired_codebooks"
        else None
    )
    codebook_gate_passed = (
        not enforce_codebook_gate
        or (
            codebook_mode == "native_label_sequence_fallback"
            and allow_native_sequence_fallback
        )
        or (
            codebook_probability_spearman is not None
            and codebook_probability_spearman >= min_codebook_spearman
            and codebook_sign_flip_rate is not None
            and codebook_sign_flip_rate <= max_codebook_sign_flip_rate
        )
    )
    passed = (
        bool(sample_ids)
        and max_repeat_difference <= max_repeat_margin_difference
        and not direction_mismatches
        and first_order == second_order
        and hard_agreement_rate is not None
        and hard_agreement_rate >= minimum_hard_agreement
        and codebook_gate_passed
    )
    return {
        "schema": "soft-score-qualification-audit-v1",
        "passed": passed,
        "sample_count": len(sample_ids),
        "repeatability": {
            "max_absolute_margin_difference": max_repeat_difference,
            "maximum_allowed_margin_difference": (
                max_repeat_margin_difference
            ),
            "direction_mismatch_count": len(direction_mismatches),
            "direction_mismatch_sample_ids": direction_mismatches,
            "sample_order_identical": first_order == second_order,
        },
        "codebook_sensitivity": {
            "mode": codebook_mode,
            "applicable": codebook_mode == "paired_codebooks",
            "native_sequence_fallback_accepted": (
                codebook_mode == "native_label_sequence_fallback"
                and allow_native_sequence_fallback
            ),
            "pearson_margin_correlation": _pearson(margin_a, margin_b),
            "spearman_margin_correlation": codebook_margin_spearman,
            "pearson_probability_correlation": _pearson(
                probability_a,
                probability_b,
            ),
            "spearman_probability_correlation": (
                codebook_probability_spearman
            ),
            "sign_flip_count": codebook_sign_flips,
            "sign_flip_rate": codebook_sign_flip_rate,
            "mean_absolute_margin_difference": (
                sum(abs(a - b) for a, b in zip(margin_a, margin_b))
                / len(margin_a)
                if margin_a
                else None
            ),
            "mean_absolute_probability_difference": (
                sum(
                    abs(a - b)
                    for a, b in zip(probability_a, probability_b)
                )
                / len(probability_a)
                if probability_a
                else None
            ),
            "gate_enforced": enforce_codebook_gate,
            "minimum_spearman": min_codebook_spearman,
            "maximum_sign_flip_rate": max_codebook_sign_flip_rate,
            "gate_passed": codebook_gate_passed,
        },
        "hard_regression": {
            "valid_count": hard_valid,
            "argmax_agreement_rate": hard_agreement_rate,
            "minimum_argmax_agreement": minimum_hard_agreement,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--second", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--qualification-config", type=Path, required=True)
    parser.add_argument("--enforce-codebook-gate", action="store_true")
    parser.add_argument("--minimum-hard-agreement", type=float)
    args = parser.parse_args()
    qualification = json.loads(
        args.qualification_config.read_text(encoding="utf-8")
    )
    if qualification.get("schema") not in {
        "soft-score-qualification-config-v1",
        "soft-score-qualification-config-v2",
        "soft-score-qualification-config-v3",
    }:
        raise ValueError("Unexpected soft-score qualification config")
    result = audit(
        args.first,
        args.second,
        enforce_codebook_gate=args.enforce_codebook_gate,
        min_codebook_spearman=float(
            qualification["minimum_codebook_probability_spearman"]
        ),
        max_codebook_sign_flip_rate=float(
            qualification["maximum_codebook_direction_flip_rate"]
        ),
        max_repeat_margin_difference=float(
            qualification["deterministic_margin_tolerance"]
        ),
        minimum_hard_agreement=(
            float(args.minimum_hard_agreement)
            if args.minimum_hard_agreement is not None
            else float(qualification["minimum_hard_argmax_agreement"])
        ),
        allow_native_sequence_fallback=bool(
            qualification.get(
                "native_label_sequence_fallback_satisfies_codebook_gate",
                False,
            )
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"passed": result["passed"], "sample_count": result["sample_count"]}))
    if not result["passed"]:
        raise RuntimeError("Soft-score qualification audit failed")


if __name__ == "__main__":
    main()
