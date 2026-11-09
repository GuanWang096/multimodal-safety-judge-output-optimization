"""Read-only recomputation of archived external diagnostics; no new inference/fit."""
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results/external"
OUT = ROOT / "results/reproduced/revision/archive"


def read(path): return json.loads(path.read_text(encoding="utf-8"))


def quantile(values, q):
    values = sorted(values)
    position = (len(values) - 1) * q
    lower = int(position)
    weight = position - lower
    return values[lower] * (1 - weight) + values[min(lower + 1, len(values) - 1)] * weight


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    budget_path = SOURCE / "beavertails_v/adaptation_budget_diagnostic.json"
    model_path = SOURCE / "falcon_test/adapted_models.json"
    pred_path = SOURCE / "falcon_test/confirmation_predictions_labeled.jsonl"
    budget = read(budget_path)
    output = {"role": "archived_posthoc_diagnostic_recomputation_no_new_inference_or_training", "beaver": []}
    for family, method in budget["methods"].items():
        for groups, item in method["budgets"].items():
            repeats = item["repeats"]
            assert len(repeats) == item["repeat_count"] == 30
            delta = [r["adapted_metrics"]["macro_f1"] - r["majority_metrics"]["macro_f1"] for r in repeats]
            assert all(abs(d - r["macro_f1_delta"]) < 1e-12 for d, r in zip(delta, repeats))
            mean = statistics.mean(delta)
            interval = [quantile(delta, q) for q in (0.025, 0.975)]
            positive = sum(d > 0 for d in delta) / len(delta)
            assert abs(mean - item["mean_macro_f1_delta"]) < 1e-12
            assert all(abs(a-b) < 1e-12 for a,b in zip(interval, item["delta_interval_across_splits"]))
            assert abs(positive - item["positive_delta_fraction"]) < 1e-12
            labels = sorted({r["calibration_record_count"] for r in repeats})
            assert all(r["calibration_group_count"] == int(groups) for r in repeats)
            output["beaver"].append({"family": family, "adaptation_groups": int(groups),
                                      "labeled_record_counts": labels, "repeats": len(repeats),
                                      "mean_macro_f1_delta": mean, "split_variability_interval": interval,
                                      "positive_delta_fraction": positive})
    model = read(model_path)["mechanism_control_hard_model"]
    def posterior(bits):
        score = model["intercept"] + sum(w*x for w,x in zip(model["coefficients"], bits))
        return 1 / (1 + math.exp(-score))
    patterns = []
    for bits in itertools.product((0,1), repeat=3):
        p = posterior(bits)
        patterns.append({"votes": list(bits), "unsafe_probability": p,
                         "adapted_hard_decision": int(p >= model["decision_threshold"]),
                         "majority_decision": int(sum(bits) >= 2)})
    assert all(r["adapted_hard_decision"] == r["majority_decision"] for r in patterns)
    rows = [json.loads(v) for v in pred_path.read_text(encoding="utf-8").splitlines() if v.strip()]
    errors = []
    for row in rows:
        bits = [int(row["raw_p_unsafe"][j] >= .5) for j in model["feature_names"]]
        p = posterior(bits)
        errors.append(abs(p - row["probabilities"]["adapted_hard_logits"]))
        predicted = "unsafe" if p >= model["decision_threshold"] else "safe"
        assert predicted == row["predictions"]["adapted_hard_logits"] == row["predictions"]["same_pool_majority"]
    assert len(rows) == 1068 and max(errors) < 1e-12
    output["falcon"] = {"feature_names": model["feature_names"], "patterns": patterns,
                         "all_eight_patterns_equal_majority": True, "archived_confirmation_records_checked": len(rows),
                         "max_posterior_reconstruction_error": max(errors), "hard_decision_disagreements": 0}
    output["interpretation"] = [
        "Beaver results are existing post-hoc adaptation-budget diagnostics; intervals describe repeated-split variability, not confirmation CIs.",
        "Falcon equality applies to this archived fitted binary-feature rule at its frozen threshold, not to every logistic model or its probability values.",
        "No claim of unchanged cross-domain superiority or statistically established Falcon superiority."]
    output["inputs"] = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in (budget_path,model_path,pred_path)}
    (OUT / "external_diagnostics.json").write_text(json.dumps(output, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    lines = ["# 外部数据归档诊断", "", "仅重算既有归档，没有新推理、重新拟合或新确认性检验。", "",
             "## BeaverTails-V：原soft-logit预算实验", "",
             "每档30次组件隔离划分；下列区间是跨划分变异，不是确认性置信区间。另一历史候选soft_plus_vote也完整保存在JSON中，未提升为最终方法。", "",
             "| 适配组数 | 标签记录数 | 平均Macro-F1差 | 跨划分区间 | 正差比例 |", "|---:|---:|---:|---|---:|"]
    for r in output["beaver"]:
        if r["family"] != "soft_logits": continue
        lo,hi = r["split_variability_interval"]
        lines.append(f"| {r['adaptation_groups']} | {r['labeled_record_counts']} | {r['mean_macro_f1_delta']:+.4f} | [{lo:.4f}, {hi:.4f}] | {r['positive_delta_fraction']:.1%} |")
    lines += ["", "## Falcon：已冻结二值模型的8种模式", "", "顺序：Qwen、SafeWork、MiniCPM。", "", "| 输入 | posterior | 二值模型 | 多数投票 |", "|---|---:|---:|---:|"]
    for r in patterns:
        lines.append(f"| {r['votes']} | {r['unsafe_probability']:.6f} | {r['adapted_hard_decision']} | {r['majority_decision']} |")
    lines += ["", f"全部8种模式相同；已逐条核对1068条归档确认预测，最大后验重构误差{max(errors):.3g}。概率值仍不是多数投票的0/1输出；不能据分类相同把二者当作两份独立分类增益证据。"]
    (OUT / "EXTERNAL_DIAGNOSTICS.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(json.dumps({"beaver_budget_method_pairs":len(output["beaver"]), "falcon_patterns":8, "falcon_checked":len(rows), "max_error":max(errors)}))


if __name__ == "__main__": main()
