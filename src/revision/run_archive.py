"""Archive-only costs and SafeWork aggregation sensitivity. No inference."""
import sys
sys.dont_write_bytecode=True
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"src"))
from multijudge.baselines import classification_metrics
P=ROOT/"results/mmds"
OUT=ROOT/"results/reproduced/revision/archive"
def rows(p):
    with p.open(encoding="utf-8") as f:return [json.loads(line) for line in f if line.strip()]
def save(name,data):
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/name).open("x",encoding="utf-8") as f:json.dump(data,f,indent=2,ensure_ascii=False,allow_nan=False)
confirm=rows(P/"evaluation/confirmatory_predictions.jsonl")
ids=[r["sample_id"] for r in confirm];y=np.array([r["expected_label"]=="unsafe" for r in confirm])
latencies=[];cost={};safework=None
for judge in ("qwen","safework","minicpm"):
    raw=rows(P/f"confirm/{judge}/judgments.jsonl");byid={r["sample_id"]:r for r in raw}
    assert len(byid)==len(raw)==513 and set(byid)==set(ids)
    ordered=[byid[i] for i in ids]
    latency=np.array([r["latency_ms_total"] for r in ordered])/1000
    vram=np.array([r["peak_vram_gib_max"] for r in ordered])
    assert np.isfinite(latency).all() and np.isfinite(vram).all()
    cost[judge]={"n":len(raw),"mean_seconds":float(latency.mean()),"p50_seconds":float(np.median(latency)),
                 "p95_seconds":float(np.quantile(latency,.95)),"sum_seconds":float(latency.sum()),
                 "peak_vram_gib_median":float(np.median(vram)),"peak_vram_gib_max":float(vram.max())}
    latencies.append(latency)
    if judge=="safework":safework=ordered
save("inference_cost.json",{"source":"archived_model_score_runs_not_new_GPU_measurements","judges":cost,
                            "derived_serial_mean_seconds":float(np.sum(latencies,axis=0).mean()),
                            "ideal_independent_parallel_max_mean_seconds":float(np.max(latencies,axis=0).mean()),
                            "parallel_warning":"derived_lower_bound_for_ideal_parallel_latency_not_measured_parallel_throughput"})
predictions={"max_turn":[],"last_turn":[],"review_ambiguous_native_argmax":[]}
details=[]
for r in safework:
    turns=sorted(r["soft_evidence"]["native_candidate_distribution"]["turns"],key=lambda t:t["turn_index"])
    assert turns
    mx=max(t["p_unsafe"] for t in turns);last=turns[-1]["p_unsafe"]
    assert abs(mx-r["soft_evidence"]["p_unsafe"])<1e-12
    ambiguous=any(t["native_predicted_label"] in ("Unnecessary refusal","Illogical completion") for t in turns)
    predictions["max_turn"].append(mx>=.5);predictions["last_turn"].append(last>=.5)
    predictions["review_ambiguous_native_argmax"].append(ambiguous)
    details.append({"sample_id":r["sample_id"],"turn_count":len(turns),"max_turn_p_unsafe":mx,
                    "last_turn_p_unsafe":last,"ambiguous_review":ambiguous})
review=np.array(predictions["review_ambiguous_native_argmax"])
save("safework_aggregation.json",{"role":"posthoc_policy_sensitivity_using_original_record_labels",
    "warning":"last-turn and any-turn target semantics differ; no relabeling or claim of universally better safety",
    "max_turn":classification_metrics(y,predictions["max_turn"]),
    "last_turn":classification_metrics(y,predictions["last_turn"]),
    "decision_disagreement_count":int(np.sum(np.array(predictions["max_turn"])!=predictions["last_turn"])),
    "ambiguous_review_count":int(review.sum()),"coverage_if_ambiguous_review":float((~review).mean()),
    "remaining_metrics":classification_metrics(y[~review],np.asarray(predictions["max_turn"])[~review]) if (~review).any() else None,
    "records":details})
print(json.dumps({"status":"completed_pending_audit","records":513,"output":str(OUT)}))
