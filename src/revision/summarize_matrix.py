"""Derive matrix tables from archived attempts and records; never rerun workloads.

Writes new descriptive artifacts only. This mechanical extraction is not the
independent semantic integrity audit. Mixed instrumentation is never pooled into
a matched timing estimate; its descriptive totals are explicitly separated.
"""
import collections
import csv
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "results/revision"
SYSTEM = BASE / "system"
DEST = ROOT / "results/reproduced/revision/system"
INPUTS = {}


def read(path, lines=False):
    raw = path.read_bytes()
    INPUTS[path.relative_to(ROOT).as_posix()] = hashlib.sha256(raw).hexdigest()
    return [json.loads(v) for v in raw.splitlines() if v.strip()] if lines else json.loads(raw)


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def summarize():
    progress = read(SYSTEM / "r1-20261030-logged-progress.json")
    assert len(progress["measured"]) == len(progress["entries"]) == 18
    workload = ROOT / "results/capacity"
    selection = read(workload / "system_selection.json")["sample_ids"]
    source = read(workload / "j3_manifest.json")
    gold = {r["sample_id"]: r for r in source["records"]}
    extracted = []
    problems = []
    for item in progress["measured"]:
        entry = item["summary"]["entry"]
        session = SYSTEM / (item["measurement_prefix"] + "-" + entry["run_id"])
        run = session / "runs" / entry["run_id"]
        summary = read(run / "summary.json")
        records = read(run / "settlement_records.jsonl", lines=True)
        attempts = read(session / "settlement_attempts.jsonl", lines=True)
        raw_path = session / "settlement_cli_full.jsonl"
        full = read(raw_path, lines=True) if raw_path.exists() else []
        state = read(run / "state_check.json")
        final = read(run / "final_state_audit.json")
        consensus = read(run / "rgg/summary.json")
        stage = read(run / "stage_a.jsonl", lines=True)
        read(session / "provenance.json")
        by_id = collections.defaultdict(list)
        by_full = collections.defaultdict(list)
        for a in attempts: by_id[a["decision_id"]].append(a)
        for a in full: by_full[a["decision_id"]].append(a)
        kinds = collections.Counter()
        reasons = collections.Counter()
        txids = []
        for record in records:
            history = by_id[record["decision_id"]]
            detailed = by_full[record["decision_id"]]
            labels = []
            for index, a in enumerate(history):
                statuses = set(a.get("statuses", []))
                if a.get("returncode") == 0 and statuses == {"VALID"} and len(a["txids"]) == 1:
                    kind = "valid"
                    txids.extend(a["txids"])
                elif a.get("returncode") not in (None, 0) and statuses == {"MVCC_READ_CONFLICT"} and len(a["txids"]) == 1:
                    kind = "mvcc"
                elif index < len(detailed) and re.search(r"proposal\s*response\s*payloads?\s+do\s+not\s+match", detailed[index]["stdout"] + detailed[index]["stderr"], re.I):
                    kind = "proposal_payload_mismatch"
                else:
                    kind = "unclassified"
                labels.append(kind)
                kinds[kind] += 1
            if record["success"]:
                reasons["settled"] += 1
                if labels != ["mvcc"] * record["retries"] + ["valid"]:
                    problems.append(entry["run_id"] + ": successful attempt history mismatch")
            elif labels == ["mvcc"] * 21:
                reasons["retry_exhausted"] += 1
            elif labels[-1] == "proposal_payload_mismatch":
                reasons["proposal_payload_mismatch"] += 1
            else:
                reasons["unclassified"] += 1
        checks = {
            "exact_sample_membership": len(records) == 96 and {r["sample_id"] for r in records} == set(selection),
            "unique_decisions": len({r["decision_id"] for r in records}) == 96,
            "stage_membership": len(stage) == 96 and {r["sample_id"] for r in stage} == set(selection),
            "success_count": reasons["settled"] == summary["success_count"] == kinds["valid"] == len(set(txids)),
            "failure_count": len(records) - reasons["settled"] == summary["failure_count"],
            "attempt_accounting": sum(kinds.values()) == len(attempts) == summary["attempt_count"],
            "mvcc_accounting": kinds["mvcc"] == summary["mvcc_conflict_attempts"],
            "positive_time": summary["elapsed_seconds"] > 0,
            "goodput_arithmetic": math.isclose(summary["successful_settlements_per_second"], reasons["settled"] / summary["elapsed_seconds"], abs_tol=1e-12),
            "final_snapshot_membership": {r["decision_id"] for r in final["records"]} == {r["decision_id"] for r in records},
            "final_statuses": all(r["digest_matches"] and r["final_status"] == ("Settled" if r["outcome"] == "settled" else "Certified") for r in final["records"]),
        }
        for judge in source["committee"]:
            fields = ("unsafeCorrectMicro", "unsafeIncorrectMicro", "safeCorrectMicro", "safeIncorrectMicro", "version")
            expected = {f: state["before"][judge][f] for f in fields}
            for record in records:
                if not record["success"]: continue
                r = gold[record["sample_id"]]
                field = ("unsafe" if r["expected_label"] == "unsafe" else "safe") + ("CorrectMicro" if r["expected_label"] == r["predictions"][judge] else "IncorrectMicro")
                expected[field] += 1000000
                expected["version"] += 1
            checks["state_" + judge] = all(expected[f] == state["after"][judge][f] for f in fields)
        problems.extend(entry["run_id"] + ": " + k for k, v in checks.items() if not v)
        extracted.append({
            "run_id": entry["run_id"], "instrumentation": item["measurement_prefix"],
            "lock_enabled": entry["lock_enabled"], "concurrency": entry["concurrency"], "repeat": entry["repeat"],
            "record_count": len(records), "settled": reasons["settled"], "failed": len(records) - reasons["settled"],
            "retry_exhausted_records": reasons["retry_exhausted"], "proposal_mismatch_records": reasons["proposal_payload_mismatch"],
            "unclassified_failure_records": reasons["unclassified"], "attempt_count": len(attempts),
            "mvcc_attempts": kinds["mvcc"], "proposal_mismatch_attempts": kinds["proposal_payload_mismatch"],
            "unclassified_attempts": kinds["unclassified"], "elapsed_seconds": summary["elapsed_seconds"],
            "effective_settlements_per_second": reasons["settled"] / summary["elapsed_seconds"],
            "consensus_completed": consensus["driver_success_count"],
            "recorded_conflicting_commits": consensus["conflicting_commit_count"],
            "recorded_safety_violations": consensus["safety_violation_events"],
            "extraction_checks": checks, "source_session": session.relative_to(ROOT).as_posix()})
    totals = {key: sum(r[key] for r in extracted) for key in (
        "record_count", "settled", "failed", "retry_exhausted_records", "proposal_mismatch_records",
        "unclassified_failure_records", "attempt_count", "mvcc_attempts", "proposal_mismatch_attempts",
        "unclassified_attempts", "consensus_completed", "recorded_conflicting_commits", "recorded_safety_violations")}
    groups = []
    for instrument in sorted({r["instrumentation"] for r in extracted}):
        for lock in (False, True):
            for concurrency in (1, 8, 16):
                selected = [r for r in extracted if r["instrumentation"] == instrument and r["lock_enabled"] == lock and r["concurrency"] == concurrency]
                if not selected: continue
                rates = [r["effective_settlements_per_second"] for r in selected]
                groups.append({"instrumentation": instrument, "lock_enabled": lock, "concurrency": concurrency,
                               "n_runs": len(selected), "run_ids": [r["run_id"] for r in selected],
                               "settled_per_run": [r["settled"] for r in selected],
                               "goodput_mean": statistics.mean(rates), "goodput_min": min(rates), "goodput_max": max(rates)})
    result = {"generated_at": dt.datetime.now(dt.timezone.utc).isoformat(), "measurement_runs": 18,
              "distinct_workload_records": len(selection), "total_record_executions": totals["record_count"],
              "totals_descriptive_only": totals, "per_run": extracted, "groups_by_instrumentation": groups,
              "mechanical_check_failures": problems,
              "semantic_audit_status": "separate_reviewer_required",
              "timing_scope": "settlement_driver_only_including_queue_retries_queries_and_logging_excluding_sampler_teardown",
              "limitations": ["96 archived MJ5 capacity records reused, not 1728 independent safety examples",
                              "v3 and v4 instrumentation differ; no matched pooled timing claim",
                              "four v3 attempts remain unclassified; final outcome is not proof of per-attempt cause",
                              "single-host deterministic-key single-controller fixture; no production key isolation claim"]}
    DEST.mkdir(parents=True, exist_ok=True)
    save(DEST / "matrix_extraction.json", result)
    fields = [k for k in extracted[0] if k != "extraction_checks"]
    with (DEST / "matrix_runs.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: v for k, v in r.items() if k in fields} for r in extracted)
    INPUTS[Path(__file__).relative_to(ROOT).as_posix()] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    save(DEST / "extraction_inputs.json", {"algorithm": "sha256", "files": INPUTS})
    print(json.dumps({"totals": totals, "mechanical_check_failures": problems}, ensure_ascii=False))


if __name__ == "__main__": summarize()
