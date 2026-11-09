"""Resume unmeasured entries; keep both measured configurations, including failures.

No change to workload, ordering, retry policy or measured settlement implementation.
Later entries add full CLI error capture. Four earlier attempts lack commit status;
their unknown cause is retained, never relabeled as MVCC or discarded.
"""
import json
import os
import re
import time
from pathlib import Path
from run_settlement import (ROOT, OUT, Runner, TracedClient, read_json, read_jsonl,
                            write_json, _source_manifest)

PROGRESS = OUT / "r1-20261030-logged-progress.json"
SOURCE = _source_manifest(ROOT, [Path(__file__).resolve()])
FULL_PREFIX = "r1-20261030-v4"


class FullOutputClient(TracedClient):
    def run_peer(self, arguments, timeout=120):
        command = json.loads(arguments[arguments.index("-c") + 1])["Args"] if "-c" in arguments else []
        if not command or command[0] != "SettleDecision":
            return super().run_peer(arguments, timeout)
        result = None
        error = None
        started = time.time()
        try:
            result = super().run_peer(arguments, timeout)
            return result
        except Exception as exc:
            error = exc
            raise
        finally:
            def text(value):
                return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else (value or "")
            row = {"decision_id": json.loads(command[1])["decisionId"],
                   "started_unix": started,
                   "returncode": result.returncode if result is not None else None,
                   "stdout": result.stdout if result is not None else text(getattr(error, "stdout", None)),
                   "stderr": result.stderr if result is not None else text(getattr(error, "stderr", None)),
                   "exception": repr(error) if error is not None else None}
            with self.guard:
                with self.trace.with_name("settlement_cli_full.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row) + "\n")


class LoggedRunner(Runner):
    def __init__(self, entry):
        if _source_manifest(ROOT, [Path(__file__).resolve()]) != SOURCE:
            raise RuntimeError("Logging wrapper changed during matrix")
        super().__init__(entry, 96, FULL_PREFIX)
        self.client = FullOutputClient(self.repo_root, self.workload_dir,
                                      self.session_dir / "settlement_attempts.jsonl")
        self.provenance["full_output_wrapper"] = SOURCE
        self.provenance["instrumentation_delta"] = "Additional complete settlement CLI outputs; unchanged transaction policy"
        write_json(self.session_dir / "provenance.json", self.provenance)


def attempt_class(row, full_output=""):
    statuses = set(row.get("statuses", []))
    if row.get("outcome_known") and statuses == {"VALID"} and row["returncode"] == 0 and len(row["txids"]) == 1:
        return "valid"
    if row.get("outcome_known") and statuses == {"MVCC_READ_CONFLICT"} and row["returncode"] != 0 and len(row["txids"]) == 1:
        return "mvcc"
    if (row.get("outcome_known") and row["returncode"] != 0 and not statuses and not row["txids"]
            and re.search(r"proposal\s*response\s*payloads?\s+do\s+not\s+match", full_output, re.I)):
        return "proposal_payload_mismatch"
    return "unresolved_attempt"


def validate(entry, prefix, client, allow_retained_gap=False):
    session = OUT / (prefix + "-" + entry["run_id"])
    run = session / "runs" / entry["run_id"]
    summary = read_json(run / "summary.json")
    records = read_jsonl(run / "settlement_records.jsonl")
    attempts = read_jsonl(session / "settlement_attempts.jsonl")
    stage = {r["decision_id"]: r for r in read_jsonl(run / "stage_a.jsonl")}
    source = read_json(ROOT / "results/capacity/j3_manifest.json")
    workload = {r["sample_id"]: r for r in source["records"]}
    selected = read_json(ROOT / "results/capacity/system_selection.json")["sample_ids"]
    assert len(records) == len(stage) == len(selected) == 96
    assert {r["sample_id"] for r in records} == set(selected)
    assert len({r["decision_id"] for r in records}) == 96 and all(r["decision_id"] in stage for r in records)
    assert len(attempts) == summary["attempt_count"]
    full_path = session / "settlement_cli_full.jsonl"
    full = read_jsonl(full_path) if full_path.exists() else []
    grouped = {}
    for row in attempts:
        grouped.setdefault(row["decision_id"], []).append(row)
    full_grouped = {}
    for row in full:
        full_grouped.setdefault(row["decision_id"], []).append(row)
    if prefix == FULL_PREFIX:
        assert len(full) == len(attempts) and set(grouped) == set(full_grouped)
    counts = {k: 0 for k in ("valid", "mvcc", "proposal_payload_mismatch", "unresolved_attempt")}
    details = []
    valid_txids = []
    for record in records:
        rid = record["decision_id"]
        history = grouped[rid]
        complete = full_grouped.get(rid, [])
        if full:
            assert len(complete) == len(history)
        labels = []
        for index, attempt in enumerate(history):
            raw = complete[index] if complete else {}
            if raw:
                assert raw["returncode"] == attempt["returncode"]
                assert abs(raw["started_unix"] - attempt["started_unix"]) < 2
            kind = attempt_class(attempt, raw.get("stdout", "") + raw.get("stderr", ""))
            counts[kind] += 1
            labels.append(kind)
            if kind == "valid": valid_txids.extend(attempt["txids"])
        assert 1 <= len(history) <= 21
        snapshot = client.query("QueryDecisionSnapshot", rid)
        assert snapshot["decisionDigest"] == stage[rid]["decision_digest"]
        if record["success"]:
            assert snapshot["status"] == "Settled"
            assert labels == ["mvcc"] * record["retries"] + ["valid"]
            assert history[-1]["txids"] == [record["commit"]["tx_id"]]
            outcome = "settled"
        else:
            assert snapshot["status"] == "Certified"
            if labels == ["mvcc"] * 21:
                outcome = "retry_exhausted"
            elif labels[-1] == "proposal_payload_mismatch" and all(v == "mvcc" for v in labels[:-1]):
                outcome = "proposal_payload_mismatch"
            elif allow_retained_gap and labels[-1] == "unresolved_attempt" and all(v == "mvcc" for v in labels[:-1]):
                outcome = "retained_unknown_attempt_cause_final_not_settled"
            else:
                raise RuntimeError("Unresolved/new attempt failure: " + rid)
        details.append({"decision_id": rid, "outcome": outcome, "attempt_classes": labels,
                        "final_status": snapshot["status"], "digest_matches": True})
    success = [r for r in records if r["success"]]
    assert counts["valid"] == len(success) == summary["success_count"] == len(set(valid_txids))
    assert len(records) - len(success) == summary["failure_count"]
    assert counts["mvcc"] == summary["mvcc_conflict_attempts"]
    assert sum(counts.values()) == len(attempts)
    if allow_retained_gap:
        # Narrow exception for the already measured second configuration only.
        assert prefix == "r1-20261030-v3" and entry["run_id"] == "lock0-c16-r1"
        assert counts == {"valid": 86, "mvcc": 682, "proposal_payload_mismatch": 0, "unresolved_attempt": 4}
    else:
        assert counts["unresolved_attempt"] == 0
    state = read_json(run / "state_check.json")
    counters = ("unsafeCorrectMicro", "unsafeIncorrectMicro", "safeCorrectMicro", "safeIncorrectMicro")
    for judge in source["committee"]:
        expected = {k: state["before"][judge][k] for k in (*counters, "version")}
        for record in success:
            item = workload[record["sample_id"]]
            label = item["expected_label"]
            key = ("unsafe" if label == "unsafe" else "safe") + ("CorrectMicro" if label == item["predictions"][judge] else "IncorrectMicro")
            expected[key] += 1000000
            expected["version"] += 1
        assert all(state["after"][judge][k] == v for k, v in expected.items())
    assert summary["all_state_checks_pass"]
    assert summary["elapsed_seconds"] > 0
    assert abs(summary["successful_settlements_per_second"] - len(success) / summary["elapsed_seconds"]) < 1e-12
    snapshots = read_json(run / "stack_presence.json")
    required = {"couchdb0", "couchdb1", "orderer.example.com", "peer0.org1.example.com", "peer0.org2.example.com", "ipfs-kubo"}
    assert any(s["phase"] == "during" for s in snapshots)
    assert all(s.get("returncode") == 0 and required <= {v["name"] for v in s.get("services", []) if v["state"] == "running"} for s in snapshots)
    audit = {"measurement_complete": True, "final_state_checks_pass": True,
             "attempt_class_counts": counts, "attempt_cause_audit_complete": counts["unresolved_attempt"] == 0,
             "original_summary_state_retained": summary["state"], "records": details,
             "note": "Final record states checked; four retained v3 attempts remain unclassified, not claimed MVCC/known invalid."}
    write_json(run / "final_state_audit.json", audit)
    return {"measurement_prefix": prefix, "summary": summary,
            "attempt_classes": counts, "attempt_cause_audit_complete": audit["attempt_cause_audit_complete"],
            "retry_exhausted_records": sum(r["outcome"] == "retry_exhausted" for r in details),
            "final_states_verified": True}


def publish(state, entries, results, **extra):
    write_json(PROGRESS, {"state": state, "pid": os.getpid(), "entries": entries,
                         "measured": results, "wrapper_source": SOURCE, **extra})
    lines = ["# 结算锁矩阵：当前状态", "", f"状态：{state}；已测量 {len(results)}/18 组。",
             "", "前两组原数据不变；后续组仅增加完整CLI日志。失败保留、不重跑、不提高20次重试上限。",
             "第二组有4次尝试缺少提交状态／完整错误，原因保留未确定；最终对应记录为Certified，不算结算成功。",
             "吞吐仅为驱动端结算阶段有效提交率，含队列、重试及查询；不是端到端吞吐。",
             "", "| 配置 | 成功/96 | 重试耗尽 | 其他失败 | MVCC尝试 | 未分类尝试 | 有效结算/s |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for item in results:
        s = item["summary"]
        lines.append(f"| {s['entry']['run_id']} | {s['success_count']}/96 | {item['retry_exhausted_records']} | {s['failure_count'] - item['retry_exhausted_records']} | {item['attempt_classes']['mvcc']} | {item['attempt_classes']['unresolved_attempt']} | {s['successful_settlements_per_second']:.4f} |")
    if extra.get("error"): lines += ["", "执行错误：" + extra["error"]]
    lines += ["", "未分类尝试不得并入MVCC计数；矩阵测量完成也不表示全部记录成功或每次尝试原因均已查明。"]
    (OUT / "MATRIX_STATUS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    if PROGRESS.exists(): raise RuntimeError("Logged continuation already exists")
    prior = read_json(OUT / "r1-20261030-continued-progress.json")
    assert prior["state"] == "failed" and len(prior["completed"]) == 1
    entries = prior["entries"]
    assert len(entries) == 18 and prior["failed_entry"] == entries[1]
    query = TracedClient(ROOT, ROOT / "results/capacity", OUT / "unused_read_only_trace.jsonl")
    results = []
    publish("validating_retained_measurements", entries, results)
    try:
        results.append(validate(entries[0], "r1-20261030-v3", query))
        results.append(validate(entries[1], "r1-20261030-v3", query, allow_retained_gap=True))
        publish("running", entries, results)
        for entry in entries[2:]:
            publish("running", entries, results, active_entry=entry)
            runner = LoggedRunner(entry)
            try: runner.execute(entry)
            except RuntimeError as exc:
                if str(exc) != "Settlement completeness/state gate failed; see artifacts": raise
            measured = validate(entry, FULL_PREFIX, runner.client)
            results.append(measured)
            publish("running", entries, results)
        publish("measurements_completed_with_retained_log_gap_pending_final_audit", entries, results)
    except Exception as exc:
        publish("failed", entries, results, error=repr(exc))
        raise


if __name__ == "__main__": main()
