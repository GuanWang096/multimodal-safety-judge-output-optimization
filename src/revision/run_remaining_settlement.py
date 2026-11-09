"""Continue the same fixed matrix, retaining bounded MVCC failures as outcomes.

The measured runner is unchanged. Only orchestration changes: an exhausted
20-retry budget is a negative outcome, not grounds to drop/rerun a configuration.
Unknown submissions or incorrect state still stop execution.
"""
import json
import os
from pathlib import Path
from run_settlement import (ROOT, OUT, Runner, TracedClient, read_json, read_jsonl,
                            write_json, _source_manifest)

PREFIX="r1-20261030-v3"
PROGRESS=OUT/"r1-20261030-continued-progress.json"

def validate_measurement(entry,client):
    session=OUT/(PREFIX+"-"+entry["run_id"])
    run=session/"runs"/entry["run_id"]
    summary=read_json(run/"summary.json")
    outcomes=read_jsonl(run/"settlement_records.jsonl")
    attempts=read_jsonl(session/"settlement_attempts.jsonl")
    original=read_jsonl(run/"stage_a.jsonl")
    digests={r["decision_id"]:r["decision_digest"] for r in original}
    assert len(outcomes)==96 and len({r["decision_id"] for r in outcomes})==96
    assert summary["all_state_checks_pass"] and summary["unknown_outcome_attempts"]==0
    final_failures=[]
    for r in outcomes:
        if r["success"]:continue
        found=[a for a in attempts if a["decision_id"]==r["decision_id"]]
        if ("SettleDecision failed after 20 MVCC retries:" not in r.get("error","") or len(found)!=21
                or not all(a.get("outcome_known") and a["returncode"]!=0 and set(a["statuses"])=={"MVCC_READ_CONFLICT"} for a in found)):
            raise RuntimeError("Failure is not fully observed exhaustion of the unchanged MVCC retry budget")
        snapshot=client.query("QueryDecisionSnapshot",r["decision_id"])
        assert snapshot["status"]=="Certified" and snapshot["decisionDigest"]==digests[r["decision_id"]]
        final_failures.append({"decision_id":r["decision_id"],"status":snapshot["status"],
                               "digest_matches":True,"observed_invalid_attempts":len(found)})
    assert len(final_failures)==summary["failure_count"]
    write_json(run/"bounded_failure_audit.json",{
        "measurement_complete":True,"all_records_settled":not final_failures,
        "failed_final_snapshots":final_failures,"retry_budget_unchanged":20,
        "interpretation":"negative_liveness_under_bounded_retries_not_safety_failure_or_completed_acceptance",
        "original_summary_state_retained":summary["state"]})
    return {**summary,"measurement_complete":True,
            "outcome":"all_settled" if not final_failures else "bounded_retry_exhaustion"}

def main():
    if PROGRESS.exists():raise RuntimeError("Continuation already exists; do not start twice")
    prior=read_json(OUT/(PREFIX+"-progress.json"))
    assert prior["state"]=="failed" and not prior["completed"]
    entries=prior["entries"] if "entries" in prior else read_json(OUT/"r1-20261030-progress.json")["entries"]
    # The failed-state writer drops entries. Reconstruct ONLY from the frozen
    # initial matrix progress file; never use observed outcomes to choose order.
    assert len(entries)==18 and entries[0]==prior["failed_entry"]
    client=TracedClient(ROOT,ROOT/"results/capacity",OUT/"unused_continuation_query_trace.jsonl")
    completed=[validate_measurement(entries[0],client)]
    metadata={"measurement_prefix":PREFIX,"entries":entries,"completed_means":"measurement_finished_not_all_records_settled",
              "orchestrator_sources":_source_manifest(ROOT,[Path(__file__).resolve(),Path(__file__).with_name("run_settlement.py")])}
    write_json(PROGRESS,{**metadata,"state":"running","pid":os.getpid(),"completed":completed})
    print("RETAINED FIRST MEASUREMENT",json.dumps(completed[0]),flush=True)
    for entry in entries[1:]:
        try:
            runner=Runner(entry,96,PREFIX)
            try:runner.execute(entry)
            except RuntimeError as exc:
                if str(exc)!="Settlement completeness/state gate failed; see artifacts":raise
            measured=validate_measurement(entry,runner.client)
            completed.append(measured)
            write_json(PROGRESS,{**metadata,"state":"running","pid":os.getpid(),"completed":completed})
        except Exception as exc:
            write_json(PROGRESS,{**metadata,"state":"failed","pid":os.getpid(),"failed_entry":entry,
                                 "error":str(exc),"completed":completed})
            raise
    write_json(PROGRESS,{**metadata,"state":"completed_pending_audit","completed":completed})

if __name__=="__main__":main()
