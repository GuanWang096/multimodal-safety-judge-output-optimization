"""Real Fabric settlement lock ablation. Run inside Ubuntu WSL.

Reuses the archived MJ5 96-record capacity workload, not the 513 accuracy set.
The v1 deterministic test credentials remain in use in this performance fixture.
RGG runs without stopping Fabric/IPFS. Stage durations are separate, not summed E2E.
"""
from __future__ import annotations
import argparse
import concurrent.futures
import contextlib
import json
import os
import random
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from functools import lru_cache
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
WF = ROOT / "src/multijudge_workflows"
OUT = ROOT / "results/reproduced/revision/system"
sys.path.insert(0,str(WF))
from run_mj5_formal import FormalRunner, read_json, read_jsonl, write_json, run_parallel_chunks, ResourceSampler
from mj5_fabric_client import MJ5FabricClient
from mj5_rgg_bridge import extract_protocol_certificates
import run_mj5_formal as original_runner

# Sources/workload are immutable during a matrix. Avoid re-reading the same
# vendored source tree over the Windows/WSL filesystem for every configuration.
_source_manifest = original_runner.source_manifest
@lru_cache(maxsize=4)
def cached_manifest(repo_root, roots):
    return _source_manifest(repo_root, list(roots))
original_runner.source_manifest = lambda repo_root, roots: cached_manifest(repo_root, tuple(roots))
REVISION_SOURCE = _source_manifest(ROOT, [Path(__file__).resolve()])

IMAGE="zte-rggpbft:revision-r1-20261030"

class TracedClient(MJ5FabricClient):
    def __init__(self,root,workload,trace):
        super().__init__(root,workload)
        self.trace=trace
        self.guard=threading.Lock()
    def run_peer(self, arguments, timeout=120):
        started=time.perf_counter()
        command=None
        if "-c" in arguments:
            command=json.loads(arguments[arguments.index("-c")+1])["Args"]
        is_settlement=command is not None and command[0]=="SettleDecision"
        result=None;error=None
        try:
            result=super().run_peer(arguments,timeout)
            return result
        except (subprocess.TimeoutExpired,OSError) as exc:
            error=exc
            raise
        finally:
            if is_settlement:
                def as_text(value):
                    return value.decode("utf-8",errors="replace") if isinstance(value,bytes) else (value or "")
                if result is not None:
                    combined=result.stdout+result.stderr
                else:
                    combined=as_text(getattr(error,"stdout",None))+as_text(getattr(error,"stderr",None))
                payload=json.loads(command[1])
                row={"decision_id":payload["decisionId"],"started_unix":time.time()-(time.perf_counter()-started),
                     "elapsed_ms":1000*(time.perf_counter()-started),"returncode":result.returncode if result is not None else None,
                     "statuses":re.findall(r"committed with status \(([^)]+)\)",combined),
                     "mvcc_conflict":("MVCC_READ_CONFLICT" in combined) if combined else None,
                     "txids":sorted(set(re.findall(r"txid \[([0-9a-fA-F]{64})\]",combined))),
                     "exception_type":type(error).__name__ if error is not None else None,
                     "outcome_known":result is not None,
                     "error_tail":combined[-1200:] if error is not None or result.returncode else None}
                with self.guard:
                    with self.trace.open("a",encoding="utf-8") as handle:handle.write(json.dumps(row)+"\n")

class Runner(FormalRunner):
    def __init__(self, entry, count, prefix):
        if _source_manifest(ROOT,[Path(__file__).resolve()])!=REVISION_SOURCE:
            raise RuntimeError("Runner source changed during matrix; stop before a new configuration")
        session=prefix+"-"+entry["run_id"]
        args=argparse.Namespace(repo_root=ROOT,workload=ROOT/"results/capacity",
                                output=OUT,qualification=False,session_id=session)
        if (OUT/session).exists():raise RuntimeError(f"Refusing to reuse session {session}")
        super().__init__(args)
        self.entries=[entry]
        self.selected_ids=self.selected_ids[:count]
        self.client=TracedClient(self.repo_root,self.workload_dir,self.session_dir/"settlement_attempts.jsonl")
        self.provenance["revision_source"] = REVISION_SOURCE
        self.provenance["revision_image_id"] = subprocess.check_output(
            ["docker","image","inspect",IMAGE,"--format","{{.Id}}"],text=True).strip()
        write_json(self.session_dir/"provenance.json",self.provenance)
        write_json(self.session_dir/"session_manifest.json",{
            "schema":"revision-settlement-v1","session_id":session,"entry":entry,
            "sample_ids":self.selected_ids,"sample_count":count,"consensus_image":IMAGE,
            "workload":"archived_MJ5_capacity_not_513_confirmation",
            "key_mode":"original_public_deterministic_fixture_not_production_keys",
            "confirmation_mode":"single_driver_ACK_not_independent_organizations",
            "stack_policy":"Fabric_IPFS_remain_running_during_consensus",
            "timing":"settlement_stage_only_not_end_to_end"})
    def consensus(self,entry):
        directory=self.entry_dir(entry)
        stage_a=sorted(read_jsonl(directory/"stage_a.jsonl"),key=lambda r:r["index"])
        inputs=[{"decision_id":r["decision_id"],"digest":r["decision_digest"]} for r in stage_a]
        dest=directory/"rgg";dest.mkdir()
        write_json(dest/"decision_digests.json",{"schema":"mj5-rgg-input-v1","entries":inputs})
        command=[sys.executable,"-B",str(self.rgg_source/"run_v2.py"),"--mode","rgg","--nodes","16",
                 "--groups","4","--rounds",str(len(inputs)),"--delay-ms","5","--round-timeout","15",
                 "--view-timeout","1","--reputation-order",",".join(map(str,self.client.registrations["validator_order"])),
                 "--seed","20260705","--run-dir",str(dest),"--image",IMAGE,"--skip-build"]
        print("CONSENSUS",entry["run_id"],flush=True)
        start=time.perf_counter()
        stop=threading.Event();stack_snapshots=[]
        def stack_snapshot(phase):
            observed=subprocess.run(["docker","ps","--format","{{json .}}"],capture_output=True,text=True,timeout=15)
            services=[json.loads(line) for line in observed.stdout.splitlines() if line.strip()]
            stack_snapshots.append({"phase":phase,"time_unix":time.time(),"returncode":observed.returncode,
                                    "services":[{"name":s["Names"],"state":s["State"],"status":s["Status"]} for s in services]})
        def watch_stack():
            while not stop.wait(5):
                try:stack_snapshot("during")
                except Exception as exc:stack_snapshots.append({"phase":"during","error":str(exc),"time_unix":time.time()})
        stack_snapshot("before")
        watcher=threading.Thread(target=watch_stack,daemon=True);watcher.start()
        try:
            with (directory/"rgg_runner.log").open("w",encoding="utf-8") as handle:
                subprocess.run(command,cwd=self.rgg_source,stdout=handle,stderr=subprocess.STDOUT,check=True,timeout=3600)
        finally:
            stop.set();watcher.join(timeout=20)
            stack_snapshot("after")
            write_json(directory/"stack_presence.json",stack_snapshots)
        summary=read_json(dest/"summary.json")
        if summary["driver_success_count"]!=len(inputs) or summary["conflicting_commit_count"] or summary["safety_violation_events"]:
            raise RuntimeError("Consensus completion/safety gate failed")
        certs=extract_protocol_certificates(dest/"events.jsonl",inputs,self.client.registrations["validator_order"],self.rgg_source)
        write_json(directory/"protocol_certificates.json",certs)
        write_json(directory/"rgg_status.json",{"state":"completed","records":len(inputs),"wall_seconds":time.perf_counter()-start})
        return stage_a,{c["decision_id"]:c for c in certs}
    def execute(self,entry):
        self.run_stage_a()
        stage_a,certs=self.consensus(entry)
        print("CERTIFY",entry["run_id"],flush=True)
        def certify(row):
            snapshot=self.client.query("QueryDecisionSnapshot",row["decision_id"])
            cert=certs[row["decision_id"]]
            ms,commit=self.client.certify_with_details(snapshot,cert["protocol_certificate"],cert["view"],cert["sequence"])
            now=self.client.query("QueryDecisionSnapshot",row["decision_id"])
            if now["status"]!="Certified" or now["decisionDigest"]!=cert["digest"]:
                raise RuntimeError("Certification binding failure")
            return {"decision_id":row["decision_id"],"elapsed_ms":ms,"commit":commit}
        certified=run_parallel_chunks(stage_a,64,8,certify)
        directory=self.entry_dir(entry)
        write_json(directory/"certification.json",certified)
        judges=read_json(self.workload_dir/"j3_manifest.json")["committee"]
        before={j:self.client.query("QueryJudge",self.judge_ids[j]) for j in judges}
        guard=threading.Lock()
        outcomes=[]
        output_guard=threading.Lock()
        def settle(row):
            start=time.perf_counter()
            try:
                with guard if entry["lock_enabled"] else contextlib.nullcontext():
                    queue_ms=(time.perf_counter()-start)*1000
                    latency,retries,commit=self.client.settle_with_commit_details(row["decision_id"],row["expected_label"])
                snap=self.client.query("QueryDecisionSnapshot",row["decision_id"])
                assert snap["status"]=="Settled" and snap["decisionDigest"]==row["decision_digest"]
                result={"decision_id":row["decision_id"],"sample_id":row["sample_id"],"success":True,
                        "queue_ms":queue_ms,"settle_ms":latency,"retries":retries,"commit":commit,
                        "total_ms":1000*(time.perf_counter()-start)}
            except Exception as exc:
                result={"decision_id":row["decision_id"],"sample_id":row["sample_id"],"success":False,
                        "error":str(exc),"total_ms":1000*(time.perf_counter()-start)}
            with output_guard:
                with (directory/"settlement_records.jsonl").open("a",encoding="utf-8") as handle:
                    handle.write(json.dumps(result)+"\n")
            return result
        print("SETTLE",entry["run_id"],flush=True)
        start=time.perf_counter()
        with ResourceSampler() as sampler:
            outcomes=run_parallel_chunks(stage_a,64,entry["concurrency"],settle)
            settlement_finished=time.perf_counter()
        sampler_finished=time.perf_counter()
        elapsed=settlement_finished-start
        after={j:self.client.query("QueryJudge",self.judge_ids[j]) for j in judges}
        workload={r["sample_id"]:r for r in self.records_for_entry(entry)}
        success=[r for r in outcomes if r["success"]]
        counters=("unsafeCorrectMicro","unsafeIncorrectMicro","safeCorrectMicro","safeIncorrectMicro")
        checks={}
        for j in judges:
            expected={k:before[j][k] for k in counters};expected["version"]=before[j]["version"]
            for item in success:
                r=workload[item["sample_id"]];gold=r["expected_label"];prediction=r["predictions"][j]
                field=("unsafe" if gold=="unsafe" else "safe")+("CorrectMicro" if gold==prediction else "IncorrectMicro")
                expected[field]+=1000000;expected["version"]+=1
            checks[j]={"expected":expected,"actual":{k:after[j][k] for k in expected},
                       "matches":all(after[j][k]==v for k,v in expected.items())}
        attempts=read_jsonl(self.session_dir/"settlement_attempts.jsonl")
        write_json(directory/"state_check.json",{"before":before,"after":after,"checks":checks})
        write_json(directory/"settlement_resources.json",sampler.samples)
        summary={"entry":entry,"record_count":len(stage_a),"success_count":len(success),
                 "failure_count":len(outcomes)-len(success),"elapsed_seconds":elapsed,
                 "sampler_teardown_seconds_excluded":sampler_finished-settlement_finished,
                 "timing_scope":"driver_settlement_phase_including_queue_retries_and_final_snapshot_queries_excluding_sampler_teardown",
                 "successful_settlements_per_second":len(success)/elapsed,
                 "attempt_count":len(attempts),"mvcc_conflict_attempts":sum(r.get("mvcc_conflict") is True for r in attempts),
                 "unknown_outcome_attempts":sum(not r.get("outcome_known",True) for r in attempts),
                 "all_state_checks_pass":all(c["matches"] for c in checks.values()),
                 "state":"completed" if len(success)==len(outcomes) and all(c["matches"] for c in checks.values()) else "failed"}
        write_json(directory/"summary.json",summary)
        print(json.dumps(summary),flush=True)
        if summary["state"]!="completed":raise RuntimeError("Settlement completeness/state gate failed; see artifacts")
        return summary

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--smoke",action="store_true")
    parser.add_argument("--prefix",default="r1-20261030")
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    if args.smoke:
        entries=[{"run_id":"smoke-unlocked-c2","judge_count":3,"concurrency":2,"repeat":0,
                  "sample_count":4,"execution_batch_count":1,"lock_enabled":False}]
        prefix=args.prefix+"-smoke"
    else:
        entries=[]
        for repeat in (1,2,3):
            block=[{"run_id":f"lock{int(lock)}-c{c}-r{repeat}","judge_count":3,"concurrency":c,
                    "repeat":repeat,"sample_count":96,"execution_batch_count":2,"lock_enabled":lock}
                   for lock in (True,False) for c in (1,8,16)]
            random.Random(20261030+repeat).shuffle(block)
            entries.extend(block)
        prefix=args.prefix
    progress=OUT/(prefix+"-progress.json")
    if progress.exists():raise RuntimeError("Progress already exists; do not silently overwrite runs")
    summaries=[]
    write_json(progress,{"state":"running","pid":os.getpid(),"entries":entries,"completed":summaries})
    for entry in entries:
        try:
            summary=Runner(entry,entry["sample_count"],prefix).execute(entry)
            summaries.append(summary)
            write_json(progress,{"state":"running","pid":os.getpid(),"entries":entries,"completed":summaries})
        except Exception as exc:
            write_json(progress,{"state":"failed","pid":os.getpid(),"failed_entry":entry,"error":str(exc),"completed":summaries})
            raise
    write_json(progress,{"state":"completed_pending_audit","entries":entries,"completed":summaries})

if __name__=="__main__":main()
