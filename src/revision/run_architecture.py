"""Bounded, staged-endpoint architecture comparison; run in Ubuntu WSL.

Uses actual IPFS/Fabric and original RGG node code. Does not change the protocol,
deploy chaincode, reset the ledger, settle labels, or infer equivalent security.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
import traceback
import urllib.request
from pathlib import Path
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives import serialization

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
WF = ROOT / "src/multijudge_workflows"
SRC = ROOT / "src/rggpbft"
WORK = ROOT / "results/capacity"
OUT = ROOT / "results/reproduced/revision/system"
sys.path[:0] = [str(WF), str(SRC)]
from mj5_fabric_client import MJ5FabricClient
from mj5_rgg_bridge import extract_protocol_certificates
from run_v2 import compose_definition, ready_node_ids, docker_cli_path, read_events

IMAGE = "sha256:1d16f1669bf445755a2342b22d02b329db140472c36606a797da2d45218b7355"
ARMS = ["signed_log", "fabric", "full"]


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append(path, value):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")


def command(args, trace=None, timeout=120):
    started = time.perf_counter()
    result = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    if trace:
        append(trace, {"command":args, "returncode":result.returncode,
                       "stdout":result.stdout, "stderr":result.stderr,
                       "elapsed_ms":1000*(time.perf_counter()-started)})
    if result.returncode:
        raise RuntimeError(f"command failed: {args}: {result.stderr}")
    return result.stdout.strip()


def records(count):
    ids = read(WORK / "system_selection.json")["sample_ids"][:count]
    by_id = {r["sample_id"]:r for r in read(WORK / "j3_manifest.json")["records"]}
    selected = [by_id[s] for s in ids]
    assert len(selected) == count and len(set(ids)) == count
    for r in selected:
        assert hashlib.sha256((WORK/r["bundle_path"]).read_bytes()).hexdigest() == r["evidence_sha256"]
    return selected


def preflight():
    client = MJ5FabricClient(ROOT, WORK)
    definition = client.run_peer(["lifecycle","chaincode","querycommitted", "--channelID","trustchannel","--name","tce"])
    assert definition.returncode == 0 and "Version: 4.1, Sequence: 5" in definition.stdout
    actual = command(["docker","image","inspect",IMAGE,"--format","{{.Id}}"])
    assert actual == IMAGE
    services = command(["docker","ps","--format","{{.Names}}"])
    for name in ["ipfs-kubo","peer0.org1.example.com","peer0.org2.example.com","orderer.example.com"]:
        assert name in services.splitlines(), name
    request = urllib.request.Request("http://localhost:5001/api/v0/version", data=b"", method="POST")
    with urllib.request.urlopen(request, timeout=20) as response:
        ipfs = json.loads(response.read())
    selected = records(16)
    return {"preflight_pass":True, "python":platform.python_version(), "platform":platform.platform(),
            "image":actual, "chaincode":definition.stdout, "services":services.splitlines(),
            "ipfs":ipfs, "checked_records":len(selected)}


class TracedClient(MJ5FabricClient):
    def __init__(self, directory):
        super().__init__(ROOT, WORK)
        self.directory = directory

    def run_peer(self, arguments, timeout=120):
        started = time.perf_counter()
        result = None
        error = None
        try:
            result = super().run_peer(arguments, timeout)
            return result
        except Exception as exc:
            error = repr(exc)
            raise
        finally:
            append(self.directory / "peer_calls.jsonl", {
                "arguments":arguments, "elapsed_ms":1000*(time.perf_counter()-started),
                "returncode":result.returncode if result is not None else None,
                "stdout":result.stdout if result is not None else "",
                "stderr":result.stderr if result is not None else "", "exception":error})


class Runner:
    def __init__(self, args):
        self.args = args
        self.directory = OUT / args.session
        self.directory.mkdir(parents=True, exist_ok=False)
        self.trace = self.directory / "docker_calls.jsonl"
        self.client = TracedClient(self.directory)
        self.selected = records(args.count)
        self.rows = []
        self.inputs = []
        self.project = "revarch" + hashlib.sha256(args.session.encode()).hexdigest()[:10]
        self.rgg = self.directory / "rgg"
        self.rgg.mkdir()
        self.compose = self.rgg / "compose.json"
        self.compose_cli = docker_cli_path(self.compose)
        self.owner = Ed25519PrivateKey.generate()
        self.public = self.owner.public_key()
        self.previous = "0"*64
        self.log = self.directory / "signed_log.jsonl"
        self.log.touch(exist_ok=False)
        with self.log.open("ab") as handle:
            handle.flush(); os.fsync(handle.fileno())
        self.source_hashes = self.sources()

    def sources(self):
        paths = [Path(__file__), ROOT/"README.md",
                 WF/"mj5_fabric_client.py", WF/"mj5_rgg_bridge.py", WF/"mj5_common.py"]
        paths += [SRC/n for n in ["protocol.py","node_v2.py","view_change.py","view_runtime.py",
                                 "driver_v2.py","collector_v2.py","run_v2.py","grouping.py","fault_policy.py"]]
        paths += [WORK/n for n in ["j3_manifest.json","system_selection.json","registrations.json"]]
        return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}

    def progress(self, state, **details):
        write(self.directory/"progress.json", {"state":state,"records_complete":len(self.rows),
              "expected_records":3*self.args.count*self.args.repeats,"updated_unix":time.time(),**details})

    def initialize(self):
        write(self.directory/"environment.json", preflight())
        self.judges = self.client.ensure_registrations(self.args.session+"::")
        write(self.directory/"manifest.json", {"session":self.args.session,"count":self.args.count,
              "repeats":self.args.repeats,"sample_ids":[r["sample_id"] for r in self.selected],
              "arms":ARMS,"order":"cyclic_rotation_by_block","source_sha256":self.source_hashes,
              "key_mode":"original_public_deterministic_protocol_fixture; random_in_memory_log_owner",
              "owner_public_key":self.public.public_bytes(serialization.Encoding.Raw,serialization.PublicFormat.Raw).hex(),
              "judge_ids":self.judges,"timing":"warm_services_IPFS_add_to_endpoint_receipt_readback_including_CLI_capture_and_full_docker_exec",
              "not_measured":["independent_organization_control","power_loss_recovery","production_security","label_settlement","GPU_inference"]})
        self.before = {j:self.client.query("QueryJudge",j) for j in self.judges.values()}
        write(self.directory/"judges_before.json", self.before)
        cfg = argparse.Namespace(mode="rgg",nodes=16,groups=4,rounds=self.args.count*self.args.repeats,
              round_timeout=15,delay_ms=5,view_timeout=1,fault_mode="none",fault_scenario="none",
              fault_nodes="",fault_delay_ms=100,reputation_order=",".join(map(str,self.client.registrations["validator_order"])),
              seed=20260705,image=IMAGE)
        compose = compose_definition(cfg, docker_cli_path(self.rgg), include_driver=False)
        for service in compose["services"].values():
            service["environment"]["HARD_TIMEOUT_SECONDS"] = "3600"
        write(self.compose, compose)
        started = time.perf_counter()
        command(["docker","compose","-p",self.project,"-f",self.compose_cli,"up","-d"],self.trace)
        deadline = time.monotonic()+40
        while len(ready_node_ids(self.rgg/"events.jsonl")) != 16 and time.monotonic()<deadline:
            time.sleep(.2)
        assert len(ready_node_ids(self.rgg/"events.jsonl")) == 16, "ready barrier"
        write(self.directory/"startup.json", {"seconds_excluded":time.perf_counter()-started,"ready_nodes":16})

    def log_record(self, record, decision, cid):
        body = {"decisionId":decision,"sampleId":record["sample_id"],"evidenceCid":cid,
                "evidenceSha256":record["evidence_sha256"],"provisionalLabel":record["provisional_label"],
                "posteriorUnsafePpm":record["posterior_unsafe_ppm"],
                "judgeOutputs":self.client.signed_outputs(record,decision,self.judges),"previous":self.previous}
        signed = {"body":body,"signature":self.owner.sign(canonical(body)).hex()}
        encoded = canonical(signed)+b"\n"
        with self.log.open("ab") as handle:
            offset = handle.tell()
            handle.write(encoded); handle.flush(); os.fsync(handle.fileno())
        with self.log.open("rb") as handle:
            handle.seek(offset)
            fetched = handle.readline()
        assert fetched == encoded
        self.public.verify(bytes.fromhex(signed["signature"]),canonical(signed["body"]))
        self.previous = hashlib.sha256(encoded).hexdigest()
        return {"status":"LocalDurableReceipt","offset":offset,"record_sha256":self.previous}

    def full_consensus(self, snapshot):
        self.inputs.append({"decision_id":snapshot["decisionId"],"digest":snapshot["decisionDigest"]})
        write(self.rgg/"decision_digests.json", {"entries":self.inputs})
        seq = len(self.inputs)-1
        program = f"import driver_v2 as d; ok=d.run_round({seq}); raise SystemExit(0 if ok else 1)"
        start = time.perf_counter()
        command(["docker","exec","-e",f"N_ROUNDS={len(self.inputs)}",self.project+"-collector-1",
                 "python","-B","-c",program], self.trace, timeout=45)
        certs = extract_protocol_certificates(self.rgg/"events.jsonl",self.inputs,
                    self.client.registrations["validator_order"],SRC)
        cert = certs[-1]
        assert len(cert["signer_node_ids"]) >= 3
        append(self.directory/"certificates.jsonl", cert)
        events = read_events(self.rgg/"events.jsonl")
        assert not any(e["type"] in {"SAFETY_VIOLATION","NODE_ERROR"} for e in events)
        driver = [e["data"] for e in events if e["type"]=="DRIVER_RESULT" and e["data"]["sequence"]==seq]
        assert len(driver)==1 and driver[0]["success"] and driver[0]["observed_commit_count"]==16
        return cert, 1000*(time.perf_counter()-start), driver[0]

    def one(self, arm, repeat, index, record):
        decision = f"{self.args.session}::{arm}::r{repeat}::{index:04d}"
        bundle = (WORK/record["bundle_path"]).read_bytes()
        row = {"arm":arm,"repeat":repeat,"index":index,"sample_id":record["sample_id"],
               "decision_id":decision,"evidence_sha256":record["evidence_sha256"],"bytes":len(bundle)}
        started = time.perf_counter()
        try:
            cid, add_ms = self.client.ipfs_add(bundle)
            fetched = self.client.ipfs_cat(cid)
            assert fetched == bundle
            row.update(cid=cid,ipfs_add_ms=add_ms,ipfs_roundtrip_ms=1000*(time.perf_counter()-started))
            if arm == "signed_log":
                row["receipt"] = self.log_record(record,decision,cid)
                row["status"] = row["receipt"]["status"]
            else:
                snapshot, ms, receipt = self.client.freeze_with_details(record,decision,cid,self.judges,int(time.time())+86400)
                row.update(freeze_invoke_ms=ms,freeze_receipt=receipt)
                if arm == "full":
                    row["ack_receipts"] = []
                    for node in self.client.registrations["validator_order"][:2]:
                        ms, receipt = self.client.submit_vote_with_details(snapshot,node,"ACK")
                        row["ack_receipts"].append({"elapsed_ms":ms,**receipt})
                    snapshot = self.client.query("QueryDecisionSnapshot",decision)
                    assert snapshot["status"] == "CommitteeConfirmed"
                    cert, ms, driver = self.full_consensus(snapshot)
                    row.update(rgg_driver_and_extract_ms=ms,rgg_driver=driver)
                    ms, receipt = self.client.certify_with_details(snapshot,cert["protocol_certificate"],cert["view"],cert["sequence"])
                    row.update(certify_invoke_ms=ms,certify_receipt=receipt)
                    final = self.client.query("QueryDecisionSnapshot",decision)
                    assert final["decisionDigest"] == snapshot["decisionDigest"] == cert["digest"]
                    snapshot = final
                assert snapshot["status"] == ("Certified" if arm=="full" else "SnapshotFrozen")
                assert snapshot["evidenceCid"] == cid and snapshot["evidenceSha256"] == record["evidence_sha256"]
                assert snapshot["posteriorUnsafePpm"] == record["posterior_unsafe_ppm"]
                assert snapshot["provisionalLabel"] == record["provisional_label"]
                row.update(status=snapshot["status"],decision_digest=snapshot["decisionDigest"],snapshot=snapshot)
            row.update(success=True,total_ms=1000*(time.perf_counter()-started))
        except Exception as exc:
            row.update(success=False,error=repr(exc),total_ms=1000*(time.perf_counter()-started))
            append(self.directory/"records.jsonl",row)
            raise
        self.rows.append(row)
        append(self.directory/"records.jsonl",row)
        self.progress("running",arm=arm,repeat=repeat,index=index)
        print(json.dumps({k:row[k] for k in ["arm","repeat","index","status","total_ms"]}),flush=True)

    def audit(self, repeat):
        checks = []
        for row in self.rows:
            if row["arm"] != "signed_log":
                snapshot = self.client.query("QueryDecisionSnapshot",row["decision_id"])
                checks.append(snapshot["status"] == row["status"] and snapshot["decisionDigest"] == row["decision_digest"])
        previous = "0"*64
        with self.log.open("rb") as handle:
            for line in handle:
                entry=json.loads(line)
                assert entry["body"]["previous"] == previous
                self.public.verify(bytes.fromhex(entry["signature"]),canonical(entry["body"]))
                previous=hashlib.sha256(line).hexdigest()
        after={j:self.client.query("QueryJudge",j) for j in self.judges.values()}
        result={"repeat":repeat,"snapshot_checks":len(checks),"snapshots_match":all(checks),
                "judge_state_unchanged":after==self.before,"signed_chain_valid":True,
                "sources_unchanged":self.sources()==self.source_hashes}
        append(self.directory/"block_audits.jsonl",result)
        assert all(result[k] for k in ["snapshots_match","judge_state_unchanged","sources_unchanged"])

    def run(self):
        self.progress("initializing")
        try:
            self.initialize()
            for block in range(self.args.repeats):
                for arm in ARMS[block%3:]+ARMS[:block%3]:
                    for index,record in enumerate(self.selected):
                        self.one(arm,block+1,index,record)
                self.audit(block+1)
            summary=[]
            for repeat in range(1,self.args.repeats+1):
                for arm in ARMS:
                    rows=[r for r in self.rows if r["arm"]==arm and r["repeat"]==repeat]
                    values=sorted(r["total_ms"] for r in rows)
                    summary.append({"repeat":repeat,"arm":arm,"completed":len(rows),
                                    "mean_ms":statistics.fmean(values),"median_ms":statistics.median(values),
                                    "p95_nearest_rank_ms":values[max(0,__import__('math').ceil(.95*len(values))-1)]})
            write(self.directory/"summary.json", {"completed":len(self.rows),"blocks":summary,
                  "scope":"different_acceptance_endpoints_not_security_equivalent; single_host_warm_services_serial_small_workload"})
            self.progress("completed")
        except Exception:
            write(self.directory/"failure.json", {"traceback":traceback.format_exc(),"time_unix":time.time()})
            self.progress("failed")
            raise
        finally:
            if self.compose.exists():
                try:
                    if self.inputs:
                        program="import driver_v2 as d; d.stop_nodes(); d.emit('FINISH', {'reason':'architecture_runner_end'})"
                        command(["docker","exec","-e",f"N_ROUNDS={len(self.inputs)}",self.project+"-collector-1",
                                 "python","-B","-c",program],self.trace,timeout=30)
                        deadline=time.monotonic()+10
                        while not (self.rgg/"summary.json").exists() and time.monotonic()<deadline:
                            time.sleep(.2)
                finally:
                    command(["docker","compose","-p",self.project,"-f",self.compose_cli,"down"],self.trace)
            self.owner=None


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--preflight",action="store_true")
    parser.add_argument("--session")
    parser.add_argument("--count",type=int,default=16)
    parser.add_argument("--repeats",type=int,default=3)
    args=parser.parse_args()
    if args.preflight:
        print(json.dumps(preflight(),indent=2)); return
    if not args.session or not all(c.isalnum() or c in "-_" for c in args.session):
        parser.error("a fresh simple session name is required")
    if not 1<=args.count<=16 or not 1<=args.repeats<=3:
        parser.error("design limit is 16 records and 3 repeats")
    Runner(args).run()


if __name__ == "__main__":
    main()
