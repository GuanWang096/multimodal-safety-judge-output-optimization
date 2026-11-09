"""Offline deterministic checks of raw architecture receipts; no ledger writes."""
import argparse
import base64
import collections
import hashlib
import json
import math
import re
import statistics
import struct
from pathlib import Path
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

ROOT=Path(__file__).resolve().parents[2]
WORK=ROOT/"results/capacity"
OUT=ROOT/"results/revision/system"


def read(path):return json.loads(path.read_text(encoding="utf-8"))
def lines(path):return [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines() if s.strip()]
def canonical(value):return json.dumps(value,sort_keys=True,separators=(",",":")).encode()
def fields(values):
    encoded=[str(s).encode() for s in values]
    return b"".join(struct.pack(">Q",len(s))+s for s in encoded)


def check(directory):
    manifest=read(directory/"manifest.json")
    rows=lines(directory/"records.jsonl")
    expected=manifest["count"]*manifest["repeats"]
    assert len(rows)==expected*3 and all(r["success"] for r in rows)
    ids=manifest["sample_ids"]
    archive={r["sample_id"]:r for r in read(WORK/"j3_manifest.json")["records"]}
    registry=read(WORK/"registrations.json")
    by_judge={manifest["judge_ids"][model]:value for model,value in registry["judge_registrations"].items()}
    validator_keys={int(r["validatorId"].split("-")[-1]):Ed25519PublicKey.from_public_bytes(bytes.fromhex(r["publicKeyHex"]))
                    for r in registry["validator_registrations"]}
    record_lookup={r["decision_id"]:r for r in rows}
    assert len(record_lookup)==len(rows)
    cids=collections.defaultdict(set);txids=[];snapshot_count=0;judge_signature_count=0

    def check_judges(outputs,decision,sample,evidence):
        nonlocal judge_signature_count
        assert len(outputs)==3 and {o["judgeId"] for o in outputs}==set(manifest["judge_ids"].values())-{
            manifest["judge_ids"].get("internvl","")}
        for output in outputs:
            assert output["decisionId"]==decision and output["sampleId"]==sample and output["evidenceSha256"]==evidence
            reg=by_judge[output["judgeId"]]
            assert output["policySha256"]==reg["policySha256"] and output["adapterVersion"]==reg["adapterVersion"]
            message=fields(["MJ5-JUDGE-OUTPUT-v1",*[output[k] for k in ["judgeId","decisionId","sampleId","label","evidenceSha256","policySha256","adapterVersion"]]])
            Ed25519PublicKey.from_public_bytes(bytes.fromhex(reg["publicKeyHex"])).verify(bytes.fromhex(output["signatureHex"]),message)
            judge_signature_count+=1

    for row in rows:
        record=archive[row["sample_id"]]
        assert row["sample_id"]==ids[row["index"]]
        bundle=(WORK/record["bundle_path"]).read_bytes()
        assert hashlib.sha256(bundle).hexdigest()==row["evidence_sha256"]==record["evidence_sha256"]
        assert len(bundle)==row["bytes"]
        cids[row["sample_id"]].add(row["cid"])
        if row["arm"]=="signed_log":continue
        snapshot_count+=1
        s=row["snapshot"]
        assert s["status"]==row["status"]==("Certified" if row["arm"]=="full" else "SnapshotFrozen")
        assert s["decisionId"]==row["decision_id"] and s["sampleId"]==row["sample_id"]
        assert s["evidenceCid"]==row["cid"] and s["evidenceSha256"]==row["evidence_sha256"]
        assert s["posteriorUnsafePpm"]==record["posterior_unsafe_ppm"] and s["provisionalLabel"]==record["provisional_label"]
        check_judges(s["judgeOutputs"],row["decision_id"],row["sample_id"],row["evidence_sha256"])
        parts=["MJ5-DECISION-SNAPSHOT-v1",*[s[k] for k in ["decisionId","sampleId","evidenceCid","evidenceSha256",
              "provisionalLabel","posteriorUnsafePpm","committeeQuorum","certificateQuorum","deadlineUnix"]]]
        parts+=s["leaderValidatorIds"]
        for output in sorted(s["judgeOutputs"],key=lambda o:o["judgeId"]):
            parts.extend(output[k] for k in ["judgeId","label","policySha256","adapterVersion","signatureHex"])
        assert hashlib.sha256(fields(parts)).hexdigest()==s["decisionDigest"]==row["decision_digest"]
        receipts=[row["freeze_receipt"]]
        if row["arm"]=="full":receipts+=row["ack_receipts"]+[row["certify_receipt"]]
        for receipt in receipts:
            assert receipt["validation_status"]=="VALID" and set(receipt["endorsing_peers"])=={"localhost:7051","localhost:9051"}
            txids.append(receipt["tx_id"])
    assert all(len(values)==1 for values in cids.values())
    assert len(txids)==5*expected and len(set(txids))==len(txids)

    previous="0"*64;log_count=0
    owner=Ed25519PublicKey.from_public_bytes(bytes.fromhex(manifest["owner_public_key"]))
    with (directory/"signed_log.jsonl").open("rb") as handle:
        while True:
            offset=handle.tell();line=handle.readline()
            if not line:break
            entry=json.loads(line);body=entry["body"]
            assert body["previous"]==previous
            owner.verify(bytes.fromhex(entry["signature"]),canonical(body))
            previous=hashlib.sha256(line).hexdigest()
            row=record_lookup[body["decisionId"]]
            assert row["arm"]=="signed_log" and row["receipt"]["offset"]==offset and row["receipt"]["record_sha256"]==previous
            assert body["evidenceCid"]==row["cid"] and body["evidenceSha256"]==row["evidence_sha256"]
            record=archive[row["sample_id"]]
            assert body["posteriorUnsafePpm"]==record["posterior_unsafe_ppm"] and body["provisionalLabel"]==record["provisional_label"]
            check_judges(body["judgeOutputs"],row["decision_id"],row["sample_id"],row["evidence_sha256"])
            log_count+=1
    assert log_count==expected

    certs=lines(directory/"certificates.jsonl")
    assert len(certs)==expected
    for seq,cert in enumerate(certs):
        row=record_lookup[cert["decision_id"]]
        assert cert["digest"]==row["decision_digest"] and cert["sequence"]==seq
        senders=set()
        for message in cert["protocol_certificate"]:
            assert type(message["sender"]) is int and message["sender"] in registry["validator_order"][:4]
            assert (message["type"],message["view"],message["sequence"],message["digest"],message["group"])==("GLOBAL_COMMIT",cert["view"],seq,cert["digest"],-1)
            material={key:message[key] for key in ["type","sender","view","sequence","digest","group","payload"]}
            validator_keys[message["sender"]].verify(base64.b64decode(message["signature"],validate=True),canonical(material))
            senders.add(message["sender"])
        assert len(senders)>=3 and len(senders)==len(cert["protocol_certificate"])

    calls=lines(directory/"peer_calls.jsonl")
    raw_valid=collections.defaultdict(set);final_reads={};failures=[]
    for call in calls:
        args=call["arguments"]
        if "-c" not in args:continue
        payload=json.loads(args[args.index("-c")+1])["Args"]
        if payload[0]=="QueryDecisionSnapshot" and call["returncode"]==0:
            final_reads[payload[1]]=json.loads(call["stdout"])
        if payload[0] in {"FreezeDecisionSnapshot","SubmitCommitteeVoteV15","CertifyDecision"}:
            if call["returncode"]!=0:failures.append(call)
            for tid,status,peer in re.findall(r"txid \[([0-9a-fA-F]{64})\] committed with status \(([^)]+)\) at ([^\s]+)",call["stdout"]+call["stderr"]):
                assert status=="VALID"
                raw_valid[tid].add(peer)
    assert not failures and set(raw_valid)==set(txids)
    assert all(p=={"localhost:7051","localhost:9051"} for p in raw_valid.values())
    for row in rows:
        if row["arm"]!="signed_log":
            assert final_reads[row["decision_id"]]==row["snapshot"]
    audits=lines(directory/"block_audits.jsonl")
    assert len(audits)==manifest["repeats"] and all(all(a[k] for k in ["snapshots_match","judge_state_unchanged","signed_chain_valid","sources_unchanged"]) for a in audits)
    events=lines(directory/"rgg/events.jsonl")
    drivers=[e["data"] for e in events if e["type"]=="DRIVER_RESULT"]
    assert len(drivers)==expected and all(d["success"] and d["observed_commit_count"]==16 for d in drivers)
    commits={(e["node"],e["data"]["sequence"]):e["data"]["digest"] for e in events if e["type"]=="COMMIT"}
    assert len(commits)==16*expected
    assert all(value==certs[seq]["digest"] for (node,seq),value in commits.items())
    assert not any(e["type"] in {"NODE_ERROR","SAFETY_VIOLATION"} for e in events)
    summary=read(directory/"summary.json")
    for block in summary["blocks"]:
        values=sorted(r["total_ms"] for r in rows if r["arm"]==block["arm"] and r["repeat"]==block["repeat"])
        assert math.isclose(statistics.fmean(values),block["mean_ms"],abs_tol=1e-9)
        assert statistics.median(values)==block["median_ms"]
        assert values[math.ceil(.95*len(values))-1]==block["p95_nearest_rank_ms"]
    return {"passed":True,"records":len(rows),"unique_sample_ids":len(ids),"signed_log_records":log_count,
            "judge_signatures_verified":judge_signature_count,"snapshot_digests_recomputed":snapshot_count,
            "unique_valid_transactions":len(txids),"full_certificates_verified":len(certs),
            "node_commits":len(commits),"block_audits":len(audits),
            "scope":"offline_receipt_and_arithmetic_validation_not_production_security"}


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--session",required=True);args=parser.parse_args()
    if not all(c.isalnum() or c in "-_" for c in args.session):parser.error("simple session name required")
    directory=OUT/args.session
    result=check(directory)
    target=ROOT/"results/reproduced/revision/system/architecture-verification.json"
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2))


if __name__=="__main__":main()
