"""Non-overwriting reproduction of a signer-identity canonicalization issue.

Uses original protocol.py unchanged. Compares its verifier with a candidate
strict-JSON-integer guard inside isolated workers. No network, ledger or GPU.
"""
import argparse
import base64
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
import run_node_key_fixture as environment

ROOT = environment.ROOT
CODE = Path(__file__).resolve().parent
OUT = ROOT / "results/reproduced/revision/protocol/sender-identity-probe-20261031"
LABEL_VALUE = "sender-identity-20261031"
command = environment.command


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    image = environment.preflight()
    if args.preflight:
        return
    if OUT.exists():
        raise RuntimeError("Existing identity-probe evidence must not be overwritten")
    OUT.mkdir()
    containers = []
    transcripts = []
    status = {"state": "running", "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
              "image_id": image, "scope": "authentication_API_counterexample_and_candidate_guard_not_live_consensus",
              "original_source_modified": False, "host_and_Docker_trusted": True, "gpu_count": 0,
              "checks": [], "cases": []}

    def save():
        (OUT / "results.json").write_text(json.dumps(status, indent=2)+"\n", encoding="utf-8")

    def call(node, request):
        encoded = base64.b64encode(json.dumps(request).encode()).decode()
        observed = json.loads(command(["docker", "exec", containers[node], "python", "-B", "/fixture/worker.py", encoded]))
        row = {"node": node, "request": request, "response": observed}
        transcripts.append(row)
        with (OUT / "transcript.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(row)+"\n")
        if "error_type" in observed:
            raise RuntimeError("Worker failed: "+json.dumps(observed))
        return observed

    def check(name, passed, details=None):
        status["checks"].append({"name": name, "passed": bool(passed), "details": details})
        save()
        if not passed:
            raise RuntimeError("Probe integrity check failed: "+name)

    save()
    try:
        with tempfile.TemporaryDirectory(prefix="zte-revision-identity-keys-") as temp:
            secret_root = Path(temp); secret_root.chmod(0o700)
            public = {}
            for node in range(4):
                directory = secret_root / str(node); directory.mkdir(mode=0o700)
                key = Ed25519PrivateKey.generate()
                secret = directory / f"node_{node}.key"
                secret.write_bytes(key.private_bytes(serialization.Encoding.Raw, serialization.PrivateFormat.Raw, serialization.NoEncryption()))
                secret.chmod(0o600)
                public[str(node)] = base64.b64encode(key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)).decode()
            (OUT / "public_registry.json").write_text(json.dumps({"membership_epoch":0,"public_keys":public}, indent=2)+"\n")
            for node in range(4):
                name = f"zte-revision-identity-20261031-{node}"
                cli = ["docker","run","-d","--name",name,"--label",f"zte.revision.fixture={LABEL_VALUE}",
                       "--network","none","--read-only","--cap-drop","ALL","--security-opt","no-new-privileges",
                       "--user",f"{os.getuid()}:{os.getgid()}","-e",f"FIXTURE_NODE_ID={node}","--entrypoint","python"]
                mounts = [(secret_root/str(node),"/run/node-secrets"),(ROOT/"src/rggpbft","/fixture/protocol"),
                          (CODE/"sender_identity_worker.py","/fixture/worker.py"),(OUT/"public_registry.json","/fixture/public_registry.json")]
                for source,target in mounts:
                    cli += ["--mount",f"type=bind,source={environment.docker_bind_path(source)},target={target},readonly"]
                cid = command(cli+[image,"-B","-c","import time; time.sleep(3600)"])
                containers.append(cid)
                inspection = json.loads(command(["docker","inspect",cid]))[0]
                saved = {"id":cid,"image":inspection["Image"],"user":inspection["Config"]["User"],
                         "network":inspection["HostConfig"]["NetworkMode"],"readonly_rootfs":inspection["HostConfig"]["ReadonlyRootfs"],
                         "cap_drop":inspection["HostConfig"]["CapDrop"],"security_opt":inspection["HostConfig"]["SecurityOpt"],
                         "mounts":[{"destination":m["Destination"],"rw":m["RW"]} for m in inspection["Mounts"]]}
                (OUT/f"container_{node}.json").write_text(json.dumps(saved,indent=2)+"\n")
                check(f"isolated_mounts_{node}", saved["image"]==image and saved["network"]=="none" and saved["readonly_rootfs"] and len(saved["mounts"])==4 and not any(m["rw"] for m in saved["mounts"]),saved)
                probe = call(node,{"action":"probe"})
                check(f"only_own_key_{node}",probe["visible_secret_files"]==[f"node_{node}.key"] and not probe["other_private_keys_readable"],probe)

            digest = hashlib.sha256(b"synthetic-sender-identity-counterexample").hexdigest()
            def message(sender):
                return {"type":"COMMIT","sender":sender,"view":0,"sequence":9,"digest":digest,"group":0,"payload":{"epoch":0}}
            def sign(node,sender):
                return call(node,{"action":"sign_with_own_key","message":message(sender)})["message"]
            expected = {"message_type":"COMMIT","view":0,"sequence":9,"digest":digest,"group":0,"allowed_members":[0,1,2,3],"threshold":3}
            cases = [
                ("three_distinct_integer_signers",[sign(n,n) for n in range(3)],True,True),
                ("one_integer_signer_repeated",[sign(0,0) for _ in range(3)],False,False),
                ("one_key_three_string_aliases",[sign(0,s) for s in ("0","00","+0")],True,False),
                ("one_key_three_fractional_ids",[sign(0,s) for s in (0.1,0.2,0.3)],True,False),
            ]
            for name,messages,expected_original,expected_guard in cases:
                observed = {}
                for mode in ("original","candidate_strict_integer"):
                    observed[mode] = call(3,{"action":"certificate","mode":mode,"messages":messages,"expected":expected})
                row = {"name":name,"original":observed["original"],"candidate_strict_integer":observed["candidate_strict_integer"],
                       "security_expected_acceptance": name=="three_distinct_integer_signers",
                       "counterexample_observed": observed["original"]["accepted"] and name.startswith("one_key_")}
                status["cases"].append(row)
                check(name+"_original_matches_reproduction_prediction",observed["original"]["accepted"]==expected_original)
                check(name+"_candidate_guard",observed["candidate_strict_integer"]["accepted"]==expected_guard)
            status["state"]="counterexample_confirmed_candidate_guard_passed_pending_audit"
            status["counterexample_count"]=sum(row["counterexample_observed"] for row in status["cases"])
            status["sources"]={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in
                               (Path(__file__).resolve(),CODE/"sender_identity_worker.py",CODE/"run_node_key_fixture.py",ROOT/"src/rggpbft/protocol.py")}
            save()
    except Exception as exc:
        status["state"]="failed"; status["error"]=repr(exc)
        if isinstance(exc,subprocess.CalledProcessError):
            status["command_stdout"]=exc.stdout; status["command_stderr"]=exc.stderr
        save()
        raise
    finally:
        removed=[]
        for cid in containers:
            label=command(["docker","inspect",cid,"--format",'{{index .Config.Labels "zte.revision.fixture"}}'])
            if label!=LABEL_VALUE:
                raise RuntimeError("Unexpected label: refusing container deletion")
            command(["docker","rm","-f",cid]); removed.append(cid)
        status["removed_fixture_container_ids"]=removed
        status["ephemeral_private_directory_removed"]="secret_root" in locals() and not secret_root.exists()
        status["finished_at"]=dt.datetime.now(dt.timezone.utc).isoformat()
        save()
    print(json.dumps({"state":status["state"],"counterexamples":status["counterexample_count"],"original_source_modified":False}),flush=True)


if __name__=="__main__":
    main()
