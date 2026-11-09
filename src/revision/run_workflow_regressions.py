"""Run existing MJ5 contract tests, offline, with original sources unchanged."""
import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/"src/fabric_chaincode"
OUT=ROOT/"results/reproduced/revision/protocol/workflow-regression-20261031"


def environment():
    env=os.environ.copy()
    env.update(GOTOOLCHAIN="local",GOPROXY="off",GOSUMDB="off",GOFLAGS="-mod=readonly",GOMAXPROCS="2")
    return env


def preflight():
    version=subprocess.check_output(["go","version"],text=True,env=environment()).strip()
    assert version=="go version go1.22.2 linux/amd64",version
    assert (SOURCE/"core/mj5_service_test.go").is_file()
    return {"preflight_pass":True,"go_version":version,"source":str(SOURCE),"network":"disabled_by_GOPROXY_off"}


def hashes():
    paths=sorted((SOURCE/"core").glob("*.go"))+[SOURCE/"go.mod",SOURCE/"go.sum",Path(__file__)]
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--preflight",action="store_true");args=parser.parse_args()
    info=preflight()
    if args.preflight:print(json.dumps(info,indent=2));return
    OUT.mkdir(parents=True,exist_ok=False)
    original=hashes();started=time.perf_counter()
    cmd=["go","test","-json","-count=1","-p=1","-run","^(TestMJ5|TestCertificateQuorumForMemberCount)","./core"]
    with (OUT/"events.jsonl").open("w",encoding="utf-8") as stdout,(OUT/"stderr.log").open("w",encoding="utf-8") as stderr:
        result=subprocess.run(cmd,cwd=SOURCE,env=environment(),stdout=stdout,stderr=stderr,timeout=180)
    events=[json.loads(s) for s in (OUT/"events.jsonl").read_text(encoding="utf-8").splitlines() if s.strip()]
    tests=[e for e in events if e.get("Action") in {"pass","fail","skip"} and e.get("Test")]
    summary={**info,"command":cmd,"exit_code":result.returncode,"wall_seconds":time.perf_counter()-started,
             "source_sha256":original,"sources_unchanged":hashes()==original,
             "top_level_tests":sum("/" not in e["Test"] for e in tests),
             "subtests":sum("/" in e["Test"] for e in tests),
             "failed":[e["Test"] for e in tests if e["Action"]=="fail"],
             "skipped":[e["Test"] for e in tests if e["Action"]=="skip"],
             "evaluation_type":"original_contract_in_memory_transaction_fixture_not_live_organization_test"}
    summary["passed"]=result.returncode==0 and summary["sources_unchanged"] and summary["top_level_tests"]==12 and not summary["failed"] and not summary["skipped"]
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in summary.items() if k!="source_sha256"},indent=2))
    raise SystemExit(0 if summary["passed"] else 1)


if __name__=="__main__":main()
