from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any


SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from mj5_fabric_client import MJ5FabricClient  # noqa: E402


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def run(args: argparse.Namespace) -> None:
    run_dir = args.run_dir.resolve()
    workload = args.workload.resolve()
    summary = read_json(run_dir / "summary.json")
    manifest = read_json(workload / "manifest.json")
    confirmations = read_jsonl(run_dir / "confirmation.jsonl")
    settlements = read_jsonl(run_dir / "certification_settlement.jsonl")
    settlement_by_id = {value["decision_id"]: value for value in settlements}
    client = MJ5FabricClient(args.repo_root.resolve(), workload)
    failures = []
    transaction_ids = []
    anchored_records = 0
    for row in confirmations:
        batch = manifest["batches"][int(row["batch_index"])]
        payload = (workload / batch["bundle_path"]).read_bytes()
        if client.ipfs_cat(str(row["evidence_cid"])) != payload:
            failures.append(f"ipfs_round_trip:{row['batch_index']}")
        if hashlib.sha256(payload).hexdigest() != str(row["batch_sha256"]):
            failures.append(f"batch_hash:{row['batch_index']}")
        snapshot = client.query("QueryDecisionSnapshot", str(row["decision_id"]))
        if snapshot.get("status") != "Settled" or snapshot.get("settled") is not True:
            failures.append(f"final_state:{row['batch_index']}")
        if (
            snapshot.get("evidenceCid") != row["evidence_cid"]
            or snapshot.get("evidenceSha256") != row["batch_sha256"]
            or snapshot.get("decisionDigest") != row["decision_digest"]
        ):
            failures.append(f"batch_anchor:{row['batch_index']}")
        expected_records = {
            manifest["records"][index]["sample_id"]: manifest["records"][index]
            for index in row["record_indices"]
        }
        chain_records = {value["sampleId"]: value for value in snapshot["records"]}
        if set(chain_records) != set(expected_records):
            failures.append(f"record_membership:{row['batch_index']}")
        for sample_id, expected in expected_records.items():
            observed = chain_records.get(sample_id, {})
            if (
                observed.get("evidenceSha256") != expected["evidence_sha256"]
                or observed.get("posteriorUnsafePpm")
                != expected["posterior_unsafe_ppm"]
                or observed.get("provisionalLabel")
                != expected["provisional_label"]
            ):
                failures.append(f"record_anchor:{sample_id}")
            else:
                anchored_records += 1
        settled = settlement_by_id.get(row["decision_id"], {})
        certificate = snapshot.get("certificate", {})
        if (
            certificate.get("certificateSha256")
            != settled.get("certificate_sha256")
            or certificate.get("protocolCertificateSha256")
            != settled.get("protocol_certificate_sha256")
        ):
            failures.append(f"certificate_binding:{row['batch_index']}")
        commits = [
            row["fabric_commits"]["freeze"],
            *row["fabric_commits"]["committee_votes"],
            settled["fabric_commits"]["certify"],
            settled["fabric_commits"]["settle"],
        ]
        for commit in commits:
            if commit["validation_status"] != "VALID":
                failures.append(f"invalid_transaction:{commit['tx_id']}")
            transaction_ids.append(commit["tx_id"])

    checks = {
        "summary_passed": summary.get("passed") is True,
        "record_count": summary.get("record_count") == 513,
        "batch_count": len(confirmations) == len(settlements) == 9,
        "all_record_anchors": anchored_records == 513,
        "transaction_count": len(transaction_ids) == 45,
        "transaction_ids_unique": len(set(transaction_ids)) == 45,
        "labels_not_accessed_before_certificate": summary.get(
            "labels_accessed_before_certificate_admission"
        )
        is False,
    }
    passed = all(checks.values()) and not failures
    report = {
        "schema": "multijudge-evidence-v2-replay-audit",
        "passed": passed,
        "run_id": summary.get("run_id"),
        "checks": checks,
        "failures": failures,
        "verified_record_count": anchored_records,
        "verified_batch_count": len(confirmations),
        "verified_valid_transaction_count": len(transaction_ids),
        "verified_ipfs_cids": [value["evidence_cid"] for value in confirmations],
    }
    write_json(run_dir / "replay_audit.json", report)
    print(json.dumps(report, indent=2))
    if not passed:
        raise RuntimeError("Evidence V2 replay audit failed")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    default_repo = Path(__file__).resolve().parents[3]
    parser.add_argument("--repo-root", type=Path, default=default_repo)
    parser.add_argument(
        "--workload",
        type=Path,
        default=default_repo / "results/replay/workload",
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
