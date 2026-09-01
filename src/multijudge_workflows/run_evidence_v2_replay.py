from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


V15_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = V15_ROOT / "src"
SCRIPT_ROOT = Path(__file__).resolve().parent
for path in (SRC_ROOT, SCRIPT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from multijudge.evidence_v2 import (  # noqa: E402
    forbidden_runtime_keys,
    verify_embedded_hash,
    verify_package,
)

from mj5_common import read_json  # noqa: E402
from mj5_fabric_client import MJ5FabricClient  # noqa: E402
from mj5_rgg_bridge import extract_protocol_certificates  # noqa: E402


CORE_CONTAINERS = (
    "couchdb0",
    "couchdb1",
    "orderer.example.com",
    "peer0.org1.example.com",
    "peer0.org2.example.com",
)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_jsonl(path: Path, values: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for value in values:
            handle.write(json.dumps(value, sort_keys=True) + "\n")
    os.replace(temporary, path)


def docker_containers(action: str) -> None:
    names = [*CORE_CONTAINERS, "ipfs-kubo"]
    subprocess.run(
        ["docker", action, *names],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    if action == "start":
        time.sleep(8)


def latency_summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean_ms": None, "median_ms": None, "max_ms": None}
    return {
        "count": len(values),
        "mean_ms": statistics.fmean(values),
        "median_ms": statistics.median(values),
        "max_ms": max(values),
    }


class EvidenceV2Replay:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.repo_root = args.repo_root.resolve()
        self.workload = args.workload.resolve()
        self.manifest = read_json(self.workload / "manifest.json")
        verify_embedded_hash(self.manifest, "manifest_sha256")
        self.fusion_config = read_json(args.fusion_config.resolve())
        verify_embedded_hash(self.fusion_config, "config_sha256")
        self.client = MJ5FabricClient(self.repo_root, self.workload)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        suffix = "smoke" if args.batch_limit == 1 else "formal"
        self.run_id = f"evidence-v2-{suffix}-{timestamp}"
        self.run_dir = args.output.resolve() / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=False)
        batches = list(self.manifest["batches"])
        if args.batch_limit is not None:
            batches = batches[: args.batch_limit]
        self.batches = batches
        selected_indices = {
            int(index) for batch in batches for index in batch["record_indices"]
        }
        self.records = {
            int(record["index"]): record
            for record in self.manifest["records"]
            if int(record["index"]) in selected_indices
        }
        self.judge_ids: dict[str, str] = {}

    def initialize(self) -> None:
        docker_containers("start")
        self.judge_ids = self.client.ensure_registrations(f"{self.run_id}::")

    def verify_batch_off_chain(self, batch: dict[str, Any]) -> tuple[bytes, dict[str, Any]]:
        path = self.workload / str(batch["bundle_path"])
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != str(batch["bundle_sha256"]):
            raise RuntimeError(f"Batch file hash mismatch: {batch['batch_index']}")
        envelope = json.loads(payload)
        if forbidden := forbidden_runtime_keys(envelope):
            raise RuntimeError(
                f"Runtime label leakage in batch {batch['batch_index']}: {forbidden}"
            )
        indices = [int(value) for value in batch["record_indices"]]
        if len(indices) != len(envelope["packages"]):
            raise RuntimeError(f"Batch package count mismatch: {batch['batch_index']}")
        for index, package in zip(indices, envelope["packages"]):
            verify_package(package, self.fusion_config)
            evidence = package["canonical_evidence"]
            record = self.records[index]
            if (
                str(evidence["sample_id"]) != str(record["sample_id"])
                or str(package["canonical_evidence_sha256"])
                != str(record["evidence_sha256"])
            ):
                raise RuntimeError(f"Evidence binding mismatch at record {index}")
        return payload, envelope

    def run_confirmation(self) -> list[dict[str, Any]]:
        rows = []
        for position, batch in enumerate(self.batches, start=1):
            print(
                f"[confirm {position}/{len(self.batches)}] batch {batch['batch_index']}",
                flush=True,
            )
            payload, _envelope = self.verify_batch_off_chain(batch)
            cid, ipfs_ms = self.client.ipfs_add(payload)
            if self.client.ipfs_cat(cid) != payload:
                raise RuntimeError(f"IPFS round-trip mismatch: {batch['batch_index']}")
            decision_id = f"{self.run_id}::batch-{int(batch['batch_index']):02d}"
            records = [self.records[int(index)] for index in batch["record_indices"]]
            snapshot, freeze_ms, freeze_commit = self.client.freeze_batch_with_details(
                records,
                decision_id,
                cid,
                str(batch["bundle_sha256"]),
                self.judge_ids,
                int(time.time()) + 86_400,
            )
            vote_ms = []
            vote_commits = []
            for node_id in self.client.registrations["validator_order"][:2]:
                latency, commit = self.client.submit_vote_with_details(
                    snapshot, node_id, "ACK"
                )
                vote_ms.append(latency)
                vote_commits.append(commit)
                snapshot = self.client.query("QueryDecisionSnapshot", decision_id)
            if snapshot["status"] != "CommitteeConfirmed":
                raise RuntimeError(f"Batch {batch['batch_index']} was not confirmed")
            chain_records = {value["sampleId"]: value for value in snapshot["records"]}
            for record in records:
                anchored = chain_records.get(record["sample_id"])
                if anchored is None or (
                    anchored["evidenceSha256"] != record["evidence_sha256"]
                    or anchored["posteriorUnsafePpm"]
                    != record["posterior_unsafe_ppm"]
                    or anchored["provisionalLabel"] != record["provisional_label"]
                ):
                    raise RuntimeError(
                        f"Fabric record anchor mismatch: {record['sample_id']}"
                    )
            rows.append(
                {
                    "batch_index": int(batch["batch_index"]),
                    "record_count": len(records),
                    "record_indices": [int(value) for value in batch["record_indices"]],
                    "decision_id": decision_id,
                    "decision_digest": snapshot["decisionDigest"],
                    "evidence_cid": cid,
                    "batch_sha256": str(batch["bundle_sha256"]),
                    "ipfs_add_ms": ipfs_ms,
                    "freeze_ms": freeze_ms,
                    "vote_ms": vote_ms,
                    "status": snapshot["status"],
                    "fabric_commits": {
                        "freeze": freeze_commit,
                        "committee_votes": vote_commits,
                    },
                    "off_chain_recomputation_passed": True,
                    "ipfs_round_trip_passed": True,
                    "fabric_record_bindings_passed": True,
                }
            )
        write_jsonl(self.run_dir / "confirmation.jsonl", rows)
        return rows

    def run_rgg(self, confirmations: list[dict[str, Any]]) -> list[dict[str, Any]]:
        inputs = [
            {"decision_id": row["decision_id"], "digest": row["decision_digest"]}
            for row in confirmations
        ]
        rgg_dir = self.run_dir / "rgg"
        rgg_dir.mkdir()
        write_json(
            rgg_dir / "decision_digests.json",
            {"schema": "mj5-rgg-input-v1", "entries": inputs},
        )
        rgg_source = self.repo_root / "src/rggpbft"
        command = [
            sys.executable,
            str(rgg_source / "run_v2.py"),
            "--mode",
            "rgg",
            "--nodes",
            "16",
            "--groups",
            "4",
            "--rounds",
            str(len(inputs)),
            "--delay-ms",
            "5",
            "--round-timeout",
            "15",
            "--view-timeout",
            "1",
            "--reputation-order",
            ",".join(map(str, self.client.registrations["validator_order"])),
            "--seed",
            "20260705",
            "--run-dir",
            str(rgg_dir),
            "--image",
            "zte-rggpbft:mj5",
            "--skip-build",
        ]
        docker_containers("stop")
        try:
            subprocess.run(command, cwd=rgg_source, check=True)
        finally:
            docker_containers("start")
        certificates = extract_protocol_certificates(
            rgg_dir / "events.jsonl",
            inputs,
            self.client.registrations["validator_order"],
            rgg_source,
        )
        write_json(self.run_dir / "protocol_certificates.json", certificates)
        return certificates

    def run_certification_and_settlement(
        self,
        confirmations: list[dict[str, Any]],
        certificates: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        certificate_by_id = {value["decision_id"]: value for value in certificates}
        # Independent labels are deliberately loaded only after every certificate exists.
        settlement = read_json(self.workload / "settlement_labels.json")
        labels = {
            str(value["sample_id"]): str(value["label"])
            for value in settlement["labels"]
        }
        rows = []
        for position, confirmation in enumerate(confirmations, start=1):
            print(
                f"[certify {position}/{len(confirmations)}] batch {confirmation['batch_index']}",
                flush=True,
            )
            decision_id = confirmation["decision_id"]
            certificate = certificate_by_id[decision_id]
            snapshot = self.client.query("QueryDecisionSnapshot", decision_id)
            certify_ms, certify_commit = self.client.certify_with_details(
                snapshot,
                certificate["protocol_certificate"],
                int(certificate["view"]),
                int(certificate["sequence"]),
            )
            snapshot = self.client.query("QueryDecisionSnapshot", decision_id)
            if snapshot["status"] != "Certified":
                raise RuntimeError(f"Certificate admission failed: {decision_id}")
            batch_labels = [
                {"sample_id": self.records[index]["sample_id"], "label": labels[self.records[index]["sample_id"]]}
                for index in confirmation["record_indices"]
            ]
            settle_ms, retries, settle_commit = (
                self.client.settle_batch_with_commit_details(decision_id, batch_labels)
            )
            snapshot = self.client.query("QueryDecisionSnapshot", decision_id)
            if snapshot["status"] != "Settled" or snapshot["settled"] is not True:
                raise RuntimeError(f"Settlement failed: {decision_id}")
            rows.append(
                {
                    "batch_index": confirmation["batch_index"],
                    "record_count": confirmation["record_count"],
                    "decision_id": decision_id,
                    "decision_digest": snapshot["decisionDigest"],
                    "certificate_sha256": snapshot["certificate"]["certificateSha256"],
                    "protocol_certificate_sha256": snapshot["certificate"][
                        "protocolCertificateSha256"
                    ],
                    "certify_ms": certify_ms,
                    "settle_ms": settle_ms,
                    "settlement_mvcc_retries": retries,
                    "status": snapshot["status"],
                    "fabric_commits": {
                        "certify": certify_commit,
                        "settle": settle_commit,
                    },
                }
            )
        write_jsonl(self.run_dir / "certification_settlement.jsonl", rows)
        return rows

    def run(self) -> None:
        started = time.perf_counter()
        write_json(
            self.run_dir / "status.json",
            {"state": "running", "run_id": self.run_id, "labels_accessed": False},
        )
        try:
            self.initialize()
            confirmations = self.run_confirmation()
            certificates = self.run_rgg(confirmations)
            settled = self.run_certification_and_settlement(confirmations, certificates)
            record_count = sum(int(value["record_count"]) for value in confirmations)
            commit_records = [
                commit
                for row in confirmations
                for commit in [
                    row["fabric_commits"]["freeze"],
                    *row["fabric_commits"]["committee_votes"],
                ]
            ] + [
                commit
                for row in settled
                for commit in row["fabric_commits"].values()
            ]
            summary = {
                "schema": "multijudge-evidence-v2-replay-summary",
                "passed": True,
                "run_id": self.run_id,
                "record_count": record_count,
                "batch_count": len(confirmations),
                "expected_record_count": 64 if self.args.batch_limit == 1 else 513,
                "off_chain_recomputation_pass_count": record_count,
                "ipfs_round_trip_pass_count": len(confirmations),
                "fabric_anchor_pass_count": record_count,
                "committee_confirmed_batch_count": len(confirmations),
                "rgg_certificate_count": len(certificates),
                "settled_batch_count": len(settled),
                "valid_fabric_transaction_count": len(commit_records),
                "all_fabric_transactions_valid": all(
                    value["validation_status"] == "VALID" for value in commit_records
                ),
                "labels_accessed_before_certificate_admission": False,
                "latency": {
                    "ipfs_add": latency_summary(
                        [float(value["ipfs_add_ms"]) for value in confirmations]
                    ),
                    "freeze": latency_summary(
                        [float(value["freeze_ms"]) for value in confirmations]
                    ),
                    "committee_vote": latency_summary(
                        [
                            float(latency)
                            for value in confirmations
                            for latency in value["vote_ms"]
                        ]
                    ),
                    "certify": latency_summary(
                        [float(value["certify_ms"]) for value in settled]
                    ),
                    "settle": latency_summary(
                        [float(value["settle_ms"]) for value in settled]
                    ),
                },
                "elapsed_seconds": time.perf_counter() - started,
            }
            if record_count != summary["expected_record_count"]:
                raise RuntimeError(
                    f"Replay count mismatch: {record_count} != {summary['expected_record_count']}"
                )
            write_json(self.run_dir / "summary.json", summary)
            write_json(
                self.run_dir / "status.json",
                {"state": "completed", "run_id": self.run_id, "labels_accessed": True},
            )
            print(json.dumps(summary, indent=2))
        except Exception as error:
            write_json(
                self.run_dir / "status.json",
                {"state": "failed", "run_id": self.run_id, "error": str(error)},
            )
            raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    default_repo = Path(__file__).resolve().parents[2]
    parser.add_argument("--repo-root", type=Path, default=default_repo)
    parser.add_argument(
        "--workload",
        type=Path,
        default=V15_ROOT / "results/replay/workload",
    )
    parser.add_argument(
        "--fusion-config",
        type=Path,
        default=V15_ROOT / "results/mmds/fit/fusion_config.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=V15_ROOT / "results/replay",
    )
    parser.add_argument("--batch-limit", type=int)
    return parser


if __name__ == "__main__":
    EvidenceV2Replay(build_parser().parse_args()).run()
