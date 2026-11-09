"""Bounded source-adoption regression; no Docker, sockets, or timing claim."""
from __future__ import annotations

import argparse
import base64
import copy
import datetime as dt
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FORMAL = ROOT / "src/rggpbft"
TRIAL = HERE.parent / "identity-fix-trial-20261102"
NAMES = ("protocol.py", "view_change.py", "view_runtime.py", "node_v2.py", "grouping.py", "fault_policy.py", "network_utils.py")
PROFILES = ((4, None), (16, 4), (20, 4), (24, 4))
SENDERS = ("0", "00", "+0", 0.1, 0.2, 0.3, 0)


def write(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hashes(directory):
    return {name: sha(directory / name) for name in NAMES}


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ingress(code, protocol, variant, output):
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

    keys = {i: Ed25519PrivateKey.generate() for i in range(24)}
    registry = {str(i): base64.b64encode(key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)).decode("ascii") for i, key in keys.items()}
    write(output / "ingress_public_registry.json", registry)
    old_lookup = protocol.public_key
    protocol.public_key = lambda i: keys[i].public_key()
    records = []
    try:
        for n, groups in PROFILES:
            for index, sender in enumerate(SENDERS):
                env = {"NODE_ID": "0", "M": str(n), "K_G": "None" if groups is None else str(groups),
                       "DELAY_MS": "0", "FAULT_MODE": "none", "FAULT_SCENARIO": "none", "FAULT_NODES": "",
                       "REPUTATION_ORDER": ",".join(map(str, range(n)))}
                previous = {key: os.environ.get(key) for key in env}
                try:
                    os.environ.update(env)
                    node = load(f"ingress_{variant}_{n}_{index}", code / "node_v2.py")
                finally:
                    for key, value in previous.items():
                        if value is None:
                            os.environ.pop(key, None)
                        else:
                            os.environ[key] = value
                assert node.verify_message is protocol.verify_message
                events, outgoing = [], []
                node.emit = lambda kind, data: events.append({"kind": kind, "data": copy.deepcopy(data)})
                node.dispatch = lambda messages: outgoing.extend(copy.deepcopy(messages))
                message = dict(type="PREPARE", sender=sender, view=0, sequence=99, digest="adoption-ingress",
                               group=-1 if groups is None else 0, payload={})
                message["signature"] = base64.b64encode(keys[0].sign(protocol.canonical_bytes(message))).decode("ascii")
                message = json.loads(json.dumps(message))
                # Check cryptographic validity only, not the implementation identity rule.
                keys[0].public_key().verify(base64.b64decode(message["signature"]), protocol.canonical_bytes(message))
                node.process_protocol(copy.deepcopy(message))  # No type guard or Schedule.deliver wrapper.
                pending = [copy.deepcopy(m) for bucket in node.pending_prepares._messages.values() for m in bucket.values()]
                expected_accept = variant == "baseline" or type(sender) is int
                expected_pending = [message] if expected_accept else []
                checks = {
                    "pending_matches_expectation": pending == expected_pending,
                    "identity_tracker_untouched_on_reject": len(node.equivocations._digests) == (1 if expected_accept else 0),
                    "invalid_event_matches_expectation": [e["kind"] for e in events] == ([] if expected_accept else ["INVALID_SIGNATURE"]),
                    "no_consensus_state_created": not node.sequence_states and not node.states and not node.committed,
                    "no_outgoing_messages": not outgoing,
                }
                records.append({"nodes": n, "groups": groups, "sender_type": type(sender).__name__, "message": message,
                                "expected_accept_into_cache": expected_accept, "observed_cached": bool(pending),
                                "pending_after": pending, "events": events, "checks": checks,
                                "all_checks_pass": all(checks.values())})
                node.executor.shutdown(wait=True)
    finally:
        protocol.public_key = old_lookup
    write(output / "ingress_cases.json", records)
    assert len(records) == 28 and all(r["all_checks_pass"] for r in records)
    return records

