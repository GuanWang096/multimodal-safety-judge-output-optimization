"""CPU-only candidate checks; formal implementation and historical results stay read-only."""
from __future__ import annotations

import argparse
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
NAMES = ("node_v2.py", "protocol.py", "view_change.py", "view_runtime.py", "grouping.py", "fault_policy.py", "network_utils.py")
PROFILES = ((4, None), (16, 4), (20, 4), (24, 4))


def write(path, data):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def hashes(directory):
    return {name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in NAMES}


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Fixture:
    def __init__(self, code, n, groups, node_id=2):
        env = dict(NODE_ID=str(node_id), M=str(n), K_G="None" if groups is None else str(groups),
                   DELAY_MS="0", FAULT_MODE="none", FAULT_SCENARIO="none", FAULT_NODES="",
                   REPUTATION_ORDER=",".join(map(str, range(n))))
        before = {key: os.environ.get(key) for key in env}
        try:
            os.environ.update(env)
            self.node = load(f"retirement_node_{n}_{groups}_{node_id}_{id(self)}", code / "node_v2.py")
        finally:
            for key, value in before.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
        self.outgoing, self.events, self.inputs = [], [], []
        self.node.dispatch = lambda outgoing: self.outgoing.extend(copy.deepcopy(outgoing))
        self.node.emit = lambda kind, data: self.events.append(dict(kind=kind, data=copy.deepcopy(data)))
        self.runtime = self.node.sequence_state(0, "retirement-value-A", 1)

    def deliver(self, message):
        self.inputs.append(copy.deepcopy(message))
        self.node.process_protocol(copy.deepcopy(message))

    def report(self, target=1, request_first=False):
        with self.node.lock:
            if request_first:
                assert self.runtime.request_next_view() == target
            return self.node.make_view_change_locked(self.runtime, target)

    def close(self):
        self.node.executor.shutdown(wait=True)


def targeted(code, variant, output):
    from protocol import make_message, quorum, verify_message
    from view_change import build_rgg_new_view, validate_global_prepared
    records = []

    def preprepare(f, view=0):
        return make_message("PRE_PREPARE", f.node.primary_for_view(view), view, 0,
                            f.runtime.selected_digest, -1, {"start_ns": 1})

    def group_certificate(f, group, view=0):
        members = f.node.FROZEN_GROUPS[group]
        votes = [make_message("COMMIT", member, view, 0, f.runtime.selected_digest, group)
                 for member in members[:quorum(len(members))]]
        return make_message("GROUP_CERTIFICATE", f.node.LEADERS[group], view, 0,
                            f.runtime.selected_digest, -1, dict(source_group=group, certificate=votes))

    def prepared(f, view=0):
        certificates = [group_certificate(f, group, view) for group in range(quorum(f.node.K_G))]
        result = make_message("GLOBAL_PREPARED", f.node.primary_for_view(view), view, 0,
                              f.runtime.selected_digest, -1, dict(group_certificates=certificates))
        validate_global_prepared(result, f.node.FROZEN_GROUPS, f.node.LEADERS, quorum(f.node.K_G))
        return result

    def new_view(f, view):
        reports = [make_message("LEADER_VIEW_CHANGE", leader, view, 0, "", -1, {"global_lock": None})
                   for leader in sorted(f.node.LEADER_SET)]
        return build_rgg_new_view(sender=f.node.primary_for_view(view), target_view=view, sequence=0,
                                 view_changes=reports, groups=f.node.FROZEN_GROUPS, leaders=f.node.LEADERS,
                                 leader_quorum=quorum(f.node.K_G), ranked_leaders=f.node.RANKED_PRIMARIES,
                                 fallback_digest=f.runtime.selected_digest, start_ns=1)

    def outgoing_types(f):
        return [m["type"] for _, m in f.outgoing]

    def record(f, case, checks, observations):
        for _, message in f.outgoing:
            assert verify_message(message, f.node.MEMBERS)
        records.append(dict(case=case, nodes=f.node.M, groups=f.node.K_G,
                            checks=checks, passed=all(checks.values()), observations=observations,
                            inputs=f.inputs, outgoing=[dict(targets=sorted(targets), message=message)
                                                       for targets, message in f.outgoing], events=f.events,
                            final_state=dict(current_view=f.runtime.current_view, requested_view=f.runtime.requested_view,
                                             global_lock=f.runtime.global_lock, prepared=f.runtime.prepared,
                                             committed_digest=f.runtime.committed_digest)))
        f.close()
        assert all(checks.values()), (variant, case, checks)

    for n, groups in PROFILES:
        for request_first in (False, True):
            f = Fixture(code, n, groups)
            report = f.report(request_first=request_first)
            f.deliver(preprepare(f))
            issued = "PREPARE" in outgoing_types(f)
            expected = variant == "baseline"
            record(f, f"report_then_old_preprepare_{request_first}",
                   dict(vote_behavior_matches_variant=issued == expected,
                        report_valid=verify_message(report, f.node.MEMBERS), no_commit=f.runtime.committed_digest is None),
                   dict(report=report, lower_view_new_vote=issued, expected_lower_view_vote=expected))

        f = Fixture(code, n, groups)
        f.deliver(preprepare(f))
        f.outgoing.clear()
        report = f.report()
        group = -1 if groups is None else f.node.GROUP
        for sender in sorted(f.node.group_members(group)):
            f.deliver(make_message("PREPARE", sender, 0, 0, f.runtime.selected_digest, group))
        issued = "COMMIT" in outgoing_types(f)
        record(f, "report_then_old_prepare_quorum",
               dict(vote_behavior_matches_variant=issued == (variant == "baseline")),
               dict(report=report, lower_view_new_vote=issued))

        f = Fixture(code, n, groups, node_id=0)
        report = f.report()
        f.node.process_request(dict(sequence=0, digest=f.runtime.selected_digest, start_ns=1), None)
        issued = bool(f.outgoing)
        record(f, "report_then_repeated_client_request",
               dict(vote_behavior_matches_variant=issued == (variant == "baseline")),
               dict(report=report, lower_view_new_vote=issued))

        if groups is None:
            continue
        for request_first in (False, True):
            f = Fixture(code, n, groups)
            report = f.report(request_first=request_first)
            gp = prepared(f)
            f.deliver(gp)
            issued = "GLOBAL_COMMIT" in outgoing_types(f)
            record(f, f"report_then_old_global_prepared_{request_first}",
                   dict(vote_behavior_matches_variant=issued == (variant == "baseline"),
                        report_has_no_lock=report["payload"]["global_lock"] is None,
                        no_commit=f.runtime.committed_digest is None),
                   dict(report=report, lower_view_new_vote=issued))

        f = Fixture(code, n, groups, node_id=0)
        report = f.report()
        for group in range(quorum(groups)):
            f.deliver(group_certificate(f, group))
        issued = "GLOBAL_COMMIT" in outgoing_types(f)
        record(f, "report_then_old_group_certificates",
               dict(vote_behavior_matches_variant=issued == (variant == "baseline")),
               dict(report=report, lower_view_new_vote=issued))

        f = Fixture(code, n, groups)
        f.deliver(prepared(f))
        issued_before = "GLOBAL_COMMIT" in outgoing_types(f)
        lock_before = copy.deepcopy(f.runtime.global_lock)
        report = f.report()
        f.outgoing.clear()
        for leader in sorted(f.node.LEADER_SET):
            if leader != f.node.NODE_ID:
                f.deliver(make_message("GLOBAL_COMMIT", leader, 0, 0, f.runtime.selected_digest, -1))
        record(f, "lock_before_report_then_delayed_signed_commits",
               dict(voted_before_report=issued_before,
                    report_preserves_lock=report["payload"]["global_lock"] == lock_before["certificate"],
                    committed_original=f.runtime.committed_digest == "retirement-value-A",
                    no_new_global_commit="GLOBAL_COMMIT" not in outgoing_types(f),
                    certificate_notification="NOTIFY" in outgoing_types(f)),
               dict(report=report, lower_view_new_vote=False))

        f = Fixture(code, n, groups)
        report = f.report(target=2)
        f.deliver(new_view(f, 1))
        f.deliver(prepared(f, 1))
        lower_vote = any(m["view"] < 2 and m["type"] in {"PREPARE", "COMMIT", "GLOBAL_COMMIT"}
                         for _, m in f.outgoing)
        expected = variant == "baseline"
        # A fresh fixture checks successful voting in the requested target view.
        record(f, "report_view_2_then_install_view_1",
               dict(vote_behavior_matches_variant=lower_vote == expected),
               dict(report=report, lower_view_new_vote=lower_vote))

        f = Fixture(code, n, groups)
        report = f.report(target=2)
        f.deliver(new_view(f, 2))
        record(f, "report_then_valid_requested_new_view",
               dict(advanced=f.runtime.current_view == 2,
                    target_view_vote=any(m["type"] == "PREPARE" and m["view"] == 2 for _, m in f.outgoing),
                    floor_not_reset=f.runtime.requested_view == 2),
               dict(report=report, lower_view_new_vote=False))

    write(output / "targeted_cases.json", records)
    return records

