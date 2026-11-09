"""Eight bounded, deterministic schedules through unmodified node handlers.

Only transport dispatch and telemetry emit are replaced. No model-checking or
live-network security claim; no secret-key or noncanonical-identity coverage.
"""
from __future__ import annotations
import argparse
import collections
import copy
import hashlib
import importlib.util
import json
import os
import sys
import time
import traceback
from pathlib import Path

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[2]
SRC=ROOT/"src/rggpbft"
OUT=ROOT/"results/reproduced/revision/protocol"
sys.path.insert(0,str(SRC))
from protocol import make_message,verify_message,quorum
from view_change import build_pbft_new_view,build_rgg_new_view


def write(path,value):
    path.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n",encoding="utf-8")


class Schedule:
    def __init__(self,n,groups,stage,directory):
        self.n=n;self.groups=groups;self.stage=stage;self.directory=directory
        self.directory.mkdir()
        self.nodes=[];self.queue=collections.deque();self.held=[];self.events=[];self.checks=[]
        self.trace=(directory/"trace.jsonl").open("w",encoding="utf-8")
        self.steps=0;self.phase="initial"
        self.a=hashlib.sha256(b"revision-cross-view-value-A").hexdigest()
        self.b=hashlib.sha256(b"revision-cross-view-value-B").hexdigest()
        self.start_ns=time.time_ns()
        override={"M":str(n),"K_G":str(groups) if groups else "None","DELAY_MS":"0",
                  "FAULT_MODE":"none","FAULT_SCENARIO":"none","FAULT_NODES":"",
                  "REPUTATION_ORDER":",".join(map(str,range(n)))}
        prior={k:os.environ.get(k) for k in [*override,"NODE_ID"]}
        try:
            os.environ.update(override)
            for node in range(n):
                os.environ["NODE_ID"]=str(node)
                spec=importlib.util.spec_from_file_location(f"trace_node_{n}_{groups}_{stage}_{node}",SRC/"node_v2.py")
                module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
                module.dispatch=lambda outgoing,source=node:self.dispatch(source,outgoing)
                module.emit=lambda kind,data,source=node:self.emit(source,kind,data)
                self.nodes.append(module)
        finally:
            for key,value in prior.items():
                if value is None:os.environ.pop(key,None)
                else:os.environ[key]=value

    def record(self,kind,**data):
        self.trace.write(json.dumps({"step":self.steps,"phase":self.phase,"kind":kind,**data},sort_keys=True)+"\n")

    def emit(self,node,kind,data):
        self.events.append({"node":node,"type":kind,"data":copy.deepcopy(data)})
        self.record("event",node=node,type=kind,data=data)

    def dispatch(self,source,outgoing):
        for targets,message in outgoing:
            for target in sorted(set(targets)-{source}):
                self.queue.append((target,copy.deepcopy(message)))

    def deliver(self,target,message):
        self.steps+=1
        if self.steps>20000:raise RuntimeError("delivery bound exceeded")
        assert type(message["sender"]) is int
        assert verify_message(message,set(range(self.n)))
        self.record("delivery",target=target,message=message)
        self.nodes[target].process_protocol(copy.deepcopy(message))

    def drain(self,hold=False):
        blocked="GLOBAL_COMMIT" if self.groups else "COMMIT"
        while self.queue:
            target,message=self.queue.popleft()
            if hold and message["type"]==blocked and message["view"]==0:
                self.held.append((target,message))
                self.record("held",target=target,message=message)
            else:self.deliver(target,message)

    def snapshot(self,label):
        states=[]
        for node in self.nodes:
            runtime=node.sequence_states[0]
            states.append({"node":node.NODE_ID,"view":runtime.current_view,
              "selected_digest":runtime.selected_digest,"prepared":runtime.prepared,
              "global_lock":runtime.global_lock,"committed_digest":runtime.committed_digest,
              "commit_certificate":runtime.commit_certificate,"members":runtime.members,
              "groups":dict(runtime.groups),"leaders":dict(runtime.leaders),"rank":runtime.rank})
        write(self.directory/(label+".json"),states)
        return states

    def check(self,name,value):
        self.checks.append({"name":name,"pass":bool(value)})
        self.record("check",name=name,passed=bool(value))
        if not value:raise AssertionError(name)

    def run(self):
        try:
            for node in self.nodes:
                node.process_request({"sequence":0,"digest":self.a,"start_ns":self.start_ns},None)
            self.drain(hold=self.stage=="locked")
            locked=self.snapshot("01_before_view_change")
            voters=sorted(self.nodes[0].LEADER_SET) if self.groups else list(range(self.n))
            self.check("all_frozen_voters_have_original_lock",all(
                (locked[i]["global_lock"] if self.groups else locked[i]["prepared"])["digest"]==self.a for i in voters))
            if self.stage=="locked":
                self.check("no_commit_before_release",all(s["committed_digest"] is None for s in locked))
            else:
                self.check("all_nodes_committed_before_view_change",all(s["committed_digest"]==self.a for s in locked))
            changes=[]
            self.phase="view_change_evidence"
            for i in voters:
                node=self.nodes[i];runtime=node.sequence_states[0]
                target=runtime.request_next_view()
                self.check(f"node_{i}_requests_view_1",target==1)
                change=node.make_view_change_locked(runtime,target)
                changes.append(change)
                self.record("generated_view_change",source=i,message=change)
            node=self.nodes[0]
            options=dict(sender=node.primary_for_view(1),target_view=1,sequence=0,view_changes=changes,
                         fallback_digest=self.b,start_ns=self.start_ns)
            if self.groups:
                good=build_rgg_new_view(**options,groups=node.FROZEN_GROUPS,leaders=node.LEADERS,
                                       leader_quorum=quorum(self.groups),ranked_leaders=node.RANKED_PRIMARIES)
            else:
                good=build_pbft_new_view(**options,members=tuple(range(self.n)),ranked_members=node.RANKED_PRIMARIES)
            write(self.directory/"view_change_messages.json",changes)
            self.check("safe_selection_overrides_conflicting_fallback",good["digest"]==self.a)
            payload=copy.deepcopy(good["payload"])
            payload["preprepare"]=make_message("PRE_PREPARE",good["sender"],1,0,self.b,-1,{"start_ns":self.start_ns})
            bad=make_message("NEW_VIEW",good["sender"],1,0,self.b,-1,payload)
            write(self.directory/"bad_new_view.json",bad)
            write(self.directory/"good_new_view.json",good)
            self.phase="conflicting_new_view"
            before_events=len(self.events)
            for target in range(self.n):self.deliver(target,bad)
            rejected=self.snapshot("02_after_conflicting_new_view")
            self.check("all_nodes_reject_conflicting_new_view",sum(e["type"]=="INVALID_NEW_VIEW" for e in self.events[before_events:])==self.n)
            self.check("invalid_message_does_not_advance_or_change_commit",all(
                s["view"]==0 and s["selected_digest"]==self.a and s["committed_digest"]==locked[i]["committed_digest"]
                for i,s in enumerate(rejected)))
            self.phase="valid_new_view"
            for target in range(self.n):self.deliver(target,good)
            accepted=self.snapshot("03_after_valid_new_view_before_new_votes")
            self.check("view_advanced_preserving_original_value",all(s["view"]==1 and s["selected_digest"]==self.a for s in accepted))
            self.check("locks_carried_before_new_round_votes",all(
                accepted[i]["global_lock" if self.groups else "prepared"]==locked[i]["global_lock" if self.groups else "prepared"] for i in voters))
            self.check("previous_commits_not_cleared",all(s["committed_digest"]==locked[i]["committed_digest"] for i,s in enumerate(accepted)))
            self.drain()
            self.phase="late_old_votes"
            for target,message in self.held:self.deliver(target,message)
            self.drain()
            completed=self.snapshot("04_completed")
            self.check("all_nodes_finish_original_value",all(s["committed_digest"]==self.a for s in completed))
            self.check("fixed_membership_and_ranking",all(
                all(s[k]==locked[i][k] for k in ["members","groups","leaders","rank"]) for i,s in enumerate(completed)))
            self.phase="stale_new_view_replay"
            before_events=len(self.events)
            for target in range(self.n):self.deliver(target,good)
            self.check("all_nodes_ignore_replayed_new_view",sum(e["type"]=="STALE_NEW_VIEW_IGNORED" for e in self.events[before_events:])==self.n)
            self.check("no_conflicting_commit_event",not any(e["type"]=="SAFETY_VIOLATION" for e in self.events))
            result={"passed":True}
        except Exception:
            result={"passed":False,"traceback":traceback.format_exc()}
        finally:
            for node in self.nodes:node.executor.shutdown(wait=True)
            self.trace.close()
        result.update(nodes=self.n,groups=self.groups,stage=self.stage,deliveries=self.steps,
                      held_old_votes=len(self.held),checks=self.checks,
                      events=dict(collections.Counter(e["type"] for e in self.events)))
        write(self.directory/"summary.json",result)
        return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--session",required=True);args=parser.parse_args()
    if not all(c.isalnum() or c in "-_" for c in args.session):parser.error("simple fresh session name required")
    directory=OUT/args.session;directory.mkdir(parents=True,exist_ok=False)
    paths=[SRC/name for name in ["node_v2.py","protocol.py","view_change.py","view_runtime.py","grouping.py","fault_policy.py"]]+[Path(__file__)]
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    write(directory/"manifest.json",{"source_sha256":hashes,"python":sys.version,
         "evaluation_type":"bounded_deterministic_real_handler_execution_not_exhaustive_model_check",
         "limitations":["fixed_membership_no_reconfiguration","canonical_integer_ids_only",
                        "public_demo_keys_not_authentication_security","no_socket_or_timeout_thread",
                        "committed_schedule_forces_new_view_for_state_retention_check"],
         "transport_replacements":["dispatch","emit"],"delivery_bound_per_trace":20000})
    results=[]
    for n,groups in [(4,None),(16,4),(20,4),(24,4)]:
        for stage in ["locked","committed"]:
            label=f"{'rgg' if groups else 'pbft'}-m{n}-{stage}"
            result=Schedule(n,groups,stage,directory/label).run()
            results.append({"case":label,**result})
            print(json.dumps({"case":label,"passed":result["passed"],"deliveries":result["deliveries"]}),flush=True)
    unchanged=all(hashlib.sha256(p.read_bytes()).hexdigest()==hashes[str(p.relative_to(ROOT))] for p in paths)
    write(directory/"summary.json",{"traces":results,"passed":sum(r["passed"] for r in results),
          "failed":sum(not r["passed"] for r in results),"sources_unchanged":unchanged})
    raise SystemExit(0 if unchanged and all(r["passed"] for r in results) else 1)


if __name__=="__main__":main()
