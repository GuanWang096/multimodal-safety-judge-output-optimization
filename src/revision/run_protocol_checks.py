"""Protocol regressions, exact fault placement enumeration and isolated-key unit tests.
No production key or deployment assurance is asserted by this test harness.
"""
import sys
sys.dont_write_bytecode=True
import base64
import copy
import importlib.util
import itertools
import json
import unittest
from pathlib import Path
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
ROOT=Path(__file__).resolve().parents[2]
CODE=ROOT/"src/rggpbft"
OUT=ROOT/"results/reproduced/revision/protocol"
sys.path.insert(0,str(CODE))
import protocol

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/"checks.json").exists():raise RuntimeError("Outputs exist; do not overwrite")
    tests=unittest.TestSuite()
    for name in ("test_protocol","test_view_change","test_view_runtime"):
        path=ROOT/"src/revision/tests"/(name+".py")
        spec=importlib.util.spec_from_file_location("revision_"+name,path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        tests.addTests(unittest.defaultTestLoader.loadTestsFromModule(module))
    with (OUT/"regression_tests.log").open("x",encoding="utf-8") as log:
        result=unittest.TextTestRunner(stream=log,verbosity=2).run(tests)
    assert result.wasSuccessful(), "Protocol regressions failed; inspect regression_tests.log"
    bounds=[]
    for n in range(1,65):
        t=(n-1)//3;q=protocol.quorum(n)
        assert 2*q-n>t and q<=n-t
        bounds.append({"members":n,"fault_bound":t,"quorum":q,"minimum_intersection":2*q-n})
    groups=[{j+4*k for k in range(4)} for j in range(4)]
    leaders={0,1,2,3};placements=[]
    for count in range(6):
        total=local_ok=leader_ok=both_ok=0
        for bad_tuple in itertools.combinations(range(16),count):
            bad=set(bad_tuple);total+=1
            a=all(len(bad&g)<=1 for g in groups);b=len(bad&leaders)<=1
            local_ok+=a;leader_ok+=b;both_ok+=a and b
        placements.append({"fault_count":count,"total_placements":total,"flat_fault_bound_met":count<=5,
                           "all_local_bounds_met":local_ok,"leader_bound_met":leader_ok,"both_met":both_ok})
    # Signers each own one ephemeral private key; verifier is given only public keys.
    class Signer:
        def __init__(self,node):self.node=node;self._key=Ed25519PrivateKey.generate()
        def public(self):return self._key.public_key()
        def sign(self,msg):
            if msg["sender"]!=self.node:raise PermissionError("cannot sign another identity")
            out=copy.deepcopy(msg);out["signature"]=base64.b64encode(self._key.sign(protocol.canonical_bytes(msg))).decode()
            return out
    signers={i:Signer(i) for i in range(4)};registry={i:s.public() for i,s in signers.items()}
    old_lookup=protocol.public_key;checks={}
    def envelope(sender):return {"type":"COMMIT","sender":sender,"view":0,"sequence":7,"digest":"a","group":0,"payload":{}}
    messages=[signers[i].sign(envelope(i)) for i in range(3)]
    try:
        protocol.public_key=lambda node:registry[node]
        checks["valid_own_signature"]=all(protocol.verify_message(m,range(4)) for m in messages)
        bad=copy.deepcopy(messages[0]);bad["sender"]=1
        checks["attacker_own_key_impersonation_rejected"]=not protocol.verify_message(bad,range(4))
        tampered=copy.deepcopy(messages[0]);tampered["digest"]="b"
        checks["digest_tamper_rejected"]=not protocol.verify_message(tampered,range(4))
        checks["unregistered_sender_rejected"]=not protocol.verify_message(messages[0],{1,2,3})
        fixture=protocol.make_message("COMMIT",0,0,7,"a",0)
        checks["public_deterministic_fixture_key_rejected"]=not protocol.verify_message(fixture,range(4))
        args=dict(message_type="COMMIT",view=0,sequence=7,digest="a",group=0,allowed_members=range(4),threshold=3)
        checks["valid_certificate"]=len(protocol.validate_certificate(messages,**args))==3
        for name,values,kwargs in (("duplicate_signer_rejected",[messages[0],messages[0],messages[1]],args),
                                  ("stale_view_rejected",messages,{**args,"view":1})):
            try:protocol.validate_certificate(values,**kwargs);checks[name]=False
            except ValueError:checks[name]=True
        try:signers[0].sign(envelope(1));checks["signer_identity_guard"]=False
        except PermissionError:checks["signer_identity_guard"]=True
    finally:protocol.public_key=old_lookup
    assert all(checks.values())
    output={"status":"completed_pending_audit","regression_tests":result.testsRun,
            "regression_source":str(ROOT/"src/revision/tests"),"implementation_under_test":str(CODE),
            "isolated_key_scope":"unit_fixture_in_one_test_process_not_container_key_isolation",
            "keys":"ephemeral_random_private_keys_not_saved","isolated_key_checks":checks,
            "quorum_bounds":bounds,"fault_placement_enumeration":placements,
            "placement_warning":"exact_combinatorial_counts_not_empirical_attack_probabilities_or_execution_traces"}
    with (OUT/"checks.json").open("x",encoding="utf-8") as f:json.dump(output,f,indent=2)
    print(json.dumps({"status":output["status"],"regression_tests":result.testsRun,"isolated_key_checks":checks}))

if __name__=="__main__":main()
