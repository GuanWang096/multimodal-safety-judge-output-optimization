"""Uncertainty of archived score/native-hard fidelity; not classification accuracy."""
import sys
sys.dont_write_bytecode=True
import json
import math
from collections import Counter
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/"results/revision/inputs/scores"
OUT=ROOT/"results/reproduced/revision/archive/qualification_uncertainty.json"
def rows(path):
    with path.open(encoding="utf-8") as f:return [json.loads(x) for x in f if x.strip()]
def wilson(k,n):
    z=1.959963984540054;p=k/n;d=1+z*z/n
    center=(p+z*z/(2*n))/d;half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [center-half,center+half]
result={}
for judge in ("qwen","safework","minicpm"):
    data=rows(BASE/judge/"qual100-a/judgments.jsonl")
    assert len(data)==100
    agreement=[];parent=list(range(100));seen={}
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    for i,r in enumerate(data):
        hard=r["hard_regression"];assert hard["status"]=="ok"
        agrees=hard["label"]==r["soft_evidence"]["predicted_label"]
        assert agrees==hard["agrees_with_soft_argmax"]
        agreement.append(int(agrees))
        keys=["conversation:"+r["conversation_sha256"]]
        keys += ["image:"+v["sha256"] for v in r["image_hashes"]]
        for key in keys:
            if key in seen:parent[find(i)]=find(seen[key])
            else:seen[key]=i
    groups=[find(i) for i in range(100)];unique=sorted(set(groups))
    successes=np.array([sum(a for a,g in zip(agreement,groups) if g==c) for c in unique])
    sizes=np.array([groups.count(c) for c in unique])
    rng=np.random.default_rng(20261030)
    weights=rng.multinomial(len(unique),np.ones(len(unique))/len(unique),size=10000)
    rates=(weights@successes)/(weights@sizes)
    audit=json.loads((BASE/judge/"qualification_audit.json").read_text())
    assert sum(agreement)/100==audit["hard_regression"]["argmax_agreement_rate"]
    result[judge]={"n":100,"agree_count":sum(agreement),"agreement":sum(agreement)/100,
                   "nominal_iid_wilson95":wilson(sum(agreement),100),
                   "known_exact_identity_group_bootstrap95":np.quantile(rates,[.025,.975]).tolist(),
                   "known_exact_identity_component_count":len(unique),"component_sizes":dict(Counter(sizes.tolist())),
                   "mismatched_sample_ids":[r["sample_id"] for r,a in zip(data,agreement) if not a]}
OUT.parent.mkdir(parents=True, exist_ok=True)
with OUT.open("x",encoding="utf-8") as f:
    json.dump({"status":"completed_pending_audit","role":"engineering_interface_fidelity_not_accuracy",
               "limitations":["Wilson assumes independent records and is nominal only",
                              "bootstrap groups only observed exact conversation/image hashes within the 100 engineering records; not full-source perceptual components",
                              "does not bound actual direct-hard majority disagreement on the distinct 513-record set"],
               "results":result},f,indent=2)
print(json.dumps({j:{"agreement":r["agreement"],"wilson95":r["nominal_iid_wilson95"],
                     "known_exact_components":r["known_exact_identity_component_count"]} for j,r in result.items()}))
