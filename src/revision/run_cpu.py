"""Revision-only CPU sensitivity analysis; never overwrite submission artifacts."""
from __future__ import annotations
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "2"
import sys
sys.dont_write_bytecode = True
import json
import math
import time
import itertools
from collections import Counter
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit

ROOT = Path(__file__).resolve().parents[2]
V15 = ROOT
WF = ROOT / "src/multijudge_workflows"
PUBLIC = ROOT
RESULTS = ROOT / "results"
OUT = RESULTS / "reproduced/revision/cpu"
sys.path[:0] = [str(V15 / "src"), str(WF)]
from fit_calibrators import load_split
from multijudge.baselines import classification_metrics, baseline_predictions, select_constrained_threshold
from multijudge.calibration import (fit_calibrator, apply_calibrator, cluster_weights,
                                   make_group_folds, calibration_metrics)
from multijudge.fusion import (FusionModel, fit_fusion_model, calibrated_feature_matrix,
                              fusion_model_from_dict, predict_calibrated_fusion)

JUDGES = ("qwen", "safework", "minicpm")
SEED = 20260729

def read(path):
    return json.loads(path.read_text(encoding="utf-8"))

def rows(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]

def encode(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)

def save(name, value):
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, default=encode, allow_nan=False)
        handle.write("\n")

def nonnegative_fit(features, y, groups, names):
    weights = cluster_weights(groups) * len(y)
    mean = np.average(features, axis=0, weights=weights)
    scale = np.sqrt(np.maximum(np.average((features-mean)**2, axis=0, weights=weights), 1e-12))
    z = (features-mean)/scale
    def objective(theta):
        eta = z @ theta[:-1] + theta[-1]
        residual = weights * (expit(eta)-y)
        loss = np.dot(weights, np.logaddexp(0, eta)-y*eta) + .5/.3*np.dot(theta[:-1],theta[:-1])
        gradient = np.r_[z.T @ residual + theta[:-1]/.3, residual.sum()]
        return float(loss), gradient
    opt = minimize(objective, np.zeros(z.shape[1]+1), jac=True, method="L-BFGS-B",
                   bounds=[(0,None)]*z.shape[1]+[(None,None)],
                   options={"maxiter":10000, "ftol":1e-13, "gtol":1e-8})
    if not opt.success:
        raise RuntimeError(str(opt.message))
    assert np.all(opt.x[:-1] >= 0)
    return FusionModel("F1", tuple(names), tuple(names), mean, scale, opt.x[:-1],
                       float(opt.x[-1]), "l2", .3, None)

def fit_variant(raw, y, groups, judges, methods, constrained):
    pars = {j: fit_calibrator(methods[j],raw[j],y,groups) for j in judges}
    calibrated = {j:apply_calibrator(pars[j],raw[j]) for j in judges}
    x,names = calibrated_feature_matrix(calibrated, judges, "F1")
    if constrained:
        model = nonnegative_fit(x,y,groups,names)
    else:
        model = fit_fusion_model(x,y,groups,feature_family="F1",feature_names=names,
                                 judge_order=judges,penalty="l2",c_value=.3,l1_ratio=None,seed=SEED)
    return pars,model

def beta_variant(method, p, y, groups):
    if method in ("identity","temperature","beta"):
        return fit_calibrator(method,p,y,groups)
    p = np.clip(p,1e-6,1-1e-6)
    w = cluster_weights(groups)
    reduced = method == "beta_b_zero"
    def objective(t):
        a = np.exp(t[0]); b = 0. if reduced else np.exp(t[1]); c = t[-1]
        eta = a*np.log(p)-b*np.log1p(-p)+c
        pp = np.clip(expit(eta),1e-6,1-1e-6)
        reg = 1e-8 * np.dot(t[:-1],t[:-1])
        return float(-np.dot(w,y*np.log(pp)+(1-y)*np.log1p(-pp))+reg)
    bounds = [(-5,5),(-10,10)] if reduced else [(-8,8),(-8,8),(-10,10)]
    opt = minimize(objective,np.zeros(len(bounds)),method="L-BFGS-B",bounds=bounds)
    if not opt.success:
        raise RuntimeError(f"{method}: {opt.message}")
    return {"method":method,"a":float(np.exp(opt.x[0])),
            "b":0. if reduced else float(np.exp(opt.x[1])),"c":float(opt.x[-1]),
            "objective":float(opt.fun),"bounds":bounds,"optimizer_success":True}

def apply_variant(pars,p):
    if pars["method"] in ("identity","temperature","beta"):
        return apply_calibrator(pars,p)
    p=np.clip(p,1e-6,1-1e-6)
    return expit(pars["a"]*np.log(p)-pars["b"]*np.log1p(-p)+pars["c"])

def metrics(y,p,t):
    pp=np.clip(p,1e-6,1-1e-6)
    return {**classification_metrics(y,p>=t),"threshold":float(t),
            "brier":float(np.mean((p-y)**2)),
            "log_loss":float(-np.mean(y*np.log(pp)+(1-y)*np.log1p(-pp)))}

def bootstrap(y,pred,ref,groups):
    unique=sorted(set(groups)); pos={g:i for i,g in enumerate(unique)}
    def counts(pr):
        c=np.zeros((len(unique),4),dtype=int)
        for yy,pp,g in zip(y,pr,groups):
            col=0 if yy and pp else 1 if not yy and not pp else 2 if not yy and pp else 3
            c[pos[g],col]+=1
        return c
    a,b=counts(pred),counts(ref)
    def f1(c):
        tp,tn,fp,fn=c.T
        return .5*(np.divide(2*tp,2*tp+fp+fn,out=np.zeros(len(tp),float),where=(2*tp+fp+fn)>0)
                    +np.divide(2*tn,2*tn+fp+fn,out=np.zeros(len(tp),float),where=(2*tn+fp+fn)>0))
    rng=np.random.default_rng(20261030); deltas=[]
    for _ in range(50):
        weights=rng.multinomial(len(unique),np.ones(len(unique))/len(unique),size=200)
        deltas.extend(f1(weights@a)-f1(weights@b))
    return {"replicates":10000,"unit":"component_id","role":"descriptive_revision_sensitivity_not_confirmatory",
            "delta_macro_f1":float(f1(a.sum(axis=0)[None,:])[0]-f1(b.sum(axis=0)[None,:])[0]),
            "percentile_95_ci":np.quantile(deltas,[.025,.975]).tolist()}

def main():
    started=time.time()
    if (OUT/"run_status.json").exists() or (OUT/"fitted_variants.json").exists():
        raise RuntimeError("This run already has outputs; preserve them and choose a new run directory.")
    frozen=read(RESULTS/"mmds/fit/fusion_config.json")
    methods=frozen["calibrator_methods"]
    fit=load_split(ROOT/"results/revision/inputs/fit.jsonl",RESULTS/"revision/inputs/scores","fit")
    select=load_split(ROOT/"results/revision/inputs/select.jsonl",RESULTS/"revision/inputs/scores","select")
    assert not set(fit["components"]) & set(select["components"])
    y=np.r_[fit["targets"],select["targets"]]
    groups=["fit:"+g for g in fit["components"]]+["select:"+g for g in select["components"]]
    raw={j:np.r_[fit["raw_probabilities"][j],select["raw_probabilities"][j]] for j in JUDGES}
    assert len(y)==1760
    folds=make_group_folds(y,groups,n_splits=5,seed=SEED+3000000)
    ref=classification_metrics(y,baseline_predictions(raw,JUDGES)["or_1_of_n"])
    lower={"minimum_unsafe_recall":max(0,ref["unsafe_recall"]-.01),
           "minimum_safe_specificity":max(0,ref["safe_specificity"]-.03)}
    save("run_design.json",{"created_unix":started,"status":"revision_exploratory","development_records":len(y),
                           "reference":ref,"constraints":lower,"seed":SEED,"C":.3,
                           "confirmation_previously_opened":True,"cpu_threads":2})
    configs={}; oofs={}; final_models={}
    variants=[("full_refit",JUDGES,False),("nonnegative",JUDGES,True)]
    variants += [("without_"+omit,tuple(j for j in JUDGES if j!=omit),False) for omit in JUDGES]
    for name,judges,constrained in variants:
        print("FIT",name,flush=True)
        oof=np.full(len(y),np.nan); fold_info=[]
        for fold in range(5):
            va=folds==fold; tr=~va; gg=np.asarray(groups)
            assert not set(gg[tr]) & set(gg[va])
            pars,model=fit_variant({j:raw[j][tr] for j in judges},y[tr],gg[tr].tolist(),judges,methods,constrained)
            oof[va]=predict_calibrated_fusion({j:raw[j][va] for j in judges},pars,model)
            fold_info.append({"fold":fold,"calibrators":pars,"model":model.to_dict()})
        assert np.all(np.isfinite(oof))
        threshold=select_constrained_threshold(y,oof,**lower)
        unrestricted=select_constrained_threshold(y,oof,minimum_unsafe_recall=0,minimum_safe_specificity=0)
        pars,model=fit_variant(raw,y,groups,judges,methods,constrained)
        configs[name]={"calibrators":pars,"model":model.to_dict(),"constrained_threshold":threshold,
                       "unrestricted_diagnostic_threshold":unrestricted,"folds":fold_info}
        final_models[name]=(pars,model);oofs[name]=oof
    for judge in JUDGES:
        name="single_calibrated_"+judge; oof=np.full(len(y),np.nan)
        for fold in range(5):
            va=folds==fold;tr=~va
            pars=fit_calibrator(methods[judge],raw[judge][tr],y[tr],np.asarray(groups)[tr].tolist())
            oof[va]=apply_calibrator(pars,raw[judge][va])
        configs[name]={"judge":judge,"calibrator":fit_calibrator(methods[judge],raw[judge],y,groups),
                       "constrained_threshold":select_constrained_threshold(y,oof,**lower),
                       "unrestricted_diagnostic_threshold":select_constrained_threshold(y,oof,minimum_unsafe_recall=0,minimum_safe_specificity=0)}
        oofs[name]=oof
    # Persist all development choices before this invocation loads confirm labels.
    save("fitted_variants.json",configs)
    save("development_oof.json",{"sample_ids":fit["sample_ids"]+select["sample_ids"],
                                  "groups":groups,"folds":folds,"targets":y,"predictions":oofs})
    print("CALIBRATION SENSITIVITY",flush=True)
    cal_results={};cal_oof={}
    fit_folds=make_group_folds(fit["targets"],fit["components"],n_splits=5,seed=SEED)
    for judge in JUDGES:
        cal_results[judge]={};cal_oof[judge]={}
        for method in ("identity","temperature","beta","beta_wide","beta_b_zero"):
            p=fit["raw_probabilities"][judge]; oof=np.full(len(p),np.nan);parameters=[]
            for fold in range(5):
                va=fit_folds==fold;tr=~va
                pars=beta_variant(method,p[tr],fit["targets"][tr],np.asarray(fit["components"])[tr].tolist())
                oof[va]=apply_variant(pars,p[va]);parameters.append(pars)
            cal_results[judge][method]={"development_oof":calibration_metrics(fit["targets"],oof,fit["components"]),
                                        "fold_parameters":parameters,
                                        "full_fit_parameters":beta_variant(method,p,fit["targets"],fit["components"])}
            cal_oof[judge][method]=oof
        print("CALIBRATED",judge,flush=True)
    save("calibration_sensitivity.json",cal_results)
    save("calibration_oof.json",{"sample_ids":fit["sample_ids"],"folds":fit_folds,"predictions":cal_oof})
    cr=rows(PUBLIC/"results/mmds/evaluation/confirmatory_predictions.jsonl")
    cy=np.asarray([int(r["expected_label"]=="unsafe") for r in cr])
    cg=[r["component_id"] for r in cr]
    assert len(cr)==513 and len({r["sample_id"] for r in cr})==513
    assert not set(cg)&set(fit["components"]+select["components"])
    assert not {r["sample_id"] for r in cr}&set(fit["sample_ids"]+select["sample_ids"])
    craw={j:np.asarray([r["raw_p_unsafe"][j] for r in cr]) for j in JUDGES}
    original=predict_calibrated_fusion(craw,frozen["final_development_calibrators"],fusion_model_from_dict(frozen["model"]))
    residual=float(np.max(np.abs(original-np.asarray([r["fusion_p_unsafe"] for r in cr]))))
    assert residual<1e-12
    original_binary=original>=frozen["threshold"]
    report={"original_frozen":metrics(cy,original,frozen["threshold"])}
    predictions={"original_frozen":original}
    for name,config in configs.items():
        if "judge" in config:
            pp=apply_calibrator(config["calibrator"],craw[config["judge"]])
        else:
            pars,model=final_models[name];pp=predict_calibrated_fusion(craw,pars,model)
        predictions[name]=pp
        report[name]={"constraint_feasible":config["constrained_threshold"]["feasible"]}
        for label,key in (("constrained","constrained_threshold"),("unrestricted_diagnostic","unrestricted_diagnostic_threshold")):
            choice=config[key]
            if not choice["feasible"]:
                report[name][label]=None;continue
            tt=choice["threshold"]
            report[name][label]={**metrics(cy,pp,tt),"difference_vs_original":bootstrap(cy,pp>=tt,original_binary,cg)}
    baselines=baseline_predictions(craw,JUDGES)
    baselines["qwen_or_safework_posthoc"]=(craw["qwen"]>=.5)|(craw["safework"]>=.5)
    for name,binary in baselines.items():
        report[name]=classification_metrics(cy,binary)
    save("confirmation_sensitivity.json",{"role":"revision_exploratory","original_replay_max_error":residual,"results":report})
    save("confirmation_predictions.json",{"sample_ids":[r["sample_id"] for r in cr],"targets":cy,"groups":cg,"probabilities":predictions})
    counts=Counter(cg)
    pattern_rows=[]
    for bits in itertools.product((0,1),repeat=3):
        mask=np.all(np.column_stack([craw[j]>=.5 for j in JUDGES])==bits,axis=1)
        pattern_rows.append({"votes":bits,"records":int(mask.sum()),"unsafe":int(cy[mask].sum()),
                             "fusion_unsafe":int(original_binary[mask].sum())})
    costs={name:{"FN":m["unsafe_false_negative"],"FP":m["unsafe_false_positive"],
                 "review_if_all_alerts":m["unsafe_true_positive"]+m["unsafe_false_positive"],
                 "illustrative_cost_FN_to_FP":{str(r):r*m["unsafe_false_negative"]+m["unsafe_false_positive"] for r in (.5,1,2,4,10)}}
           for name,m in report.items() if "unsafe_false_negative" in m}
    save("archive_diagnostics.json",{"component_sizes":dict(sorted(Counter(counts.values()).items())),
                                       "component_count":len(counts),"vote_patterns":pattern_rows,"costs":costs})
    lines=["# CPU返修敏感性实验", "", "新比较均为返修期探索性分析；原冻结模型不变。", "",
           "| 模型 | 开发约束可行 | Macro-F1 | unsafe recall | safe specificity |", "|---|---|---:|---:|---:|"]
    m=report["original_frozen"]
    lines.append(f"| 原冻结模型 | 原配置 | {m['macro_f1']:.4f} | {m['unsafe_recall']:.4f} | {m['safe_specificity']:.4f} |")
    for name in configs:
        m=report[name]["constrained"]
        if m is None:lines.append(f"| {name} | 否 | — | — | — |")
        else:lines.append(f"| {name} | 是 | {m['macro_f1']:.4f} | {m['unsafe_recall']:.4f} | {m['safe_specificity']:.4f} |")
    with (OUT/"SUMMARY.md").open("x",encoding="utf-8") as handle:handle.write("\n".join(lines)+"\n")
    save("run_status.json",{"status":"completed_pending_audit","elapsed_seconds":time.time()-started,
                            "original_replay_max_error":residual,"confirmation_records":513,"development_records":1760})
    print("COMPLETED",str(OUT),flush=True)

if __name__ == "__main__":
    main()
