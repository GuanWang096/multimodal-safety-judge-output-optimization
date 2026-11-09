# Optimizing Multimodal Safety Judge Outputs for Auditable Enterprise Model Acceptance

## Scope

This repository contains the code and non-proprietary artifacts needed to inspect and reproduce the article's output-layer analysis from archived judge scores. It also contains the evidence-record, Hyperledger Fabric, RGG-PBFT, capacity-analysis, and exploratory sequential reputation-update code used by the reported experiments.

The repository does not redistribute MMDS, BeaverTails-V, Falcon-test media, model checkpoints, or third-party framework binaries. Obtain those materials from their original providers under the applicable licenses. GPU inference is therefore a separate step. The archived primary predictions and formal result files allow the statistical analysis to be checked without downloading model weights.

The file `src/multijudge/evidence_v2.py` retains a historical internal filename for import compatibility. It implements the canonical evidence record; “Evidence V2” is not the method or product name.

The file `results/capacity/replay_keys.json` contains deterministic, replay-only experiment keys that are intentionally public and must never be reused as production credentials.

## Directory map

- `src/multijudge/`: judge interfaces, calibration, fusion, metrics, and evidence-record code.
- `src/rggpbft/`: grouped consensus implementation and analysis helpers.
- `src/fabric_chaincode/`: Fabric chaincode source without vendored dependencies.
- `src/multijudge_workflows/`: experiment, freeze, audit, replay, and system drivers.
- `results/mmds/`: frozen MMDS split, configuration, raw score evidence, predictions, and confirmatory results.
- `results/external/`: analysis-ready BeaverTails-V and Falcon-test artifacts.
- `results/replay/`: canonical evidence-record workload and formal replay summaries.
- `results/capacity/`: formal local-capacity aggregates and integrity audit.
- `results/reputation/`: machine-readable sequential reputation-update results.

Failed consistency-view and source-prior-residual experiments, smoke runs, superseded MJ5 runs, caches, raw restricted media, model weights, and Fabric vendor code are excluded.

The repository has only two top-level directories: `src/` and `results/`. Subdirectories separate executable components and evidence classes only where filenames would otherwise collide or different runtimes are required.

## Environment

Python 3.10 or newer is required. For score analysis and tests:

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

Install `requirements-gpu.txt` only for GPU model inference. Docker, IPFS, Hyperledger Fabric, Go, and Linux traffic-control support are required for the corresponding systems experiments. Those measurements were originally made on one host and should not be interpreted as multi-host production capacity.

Repository-wide checksum inventories are intentionally omitted. Digest fields that are part of the evaluated evidence-record, model-freeze, contract, or consensus interfaces remain in the corresponding code and result artifacts because they are required to reproduce those mechanisms.

## Primary 513-record analysis

The following command regenerates the confirmatory result and prediction files from the archived score evidence, frozen configuration, split manifest, labels, and analysis plan:

```bash
python src/multijudge_workflows/evaluate_softfusion_confirmatory.py \
  --manifest results/mmds/data/confirm-unlabeled.jsonl \
  --labels results/mmds/data/confirm-labels.jsonl \
  --result-root results/mmds/confirm \
  --raw-freeze results/mmds/confirm/raw_output_freeze.json \
  --fusion-config results/mmds/fit/fusion_config.json \
  --analysis-plan results/mmds/fit/confirmatory_analysis_plan.json \
  --output-root results/reproduced/softfusion-confirmatory
```

Expected headline values are Macro-F1 0.8096 for frozen fusion, 0.7512 for 2-of-3 score-binarized majority over the same judge scores, and a difference of 0.0583. The component-bootstrap 95% confidence interval is [0.0257, 0.0942], and the one-sided component-randomization p value is 0.00030.

## Sequential judge-reputation diagnostic

This separate diagnostic uses an archived four-judge pool, 78 validation records, and 330 test records. It performs stepwise class-conditional reliability updates from reference-label feedback. It is not the frozen three-judge fusion model, a deployed online update service, or Byzantine-validator detection.

```bash
python src/multijudge_workflows/run_mj3_mj4.py \
  --test-dir results/reputation/test \
  --test-manifest src/multijudge_workflows/configs/mmds_test_formal.json \
  --validation-dir results/reputation/val \
  --validation-manifest src/multijudge_workflows/configs/mmds_val_formal.json \
  --freeze results/reputation/validation_frozen.json \
  --output-dir results/reproduced/sequential-reputation
```

The formal artifact executes cumulative counts, exponential decay 0.90, and a sliding window of 10. The unsupported 37,800-run validator-placement table from an earlier draft is not included.

## Tests and extension points

Run the repository's unit tests with:

```bash
python -m pytest src/multijudge_workflows/tests
```

Run the Fabric chaincode tests from its Go module:

```bash
cd src/fabric_chaincode
go test ./...
```

External-dataset scripts are supplied, but regenerating raw judge scores requires the licensed source records and model checkpoints. The evidence replay and MJ5 scripts preserve the exact evaluated interfaces; full re-execution additionally requires the Docker, IPFS, and Fabric services described by their command-line help and source comments.

The consensus code applies `ceil((n + t + 1) / 2)`, with `t = floor((n - 1) / 3)`, to local groups and the leader committee. Boundary tests cover the membership sizes used in the article.

## Repository version

The initial-submission tag `v1.0.0` remains at commit `de61496dd6195bc54d9cec205f0f117534d2824f`. Revision release `v1.1.0` adds the revision diagnostics and two adopted local protocol guards: strict integer sender identities and retirement of lower-view voting after a higher-view report. It preserves the original frozen statistical configuration and results. Historical performance measurements predate these two guards; the new CPU regressions are not replacement performance measurements. The separate 330-record sequential judge-reputation diagnostic remains outside the 513-record confirmatory claim hierarchy.

## Revision diagnostics

`src/revision/` contains relocated experiment drivers and bounded protocol checks. `results/revision/` contains the retained revision results, including unsuccessful settlement cases, and minimal numeric development inputs. Relocation changes paths and output destinations, not analysis algorithms. Original manifests may retain historical workspace-relative paths; these are provenance records, not current execution instructions.

From the repository root, the following checks require only a CPU and the analysis/test dependencies:

```bash
python src/revision/run_cpu.py
python src/revision/run_archive.py
python src/revision/run_qualification.py
python src/revision/summarize_external.py
python src/revision/run_adopted_checks.py
python src/revision/run_cross_view_traces.py --session cross-view
python src/revision/run_protocol_checks.py
python src/revision/summarize_matrix.py
python src/revision/verify_architecture.py --session architecture-formal-20261031
```

Fresh outputs go to `results/reproduced/revision/`; generators refuse to replace existing runs. The retained outputs are under `results/revision/{cpu,archive,protocol,system}`. The CPU analysis refits exploratory alternatives on 1,760 development records and evaluates the archived 513-record confirmation set. Its design is post hoc; it does not replace the original confirmatory result. Minimal inputs retain sample/component identifiers, policy labels, hashes and numeric scores, without source media or conversations.

The protocol entry point reruns 28 sender-ingress cases (24 alias rejections and four controls), 34 voting-retirement cases with 80 checks, and 30 existing regressions. The separate cross-view driver runs eight fixed schedules with 128 assertions. These are bounded in-process tests, not performance or production-authentication tests. The retained node-local-key fixture results contain public keys and signatures only; fresh private keys are generated at runtime and are not published.

The system archive contains all 18 retained settlement runs (1,728 executions of the 96-record capacity workload), including the 79 unsuccessful settlements, and the 144-execution architecture diagnostic. The 96 selected J3/J4 evidence bundles are supplied under `results/capacity/bundles/`; they contain archived outputs and media references, not original media. Offline extraction can be rerun without Docker. Re-executing `run_settlement.py`, `run_architecture.py`, or container-key drivers requires the documented WSL/Linux, Docker, Fabric and IPFS setup and creates service state. The original sender probe records pre-fix behavior; its expectations and source version must be matched to that historical result, rather than applied as pass criteria for the corrected verifier. Use `run_adopted_checks.py` for current guards. Original collector versions differ; the retained matrix is descriptive, not a matched causal speed comparison. The initial staged replay and revision local-service diagnostics do not establish independently administered cross-organization deployment.

## License and third-party assets

The authors' original software and accompanying software documentation are licensed under the [Apache License 2.0](LICENSE). This permits use, modification and redistribution, including commercial use, subject to its terms. Retain the license and applicable notices when redistributing, and identify changes to modified files.

This software license does not relicense third-party datasets, checkpoints, dependencies, or dataset-derived artifacts. Those remain subject to their providers' terms; in particular, the MMDS and BeaverTails-V sources carry non-commercial restrictions. See [THIRD_PARTY.md](THIRD_PARTY.md) for the asset-specific sources and scope. Restricted raw media and model weights are not redistributed. The public code repository is separate from the journal's supplementary-material upload.
