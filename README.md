# Optimizing Multimodal Safety Judge Outputs for Auditable Enterprise Model Acceptance

Article: **Optimizing Multimodal Safety Judge Outputs for Auditable Enterprise Model Acceptance**  
Journal: **Multimedia Systems**  
Repository snapshot: **2026-09-02**

Authors: Jinxin Zhang, Guan Wang, Zhipeng Ruan, Changsheng Wan, Yunhao Xia, and Weichen Lu.

Affiliations:

1. School of Cyber Science and Engineering, Southeast University, Nanjing, China.
2. ZTE Corporation, Shenzhen, China.

Corresponding author: Guan Wang, `guan.wang@seu.edu.cn`.

Canonical repository: <https://github.com/GuanWang096/multimodal-safety-judge-output-optimization>

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

Expected headline values are Macro-F1 0.8096 for frozen fusion, 0.7512 for same-output 2-of-3 majority, and a difference of 0.0583. The component-bootstrap 95% confidence interval is [0.0257, 0.0942], and the one-sided component-randomization p value is 0.00030.

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

## Repository release

This is the clean public repository for the current article. It is independent of the earlier 330-record reliability-aggregation manuscript because that history describes a different method and evidence base. A versioned release tag and archival snapshot should be created after the submission files are frozen. No open-source license is asserted until the authors' institutions approve one. The code repository is not a journal ESM upload.
