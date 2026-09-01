# Falcon-test Prospective External Confirmation Protocol

## Purpose

Falcon-test is used to prospectively test whether the fixed three-judge soft-evidence adaptation improves multimodal response-safety classification beyond simple majority voting, and whether continuous scores add value beyond an equally adapted hard-label model.

The protocol is specified before the complete CSV is parsed or its labels are summarized. Public preview records with raw IDs 0--21 were visible during schema verification and are excluded permanently.

## Locked design

- Dataset: human-annotated `Falcon-test` from `zhangrjjj/Falcon`, revision `53341884f7760e434481d88da02d50668b19d566`.
- Source integrity: `Falcon_test.csv` SHA256 is `cc40bc880fb948f96494daac5e4b72f50984d3541e2f75a101aeca2a8d2ec0f4`.
- Grouping unit: identical decoded image and exact stripped query.
- Split: deterministic group hash, 40% adaptation and 60% untouched confirmation, without label stratification.
- Judges: Qwen3-VL-8B-Instruct, SafeWork-RM-Safety-7B, and MiniCPM-V-4.5 using already qualified soft-score adapters.
- Primary model: three calibrated probability logits, L2 logistic regression, `C=0.03`, no class weighting, and decision threshold `0.5`.
- Mechanism control: the same logistic configuration fitted to the three binary judge decisions.
- Primary baseline: same-pool two-of-three hard majority.
- Confirmation labels may be opened once, only after all confirmation predictions and fitted model parameters have been serialized and hashed.

No feature family, regularization strength, threshold, judge membership, split, sample subset, or category subset may be changed after freeze.

## Primary claims

`FALCON-C1` compares the adapted soft model with same-pool majority voting on confirmation Macro-F1. `FALCON-C2` compares the adapted soft model with the prespecified adapted hard-label logistic model. The two one-sided paired group-randomization tests form one Holm-corrected family. Paired group-bootstrap confidence intervals are also reported.

A failed endpoint is reported as a failed prospective confirmation, not converted into an exploratory success claim. MMDS-frozen zero-shot fusion and other logical voting rules are secondary stress tests.

## Data isolation

The GPU inference manifest contains no labels. Preparation writes adaptation and confirmation labels to separate files. All three judge runs must finish before adaptation labels are used. The fitted models, confirmation predictions, and their hashes must be frozen before the confirmation-label file is read by the evaluator.

The executable specification is `configs/falcon_prospective_protocol_v1.json`; the freeze record binds that file, the raw dataset hash, and the download manifest before bulk parsing.
