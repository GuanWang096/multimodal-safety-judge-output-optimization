# BeaverTails-V Frozen External Transfer on AutoDL

This run uses all 1,180 records from the BeaverTails-V evaluation split. The
three MMDS calibrators, logistic fusion coefficients, feature standardization,
and decision thresholds remain frozen. External labels are not uploaded to the
GPU host.

## Upload and Extract

Upload both archives to `/root` and run:

```bash
cd /root
tar -xzf beavertails-v-external-code-20260802.tar.gz -C /root/src
tar -xzf beavertails-v-external-images-20260802.tar.gz -C /root/datasets
cd /root/src
sed -i 's/\r$//' src/multijudge_workflows/autodl_*beavertails*.sh
chmod +x src/multijudge_workflows/autodl_*beavertails*.sh
```

No `pip install` is needed when the previous soft-fusion environment remains
available. The scripts check CUDA, model paths, the 1,180-record manifest, and
all image paths before inference.

## Low-Cost Smoke

Run three records per judge before the formal job:

```bash
cd /root/src
./src/multijudge_workflows/autodl_run_beavertails_external_all.sh smoke
```

The smoke phase validates only execution and score integrity. It does not access
external labels or select a model, calibrator, fusion weight, or threshold.

## Formal Inference

After all smoke checks pass:

```bash
cd /root/src
./src/multijudge_workflows/autodl_run_beavertails_external_all.sh formal
```

Estimated RTX 5090 wall time is approximately 3-5 hours in total. SafeWork is
expected to account for most of the runtime. Resume is supported because each
judge appends completed sample IDs to its own `judgments.jsonl`.

## Freeze and Package Raw Outputs

After all three formal runs complete:

```bash
cd /root/src
./src/multijudge_workflows/autodl_freeze_beavertails_outputs.sh
```

Download:

- `/root/beavertails-v-external-raw-20260802.tar.gz`

The archive contains raw judge outputs and their label-free freeze record. Final
metrics are computed locally only after the archive is downloaded.
