# MILK10k Reliability-Aware Multimodal Skin-Lesion Diagnosis

Code, frozen data splits, and evaluation outputs for a reliability-aware multimodal fusion study
on the MILK10k dermatology dataset (clinical image, dermoscopy image, and tabular metadata). The
full frozen research protocol (dataset handling, splits, model architecture, success criteria,
statistical testing procedure) is documented in `CLAUDE.md`.

**Headline finding:** the proposed reliability-aware fusion model does **not** significantly
outperform conventional multimodal baselines (unimodal, naive concatenation, late fusion, gated
fusion) on accuracy (macro F1), calibration (ECE), or selective prediction (disagreement vs.
entropy), under paired bootstrap significance testing. See `outputs/results/significance_test_summary.md`
for the full breakdown, and the accompanying paper for discussion.

## 1. Setup

```
pip install -r requirements.txt
```

Extract the MILK10k dataset locally from the original archive downloaded from the
[ISIC MILK10k Challenge](https://challenge.isic-archive.com/):

```
python src/extract_local.py --zip "<path to the downloaded MILK10k archive>" --out "<extraction target folder>"
```

Point `configs/config.yaml`'s `data.data_root` at that extracted folder.

## 2. Phase 1 — Dataset audit

```
cd src
python audit_dataset.py
```

Review `outputs/results/dataset_audit.json`. If any class falls below the rarity threshold,
decide explicitly how to handle it (merge / exclude / accept with a sensitivity-analysis caveat)
before proceeding.

## 3. Phase 2 — Freeze the split

```
python make_splits.py
```

Writes `splits/seed{0..4}_{train,val,test}.csv`, a lesion-level 70/10/20 stratified split. These
files are the splits actually used throughout this study and are included in this repository for
exact reproducibility; once training starts, treat them as frozen.

## 4. Phase 3-8 — Train and evaluate

```
python src/run_all.py --device cuda:0
```

See `run_all.py`'s docstring for the full set of training groups and `--dry_run` to preview the
command list before committing GPU time. Each run auto-resumes from its own checkpoint if
interrupted (`train.py --no_resume` to force a clean restart).

This project was trained across multiple GPUs in practice; adjust `--device` / `--group` to your
own hardware. If training across more than one machine, run Phase 1-2 once and copy the resulting
`splits/` folder (do not regenerate it) to every machine, so all runs train and evaluate on
byte-identical splits.

## 5. Evaluation and analysis scripts

- `evaluate.py` / `evaluate_baselines.py` — evaluate a checkpoint across all 7 modality
  configurations; computes macro F1, balanced accuracy, AUROC, ECE, Brier, NLL.
- `save_predictions.py` / `save_predictions_baselines.py` — dump per-lesion predictions.
- `save_risk_scores.py` — dump per-lesion entropy, cross-modal disagreement, and correctness for
  the reliability-aware model.
- `bootstrap_ci.py` — paired bootstrap confidence intervals and significance tests (3000
  resamples) for all headline comparisons.
- `stress_test.py` — controlled contradictory-evidence swap experiment.
- `subgroup_analysis.py`, `analyze_outliers.py` — subgroup performance/calibration and outlier
  analysis.

## 6. Results

All result tables, figures, and significance-test outputs referenced in the paper are in
`outputs/results/`, including:

- `evaluate_baselines_sweep.csv`, `evaluate_reliability_aware_sweep_combined.csv` — raw
  per-checkpoint evaluation results.
- `significance_test_*.csv` / `significance_test_summary.md` — paired bootstrap significance
  tests for macro F1, ECE, and disagreement-vs-entropy selective prediction.
- `predictions_baselines.csv`, `predictions_full_stage_gpu0/1.csv` — per-lesion predictions.
- `risk_scores_full_stage.csv` — per-lesion entropy/disagreement/correctness.
- `confusion_matrix_cdm.png`, `reliability_diagrams.png` — figures used in the paper.

Model checkpoints (211 files, several GB) are not included in this repository; contact the
corresponding author if you need a specific trained checkpoint.

## License

MIT — see `LICENSE`.
