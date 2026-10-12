# Phase 4 Baseline Evaluation Summary

Evaluates all 75 baseline checkpoints (B1 unimodal x3, B2 naive_concat x4, B3 late_fusion x4,
B4 gated_fusion x4 modality configs, x5 seeds each) on the held-out **test** split via
`evaluate_baselines.py`, and compares them against the proposed reliability-aware model at its
**Full** ablation stage (from the Phase 6-7 sweep, `evaluate_reliability_aware_sweep_combined.csv`).
Each baseline checkpoint is evaluated only at the single modality_config it was trained on (see
`evaluate_baselines.py` docstring for why this is the correct comparison, not a bug).

Raw data: `evaluate_baselines_sweep.csv` (75 rows), aggregates in `baseline_agg_by_family_config.csv`
and `reliability_full_agg_by_config.csv` (mean +/- std across 5 seeds).

## Headline result: UPDATED after significance testing — no significant win anywhere

**This section originally read the raw point-estimate deltas below as two wins for the proposed
model (C,D and C,D,M). A paired bootstrap significance test (see
`significance_test_summary.md`) shows neither is actually statistically significant.** Read that
file for the full corrected picture; the short version is below the table.

Comparing reliability-aware (Full stage) against the **best conventional baseline for each
modality_config** (not just gated_fusion — whichever of B1-B4 scores highest at that config):

| modality_config | best baseline (family) | baseline F1 | baseline ECE | reliability_full F1 | reliability_full ECE | delta F1 (point est.) | significance (paired bootstrap) |
|---|---|---|---|---|---|---|---|
| C       | unimodal     | 0.3860 | 0.1586 | 0.3148 | 0.2264 | -0.0712 | **significant loss** |
| D       | unimodal     | 0.4436 | 0.1363 | 0.4151 | 0.2338 | -0.0286 | **significant loss** |
| M       | unimodal     | 0.1595 | 0.0464 | 0.1570 | 0.1051 | -0.0024 | not significant (tied) |
| C,D     | naive_concat | 0.4529 | 0.1051 | 0.4572 | 0.1525 | +0.0043 | not significant (tied) |
| C,M     | gated_fusion | 0.3970 | 0.1442 | 0.3511 | 0.1004 | -0.0459 | **significant loss** |
| D,M     | gated_fusion | 0.4486 | 0.1358 | 0.4302 | 0.1332 | -0.0183 | not significant (tied) |
| C,D,M   | gated_fusion | 0.4502 | 0.1085 | 0.4654 | 0.0926 | +0.0152 | not significant (tied) |

**Corrected conclusion: the reliability-aware model does not significantly beat the best
conventional baseline's macro F1 at ANY of the 7 modality configs.** It is statistically tied at
4 configs (M, C,D, D,M, C,D,M — including the near-full-modality configs that looked like wins
before testing) and *significantly worse* at 3 (C, D, C,M — all single- or two-modality configs
missing D).

This is not necessarily disqualifying per CLAUDE.md's own success criteria ("do not optimize for
+1.2% accuracy over baseline... even if improvement over strong baselines is modest"), and it
makes architectural sense: B1-B4 baselines are 15+60 *separately trained, config-specialized*
models (one dedicated model per modality subset), while reliability_aware is a *single* checkpoint
per seed handling all 7 configs via modality dropout — a fundamentally harder training regime.
But it means the paper cannot claim any accuracy win, statistically significant or otherwise — the
honest framing is "statistically indistinguishable from strong baselines while being one model
instead of seven," plus the significant single-/pair-modality regressions need to be reported as
a real limitation, not smoothed over.

## Calibration (ECE): also mixed, not uniformly better

The Phase 6-7 write-up's calibration claim ("ECE improves 0.24 -> 0.116 as modalities are added")
was about the reliability-aware model's *own* internal degradation curve, not a comparison to
baselines. Compared to the best conventional baseline per config:

- **Better calibrated** at C,M (0.100 vs 0.144), D,M (0.133 vs 0.136, roughly tied), C,D,M (0.093
  vs 0.108), C,D (0.153 vs 0.105 — actually *worse* here).
- **Substantially worse calibrated** at single modalities: C (0.226 vs 0.159), D (0.234 vs 0.136).

So the reliability-aware model's calibration advantage is real but conditional on multiple
modalities being present — under single-modality operation it is both less accurate AND worse
calibrated than a plain unimodal classifier. Combined with the earlier stress-test finding
(contradiction damage is flat across all ablation stages, i.e. the reliability mechanism gives no
measurable protection against a contradictory dominant modality), this reinforces a consistent
pattern across this evaluation phase: **the reliability-aware mechanism's benefits are real but
narrow** — most visible near full-modality operation, essentially absent or reversed under
single-modality / degraded conditions. Worth stating plainly in the paper's limitations section
rather than leading with the C,D,M numbers alone.

## Files

- `evaluate_baselines.py` (src/) — the evaluation script (new, this phase)
- `evaluate_baselines_sweep.csv` — raw 75-row results
- `baseline_agg_by_family_config.csv`, `reliability_full_agg_by_config.csv` — seed-averaged aggregates
