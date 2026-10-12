# Significance Test: Baseline vs Reliability-Aware (Full stage)

Paired bootstrap (3000 resamples, 95% CI, pooled n=5245 lesions across 5 seeds per config,
resampled jointly so both models see the exact same bootstrap lesion set each iteration) on
macro F1, comparing the reliability-aware model (Full stage) against the **best** conventional
baseline for each modality_config (same "best baseline" picks as
`phase4_baseline_evaluation_summary.md`). Per-lesion predictions:
`predictions_baselines.csv` (new, this step) and `predictions_full_stage_gpu0/1.csv` (existing).

| modality_config | best baseline | delta F1 (reliability - baseline) | 95% CI | verdict |
|---|---|---|---|---|
| C     | unimodal     | -0.0686 | [-0.0939, -0.0424] | **significant loss** |
| D     | unimodal     | -0.0292 | [-0.0540, -0.0048] | **significant loss** |
| M     | unimodal     | +0.0034 | [-0.0075, +0.0143] | not significant (tied) |
| C,D   | naive_concat | +0.0034 | [-0.0251, +0.0299] | not significant (tied) |
| C,M   | gated_fusion | -0.0404 | [-0.0672, -0.0138] | **significant loss** |
| D,M   | gated_fusion | -0.0161 | [-0.0415, +0.0098] | not significant (tied) |
| C,D,M | gated_fusion | +0.0172 | [-0.0238, +0.0546] | not significant (tied) |

## This changes the headline, not just adds a caveat

The raw point-estimate deltas in `phase4_baseline_evaluation_summary.md` made C,D and C,D,M look
like wins for the proposed model (+0.0043, +0.0152 F1). Under a proper paired bootstrap **neither
is statistically significant** — the C,D,M "win" specifically has a 95% CI of [-0.024, +0.055],
comfortably straddling zero. So the correct statement for the paper is:

**The reliability-aware model does not significantly outperform the best conventional baseline on
macro F1 at any of the 7 modality configs.** At 4 configs (M, C,D, D,M, C,D,M) the two are
statistically tied; at 3 configs (C, D, C,M) the reliability-aware model is *significantly worse*.
There is no config where it wins significantly.

This is not a negative result for the project overall — it is exactly consistent with CLAUDE.md's
stated success criteria ("do not optimize for +1.2% accuracy... success = better calibrated,
graceful degradation, disagreement predicts error, even if F1 improvement is modest"). But it
means the paper's accuracy claims must be written as "statistically indistinguishable from strong
baselines at full/near-full modality" — not as an accuracy win, and the single-modality
regressions (C, D, C,M) need to be reported as an honest limitation, not omitted. "Simpler to
deploy (one model instead of 7) and better calibrated in that regime" is NOT a safe claim either
— see the ECE significance test below, which the F1 result above prompted us to also check
properly rather than trust the point estimates.

## ECE (calibration) significance test — same method, applied to calibration

The same paired bootstrap, run on ECE instead of macro F1 (`ece(y_true, y_probs)`, matching
`metrics.py`'s implementation exactly, delta = reliability_ECE - baseline_ECE so **negative means
reliability-aware is better calibrated**):

| modality_config | best baseline | delta ECE (reliability - baseline) | 95% CI | verdict |
|---|---|---|---|---|
| C     | unimodal     | +0.0691 | [+0.0551, +0.0808] | **significantly WORSE calibrated** |
| D     | unimodal     | +0.0969 | [+0.0846, +0.1092] | **significantly WORSE calibrated** |
| M     | unimodal     | +0.0611 | [+0.0440, +0.0745] | **significantly WORSE calibrated** |
| C,D   | naive_concat | +0.0507 | [+0.0376, +0.0626] | **significantly WORSE calibrated** |
| C,M   | gated_fusion | -0.0442 | [-0.0558, -0.0296] | **significantly BETTER calibrated** |
| D,M   | gated_fusion | -0.0025 | [-0.0145, +0.0094] | not significant (tied) |
| C,D,M | gated_fusion | -0.0036 | [-0.0165, +0.0072] | not significant (tied) |

**This is a materially worse picture than the point-estimate summary in
`phase4_baseline_evaluation_summary.md` suggested.** The proposed model is significantly *worse*
calibrated than the best baseline at 4 of 7 configs — including C,D, which the original write-up
called "better calibrated at C,M/D,M/C,D,M, worse at C,D" without knowing which differences were
real. It is significantly *better* calibrated at exactly one config (C,M). At the two configs
that matter most for a "full modality, deployed system" story (D,M and C,D,M) calibration is
statistically **tied**, not better.

Putting the F1 and ECE results together: at C,D,M (the config the paper would naturally lead
with), the reliability-aware model is statistically indistinguishable from the best conventional
baseline on BOTH macro F1 and calibration. The one place it's unambiguously and significantly
better on both counts simultaneously is nowhere in this table — C,M is significantly better
calibrated but not significantly different on F1, and no config is a win on both. **The honest
paper framing is not "our model wins" on any axis measured here — it's "statistically comparable
accuracy and calibration to strong baselines at full modality, using one model instead of seven,
with real regressions in reduced-modality operation that need to be disclosed."** The disagreement
/ selective-prediction results (Phase 6-7, AURC / error-detection AUROC) — tested next — were the
remaining place the proposed model's case could still be made, since accuracy and calibration
alone don't carry it.

## Disagreement significance test — the last remaining card, and it doesn't help either

This is an internal check (no baseline comparison possible — B1-B4 have no per-modality branch
outputs to disagree with each other), asking: does combining disagreement with entropy
(`risk_score = 0.5*entropy_norm + 0.5*disagreement_norm`, as used throughout Phase 6-7) actually
detect errors better than entropy alone, or does adding disagreement just add noise? Per-lesion
entropy, disagreement, and correctness dumped via `save_risk_scores.py` (35 cells: 4 multi-modality
configs x 5 seeds — single-modality configs have no disagreement signal, excluded), then paired
bootstrap (3000 resamples, pooled n=5245 per config) on AURC (lower = better selective prediction)
and error-detection AUROC (higher = better), delta = combined - entropy_alone:

| modality_config | AURC entropy | AURC combined | delta AURC | verdict (AURC) | err-AUROC entropy | err-AUROC combined | delta err-AUROC | verdict (err-AUROC) |
|---|---|---|---|---|---|---|---|---|
| C,D   | 0.1339 | 0.1375 | +0.0036 | not significant | 0.7763 | 0.7614 | -0.0149 | **significantly WORSE** |
| C,M   | 0.2099 | 0.2062 | -0.0036 | not significant | 0.7167 | 0.7151 | -0.0015 | not significant |
| D,M   | 0.1432 | 0.1529 | +0.0097 | **significantly WORSE** | 0.7743 | 0.7548 | -0.0195 | **significantly WORSE** |
| C,D,M | 0.1201 | 0.1263 | +0.0061 | **significantly WORSE** | 0.7805 | 0.7641 | -0.0164 | **significantly WORSE** |

**Adding disagreement to entropy does not improve error detection at any of the 4 configs tested,
and makes it significantly worse at 3 of 4 (D,M and C,D,M on both metrics; C,D on error-AUROC).**
Only C,M is a statistical wash. This directly contradicts the framing in CLAUDE.md's success
criteria ("disagreement/uncertainty measurably predicts error") as currently implemented with
lam=0.5 — entropy alone is a better selective-prediction signal than entropy+disagreement combined,
at the model's default weighting. Two honest paths forward for the paper, neither of which this
analysis alone can pick between: (1) report this as a genuine negative result — the reliability
mechanism's disagreement signal, as built, does not add selective-prediction value beyond plain
softmax entropy; or (2) if pursued further, sweep lam away from 0.5 (the combined score's weight
on disagreement vs entropy) to check whether some other mixing weight recovers an advantage before
concluding the mechanism itself is not useful — that would be new analysis, not re-litigating this
result on the same data.

Combined with the stress-test finding (disagreement/reliability mechanism gives no measurable
protection against a contradictory dominant modality, flat across all 6 ablation stages) and the
F1/ECE results above, **none of the three quantitative axes tested this session (accuracy,
calibration, selective prediction via disagreement) show a significant advantage for the
reliability-aware model over simpler alternatives** at the standard operating point (lam=0.5,
Full stage, C,D,M). This is the honest state of the evidence as of this analysis — worth stating
plainly rather than discovered by a reviewer.

## Files

- `save_predictions_baselines.py` (src/) — dumps per-lesion baseline predictions
- `save_risk_scores.py` (src/) — dumps per-lesion entropy/disagreement/correctness for the Full-stage model
- `predictions_baselines.csv` — 78,675 rows (75 cells x 1049 lesions)
- `risk_scores_full_stage.csv` — 36,715 rows (35 cells x 1049 lesions)
- `significance_test_baseline_vs_reliability.csv` — macro F1 significance table
- `significance_test_ece_baseline_vs_reliability.csv` — ECE significance table
- `significance_test_disagreement_vs_entropy.csv` — AURC / error-AUROC significance table
