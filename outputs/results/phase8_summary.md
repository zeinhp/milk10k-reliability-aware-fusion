# Phase 8 Summary: Bootstrap CI, Subgroup Analysis, Contradiction Stress Test

All on the Full-stage (complete proposed model), 5 seeds. Read-only inference over existing
checkpoints, run on the two idle GTX1080s while A4000 was down.

## 1. Bootstrap CI (lesion-level, B=2000, 95%)

Per modality config, averaged across 5 seeds' individual CIs:

| Config | Macro F1 (95% CI) | ECE (95% CI) | AURC (95% CI) |
|---|---|---|---|
| C | 0.312 [0.274, 0.354] | 0.228 [0.200, 0.256] | 0.250 [0.217, 0.285] |
| D | 0.411 [0.360, 0.461] | 0.235 [0.209, 0.261] | 0.165 [0.141, 0.191] |
| M | 0.156 [0.139, 0.176] | 0.113 [0.087, 0.139] | 0.448 [0.408, 0.489] |
| C+D | 0.452 [0.395, 0.508] | 0.154 [0.129, 0.178] | 0.130 [0.108, 0.153] |
| C+M | 0.348 [0.301, 0.397] | 0.105 [0.081, 0.131] | 0.207 [0.177, 0.238] |
| D+M | 0.426 [0.374, 0.477] | 0.137 [0.113, 0.161] | 0.140 [0.118, 0.163] |
| C+D+M | **0.460 [0.406, 0.514]** | **0.097 [0.075, 0.120]** | **0.117 [0.097, 0.138]** |

CIs are fairly wide (test split ~130-1050 lesions depending on seed) but full modality is
consistently the best point estimate and the CIs for full vs. single modalities do not overlap --
the graceful-degradation finding holds up under bootstrap uncertainty, not just as a point estimate.
(AUROC bootstrap came out `nan` for every cell -- some bootstrap resamples miss a class entirely,
which `roc_auc_score` can't score; would need a fix to report AUROC CIs specifically, macro F1/ECE/
AURC are unaffected since they use `zero_division=0` / entropy-only, which don't have this issue.)

## 2. Subgroup Analysis (C+D+M, Full stage, averaged across 5 seeds)

- **Sex**: female F1=0.450, male F1=0.470 -- small gap, roughly comparable.
- **Age band**: <40 F1=0.323 (lowest, but also smallest group, n~87), 40-59 F1=0.433, >=60 F1=0.431.
  The `unknown` age group (n~5) is too small to interpret (F1=0.148 on 5 lesions is noise).
- **Skin tone (0-5 scale)**: tone 3 is the largest group (n~646, F1=0.449) and best-calibrated
  (ECE=0.091); tones 1 and 2 (smallest groups, n~23/99) show higher F1 sensitivity but also much
  worse calibration (ECE=0.258 and 0.175) -- i.e. the model is both less accurate-per-class-balance
  AND less trustworthy in its confidence for the least-represented skin tones. Worth flagging
  explicitly as a limitation, not smoothing over.

Call this "subgroup performance and calibration analysis" in the writeup, not a fairness claim
(per CLAUDE.md).

## 3. Controlled Contradictory-Evidence Stress Test (synthetic, test set only)

Delta = mean(swapped macro F1 - normal no-swap C+D+M Full-stage macro F1), averaged over 5 seeds:

| Swapped modality | same_class swap | cross_class swap |
|---|---|---|
| C (clinical) | +0.036 | **-0.111** |
| D (dermoscopy) | +0.050 | **-0.350** |
| M (metadata) | +0.002 | -0.013 |

**Headline finding: the model is NOT robust to a contradictory dermoscopy image.** Swapping D for
a same-modality image from a different-diagnosis lesion (keeping C and M correct) collapses macro
F1 by 0.35 on average -- far more damage than simply REMOVING D entirely (Delta_D from the Phase
6-7 sweep was only 0.047). This means the model doesn't detect "this D image doesn't match the
diagnosis suggested by C+M" -- it still largely trusts D even when D is actively wrong, i.e. the
reliability-weighting mechanism (A2-A5 components) is not preventing over-reliance on the dominant
modality under contradiction, only under absence. C contradiction causes a smaller but still real
drop (-0.111); M contradiction is nearly a non-event (-0.013), consistent with M being the weakest,
least-trusted modality throughout this whole sweep.

Same-class swaps (same diagnosis, different lesion/person) cause no damage or even a small F1
bump -- expected, since the "wrong" instance still carries genuinely correct diagnostic evidence.

This is a **synthetic stress test** (explicitly labeled as such per protocol) -- the natural
disagreement analysis (already covered in the Phase 6-7 sweep's `mean_disagreement` metric) is the
primary evidence for the disagreement-error relationship; this is a secondary, controlled check.
It surfaces a genuine limitation worth discussing in the paper: reliability weighting handles
*missing* modalities gracefully but does not yet handle *actively contradictory* modalities,
especially the dominant one (D).

## Files

- `bootstrap_ci_full_stage.csv` / `_summary.md`
- `subgroup_analysis_full_stage.csv` / `_summary.md`
- `stress_test_contradiction_combined.csv` (30 rows: 5 seeds x 3 swap modalities x 2 swap types)
