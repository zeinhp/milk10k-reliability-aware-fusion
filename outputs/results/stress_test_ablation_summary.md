# Extended Contradiction Stress Test: All 6 Ablation Stages (A1-Full)

180 cells = 5 seeds x 6 ablation stages x 3 swap targets (C/D/M) x 2 swap types (same_class,
cross_class). Delta = swapped macro F1 - that (seed, stage)'s own normal no-swap C+D+M macro F1
(from the Phase 6-7 sweep), averaged across 5 seeds.

## Delta F1 by ablation stage and swap condition

| Stage | C same | C cross | D same | D cross | M same | M cross |
|---|---|---|---|---|---|---|
| A1 (no reliability mechanism at all) | +0.032 | -0.122 | +0.038 | **-0.348** | +0.015 | -0.021 |
| A2 (+ reliability weighting) | +0.032 | -0.124 | +0.023 | **-0.345** | -0.004 | -0.032 |
| A3 (+ modality dropout) | +0.035 | -0.128 | +0.036 | **-0.338** | +0.013 | -0.020 |
| A4 (+ disagreement) | +0.063 | -0.093 | +0.044 | **-0.329** | +0.002 | -0.017 |
| A5 (+ calibration) | +0.049 | -0.126 | +0.033 | **-0.345** | +0.018 | -0.043 |
| Full (+ abstention) | +0.036 | -0.111 | +0.050 | **-0.350** | +0.002 | -0.013 |

## Headline finding: the reliability mechanism provides NO measurable protection against a
## contradictory dermoscopy image, at any ablation stage.

D-cross_class damage is flat at approximately -0.33 to -0.35 macro F1 across every single stage,
from A1 (the plain baseline with none of the 5 proposed components) all the way to Full (all 5
components active). Adding reliability weighting (A2), modality dropout (A3), disagreement-aware
fusion (A4), calibration (A5), or abstention (Full) makes essentially **zero difference** to how
badly the model is fooled by a dermoscopy image that contradicts the other evidence. This is a
clean negative result, not noise -- the effect size and its flatness are consistent across all 5
seeds and all 6 stages.

C-cross_class shows a similar flat pattern (-0.09 to -0.13 across stages, no clear monotonic
trend with added components) -- A4 (disagreement) is nominally the least-damaged stage for C, but
the difference from A1 is small and not a clean progression, so this reads as noise rather than a
real effect specific to the disagreement mechanism.

M-cross_class stays small throughout (-0.01 to -0.04), consistent with M being the weakest,
least-trusted modality across this entire study.

## Interpretation for the paper

The 5 ablation components (A2-A5, Full) were designed to make the model robustness-aware --
reliability weighting and disagreement-aware fusion in particular sound like they should reduce
over-reliance on a single contradictory modality. This experiment shows they do NOT do that in
practice, at least not against a dermoscopy contradiction. A plausible explanation: modality
dropout (A3) and reliability weighting (A2) were trained against *missing* modalities (zeroed-out
input, availability=0), not *actively wrong* modalities (a real image, availability=1, that just
happens to contradict the diagnosis) -- the model never saw a training signal telling it "this
input is present but untrustworthy," so there's no reason to expect it generalizes to that failure
mode. This is a genuinely useful negative result and a concrete direction for future work
(training-time modality corruption/contradiction injection, not just modality dropout), and should
be reported as a limitation rather than smoothed over -- it's arguably the most important finding
of the whole reliability-aware evaluation.

## File

`stress_test_contradiction_all_stages_combined.csv` (180 rows: seed, ablation_stage,
swap_modality, swap_type, n_samples, macro_f1, balanced_acc)
