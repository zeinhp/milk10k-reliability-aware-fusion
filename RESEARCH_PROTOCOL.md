# Project: Reliability-Aware Multimodal Skin-Lesion Diagnosis on MILK10k

This file is the frozen research protocol for this project. Follow it exactly unless the user
explicitly changes a decision below. Do not re-litigate decisions already made here.

## Title

**Reliability-Aware Multimodal Skin-Lesion Diagnosis under Missing and Contradictory Evidence:
Calibration, Selective Prediction, and Graceful Degradation on MILK10k**

## Positioning

Novelty is NOT a new fusion architecture. Novelty is treating multimodal dermatology AI as a
**reliability problem** (calibration, missing evidence, contradictory evidence, selective
prediction) rather than solely a classification-accuracy problem. Do not chase SOTA accuracy.

Related work to explicitly cite and differentiate from (NOT to replicate as a contribution):
- MILK10k dataset paper (Combalia et al. / MILK study team) — introduces the dataset, hierarchical
  48→11 diagnosis taxonomy, baseline top-1 accuracy ~53.6% with hierarchical evaluation.
- JAAD companion paper — "Comparative and complementary diagnostic value of dermatoscopy and
  clinical close-up photography... from the MILK10k dataset" — already answers "how much does
  each modality contribute" (our RQ1). Frame RQ1 in this project as a **contextual baseline
  replication**, not a primary contribution. Primary contributions are RQ2 (missing evidence) and
  RQ3 (contradictory evidence / disagreement as a failure signal).

## Dataset — MILK10k (verified against the actual downloaded archive)

Source file: `archive (1).zip` (~360MB), originally from D:\Claude\milk10k on the user's machine.

Verified contents:
- `MILK10k_Training_Input/<lesion_id>/<isic_id>.jpg` — 5,240 lesions x 2 images = 10,480 images
  (one `clinical: close-up`, one `dermoscopic` per lesion).
- `MILK10k_Training_Metadata.csv` — columns: `lesion_id, image_type, isic_id, attribution,
  copyright_license, image_manipulation, age_approx, sex, skin_tone_class, site,
  MONET_ulceration_crust, MONET_hair, MONET_vasculature_vessels, MONET_erythema,
  MONET_pigmented, MONET_gel_water_drop_fluid_dermoscopy_liquid,
  MONET_skin_markings_pen_ink_purple_pen`. One row per image (10,480 rows).
  - `sex`: male/female, no missing observed in the full training metadata.
  - `skin_tone_class`: 0-5 (6-level scale).
  - `site`: head_neck_face, trunk, lower_extremity, upper_extremity, hand, foot, genital, and
    ~0.6% blank (missing) — must be handled with an explicit missing category, not dropped.
  - MONET_* columns are per-image concept scores (continuous, ~0-1) for ulceration/crust, hair,
    vasculature/vessels, erythema, pigmentation, gel/fluid artifact, skin markings/pen ink. These
    are available "for free" and are NOT currently used as model input, but MUST be used in the
    Section 23-equivalent error analysis to check whether high modality disagreement correlates
    with specific visual artifacts (hair, ink, fluid) rather than being pure model noise.
- `MILK10k_Training_GroundTruth.csv` — one row per **lesion** (5,240 rows), one-hot over 11
  columns: `AKIEC, BCC, BEN_OTH, BKL, DF, INF, MAL_OTH, MEL, NV, SCCKA, VASC`. This is the primary
  11-class label. The dataset's underlying diagnosis taxonomy is hierarchical (~48 fine-grained
  diagnoses collapsing into these 11 broad groups, per the MILK10k paper), and the ISIC MILK10k
  Benchmark explicitly cautions that flat accuracy alone is a weak metric for this taxonomy.
  DECISION: this project intentionally works at the flat 11-class broad-category level because
  the research question is about reliability, not taxonomy — state this explicitly in the paper
  as a scoping decision, do not silently ignore it.
- `MILK10k_Training_Supplement.csv` — keyed by `isic_id`: `diagnosis_full,
  diagnosis_confirm_type, invasion_thickness_interval`. Not a primary input; useful only for audit
  / error analysis (e.g. filtering to histopathology-confirmed subset for a sensitivity check).
- `MILK10k_Test_Input/` + `MILK10k_Test_Metadata.csv` — 479 lesions, 958 images, **no ground
  truth** (blind official ISIC Challenge test set, evaluated only via leaderboard submission).
  This is NOT the test split used for this project's own experiments — do not confuse the two.
  Optional bonus: submit the final trained model to the official MILK10k Benchmark leaderboard for
  an additional external-validation data point, but the project's primary train/val/test split
  below is carved out of the 5,240-lesion training set only.
- License: CC-BY-NC (non-commercial, attribution required).

## Experimental Split

- Unit of analysis: **lesion**, not image. Unit of split: lesion. Both images (clinical +
  dermoscopy) of a lesion must always stay in the same partition — never split them.
- Split ratio: 70 / 10 / 20 (train / val / test), stratified by the 11-class diagnosis label.
- Secondary balancing (best-effort, do not break class stratification for it): contributing
  centre, if this field is derivable/available.
- **Freeze the test set once. Never touch it during model development.**
- Run all headline experiments with **5 independent random seeds**; report mean ± SD, not
  single-run numbers.
- Before freezing: audit minimum per-class, per-split counts (Phase 1). If any of the 11 classes
  has too few lesions to stratify safely into 70/10/20 (e.g. < ~20-30 lesions total), document it
  and decide explicitly (do not silently merge or drop) — flag to the user rather than deciding
  unilaterally, since this changes what the primary metric represents.

**DECISION (made after Phase 1 audit on the real data, before any training ran):** MAL_OTH has
only 9 lesions total (6 train / 1 val / 2 test after stratified splitting) — far below the ~30
minimum, and its per-class metrics will be too noisy to interpret on their own (a single val
lesion). Rather than silently dropping or merging it, or waiting to see how bad its numbers look
before deciding (which would be post-hoc cherry-picking, not a principled decision), this project
commits NOW, before training, to a pre-planned two-way reporting scheme:
  1. Primary: Macro F1 (and all other headline metrics) over all 11 classes, MAL_OTH included.
  2. Sensitivity analysis: the same metrics recomputed over the 10 classes excluding MAL_OTH,
     reported alongside the primary numbers in every results table, not buried in an appendix.
Label MAL_OTH's own per-class numbers explicitly as descriptive/unreliable (n=9) wherever shown,
never presented as a reliable per-class estimate. Do not drop MAL_OTH from training or from the
frozen splits — this only affects how results are aggregated/reported at evaluation time, so it
costs no retraining and can be applied uniformly across all already-frozen seed0-4 splits.

## Modalities

- `C` = clinical close-up image
- `D` = dermoscopic image
- `M` = metadata (age_approx, sex, site, skin_tone_class — the 4 structured fields actually
  available; MONET_* concept scores are reserved for error analysis, not used as fusion input by
  default, since they are themselves model-derived annotations and using them as input muddies the
  "raw evidence" framing — revisit only if the user asks)

## Seven Modality Configurations (mandatory, not optional)

C, D, M, C+D, C+M, D+M, C+D+M — all seven must be evaluated for every model family. This is the
foundation of the missing-modality analysis, not just a unimodal-vs-trimodal comparison. Note in
the paper narrative that C, C+D, C+M are the more clinically realistic deployment scenarios
(telederm without a dermatoscope, specialist without structured metadata); D-only and M-only exist
for completeness of the factorial ablation, not because they are common deployment scenarios.

## Preprocessing

Train: Resize 256x256 -> RandomCrop 224x224 -> flip -> rotation ±15° -> mild color jitter ->
normalize (backbone-appropriate mean/std). Val/test: Resize -> CenterCrop 224x224 -> normalize,
no stochastic augmentation. Keep augmentation conservative — do not destroy diagnostic texture.

## Backbone

ConvNeXt-Tiny (primary), generic ImageNet initialization only. No dermatology-pretrained weights
(contamination risk with ISIC-derived pretraining data). Two independent image branches (E_C, E_D)
— do not share weights between clinical and dermoscopy encoders unless later shown necessary.

## Metadata Encoder

age_approx -> normalized scalar. sex, site, skin_tone_class -> learned embeddings, each with an
explicit "missing" category/token (never drop a lesion for missing metadata). Concatenate -> MLP
-> f_M in R^256.

## Baselines (must all be implemented before the proposed model)

- B1 Unimodal: C-only, D-only, M-only classifiers.
- B2 Naive concatenation: z = [f_C; f_D; f_M] -> classifier.
- B3 Late fusion: average of per-modality softmax outputs.
- B4 Gated/attention fusion: standard learnable modality weighting (this is the strongest
  baseline — the proposed model's improvement over B4 specifically is what justifies calling it
  "reliability-aware" rather than "just another attention fusion").

## Proposed Model — Reliability-Aware Fusion

For each modality m in {C, D, M}: f_m = E_m(x_m). Reliability estimator r_m = R_m(f_m) in [0,1].

**Reliability estimator training objective (this was the single biggest open gap in the protocol
and has now been decided — implement exactly this unless told otherwise):**
R_m is trained with an explicit auxiliary correctness-supervision loss, not left to emerge purely
from the end-to-end classification loss. Concretely: each modality branch also has its own
lightweight per-modality classifier head P_m(f_m) (used only during training, detached from the
main fusion gradient via stop-gradient / `.detach()` on f_m for this head so it doesn't distort
the shared encoder). Define correct_m = 1 if argmax P_m(f_m) == y else 0 for available modalities.
Train r_m with binary cross-entropy against correct_m:
  L_reliability = (1/|available modalities|) * sum_m BCE(r_m, correct_m)
Total loss = L_classification(final prediction, y) + lambda_r * L_reliability
This is what distinguishes the reliability estimator from plain learned attention (B4): B4's gate
is supervised only indirectly through the final classification loss; here r_m has a direct,
interpretable supervision target (was this modality's own evidence correct). Start lambda_r = 0.5,
tune on validation only.

Availability mask a_m in {0,1}. Normalized weight:
  alpha_m = a_m * exp(r_m) / sum_j a_j * exp(r_j)
Fusion: z = sum_m alpha_m * P_m_proj(f_m), where P_m_proj is a projection layer into a shared
latent space (separate from the auxiliary classifier head above). Final prediction:
  p(y|x) = softmax(g(z))

Missing-modality handling inside encoders: when a_m = 0, do NOT run E_m at all (skip the branch
entirely rather than zero-imputing pixels/features) — this avoids wasted compute and avoids the
model learning to interpret an all-zero image as a real input.

## Arbitrary Missing-Modality Training (modality dropout)

During training, stochastically sample which subset S ⊆ {C, D, M} (non-empty) is available for
each batch item, covering all 7 configurations over the course of training — not just full
C+D+M. This trains a single unified model p(y|S) for arbitrary modality availability rather than
7 separate models. Bootstrap a reasonable default: 50% of steps use full C+D+M, remaining 50%
uniformly sample one of the other 6 non-empty subsets. Tune this ratio on validation if time
allows; do not over-engineer this before baselines are working.

## Modality Disagreement

Per-modality softmax outputs p_C(y), p_D(y), p_M(y) (from the auxiliary per-modality heads).
Pairwise Jensen-Shannon divergence D_CD, D_CM, D_DM; global disagreement D_global = mean of the
three (only computable when all three modalities are available). Hypothesis to test, not assume:
higher D_global correlates with higher P(error).

## Controlled Contradictory-Evidence Stress Test (test set only)

Swap one modality's evidence across lesions while keeping the other two matched to the original
lesion (e.g. (C_i, D_i, M_j) for metadata contradiction, j != i). Run two separate controlled
conditions and do not mix them in one reported number:
1. Same-diagnosis-class swap (tests identity/context inconsistency at constant diagnosis).
2. Cross-diagnosis-class swap (tests genuine diagnostic contradiction).
This is a synthetic stress test — say so explicitly in the paper, and treat naturally occurring
disagreement (previous section) as the primary evidence for the disagreement-error relationship,
with this controlled experiment as a secondary robustness check.

## Uncertainty, Selective Prediction, Calibration

- Predictive entropy H(p), confidence = max_k p_k.
- Risk score R(x) = lambda * H(p) + (1 - lambda) * D_global; lambda tuned on validation only.
- Abstain if R(x) > tau; report coverage and selective risk; plot risk-coverage curve; report AURC
  and error-detection AUROC.
- Calibration: ECE, Brier score, NLL, reliability diagrams; temperature scaling fit on validation
  only. Report calibration **per modality configuration** (all 7), not just for C+D+M.

## Primary Metric

**Macro F1** is primary (accuracy is NOT primary — this is imbalanced 11-class dermatology data).
Secondary: balanced accuracy, macro AUROC, macro sensitivity/specificity, accuracy, NLL, Brier,
ECE. Reliability-specific: risk-coverage curve, AURC, error-detection AUROC.

## Missing-Modality Robustness Metric

Delta_S = F1_{CDM} - F1_S for each of the 6 non-full subsets S. Graceful Degradation Rate
GDR = mean(Delta_S) — report this as a project-defined metric, explicitly labeled as non-standard,
never as if it were an established metric. Primary evidence is still the 7 configuration-specific
results table/figure, not GDR alone.

## Statistical Analysis

Lesion-level bootstrap, B=2000, for 95% CI on Macro F1, balanced accuracy, AUROC, ECE, AURC.
Paired bootstrap difference (Delta F1 = F1_proposed - F1_baseline) with 95% CI for model
comparisons — report CI, not p-values alone.

## Ablation (must run in this order, cumulative)

A1 (no reliability weighting, no dropout, no disagreement, no calibration, no abstention) -> A2
(+ reliability weighting) -> A3 (+ modality dropout) -> A4 (+ disagreement) -> A5 (+ calibration
i.e. temperature scaling) -> Full (+ abstention).

## Subgroup Analysis

Sex (male/female), age bands fixed BEFORE looking at results (<40, 40-59, >=60), skin_tone_class
per MILK10k's own 0-5 scale. Report Macro F1, sensitivity, calibration, error rate per subgroup.
Call this "subgroup performance and calibration analysis" — never claim "fair AI".

## Execution Order (do not skip ahead)

Phase 1 Dataset audit -> Phase 2 Freeze split/seeds -> Phase 3 Unimodal baselines (B1) ->
Phase 4 Conventional multimodal baselines (B2, B3, B4) -> Phase 5 Proposed model -> Phase 6
Reliability experiments (7 configs, calibration, missing modality, disagreement, contradiction) ->
Phase 7 Uncertainty / selective prediction -> Phase 8 Robustness (5 seeds, bootstrap CI, subgroup,
ablation, error analysis). Do not write the proposed model (Phase 5) before baselines (Phase 3-4)
are implemented and reproducible.

## Success Criteria

Do not optimize for "+1.2% accuracy over baseline." Success = the proposed model is better
calibrated, degrades more gracefully under missing modalities, and disagreement/uncertainty
measurably predicts error — even if Macro F1 improvement over strong baselines (B4) is modest.
