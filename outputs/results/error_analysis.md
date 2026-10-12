# Error Analysis: Pooled Confusion Matrix (C+D+M, Full stage, 5 seeds combined)

n=5245 predictions (5 seeds' test splits combined -- NOT the same lesions repeated, since each seed has its own train/val/test split; pooling is valid for an aggregate error pattern view, not for per-lesion analysis).

## Per-class F1 and support

| Class | Support | F1 |
|---|---|---|
| BCC | 2525 | 0.881 |
| NV | 745 | 0.732 |
| BKL | 545 | 0.465 |
| SCCKA | 475 | 0.630 |
| MEL | 450 | 0.575 |
| AKIEC | 305 | 0.425 |
| DF | 55 | 0.465 |
| INF | 50 | 0.286 |
| VASC | 45 | 0.556 |
| BEN_OTH | 40 | 0.235 |
| MAL_OTH | 10 | 0.000 |

## Top 15 confusions (true -> predicted)

| True | Predicted | Count | % of true class |
|---|---|---|---|
| BKL | BCC | 130 | 23.9% |
| MEL | NV | 113 | 25.1% |
| NV | MEL | 107 | 14.4% |
| SCCKA | BCC | 91 | 19.2% |
| AKIEC | BCC | 70 | 23.0% |
| BCC | BKL | 59 | 2.3% |
| BCC | AKIEC | 58 | 2.3% |
| BCC | SCCKA | 58 | 2.3% |
| SCCKA | AKIEC | 57 | 12.0% |
| AKIEC | BKL | 56 | 18.4% |
| BKL | AKIEC | 55 | 10.1% |
| NV | BKL | 53 | 7.1% |
| BKL | SCCKA | 52 | 9.5% |
| MEL | BKL | 43 | 9.6% |
| AKIEC | SCCKA | 38 | 12.5% |

## Observations

- Lowest F1 classes: MAL_OTH, BEN_OTH, INF (check support column -- likely small-sample rare classes, consistent with the earlier outlier analysis finding that rare-class instability drives macro-F1 variance, not systematic model failure).
- Highest F1 classes: BCC, NV, SCCKA.
- Single largest confusion: BKL predicted as BCC (130 cases, 23.9% of all true-BKL cases).
- **Clinically notable**: MEL (melanoma) <-> NV (nevus) is the #2 and #3 largest confusion pair
  (113 MEL->NV, 107 NV->MEL). This is the single most safety-relevant error mode in the whole
  matrix -- missing a melanoma as a benign nevus is the highest-stakes mistake a skin-lesion
  classifier can make. Worth a dedicated callout in the paper's limitations/error-analysis section,
  not just a line in the confusion table.
- Most other confusion mass is within the keratinocytic-lesion cluster (BKL/AKIEC/SCCKA/BCC all
  confusing into each other) -- clinically more forgivable (all are common, usually
  non-life-threatening, and often managed similarly), but still worth noting as the dominant error
  pattern by volume.
