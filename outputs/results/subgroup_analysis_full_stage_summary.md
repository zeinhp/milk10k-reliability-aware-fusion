# Subgroup Performance and Calibration Analysis (Full stage, C+D+M config)

Averaged across the 5 seeds. Groups with <5 lesions in a given seed's test split are excluded from that seed (too few to compute a meaningful per-group metric); the n below is the average n across seeds where the group had enough lesions.

## sex

| Group | n (avg) | Macro F1 | Sensitivity (macro) | ECE | Error rate |
|---|---|---|---|---|---|
| female | 428 | 0.4500 | 0.4963 | 0.0935 | 0.2679 |
| male | 621 | 0.4700 | 0.4657 | 0.0996 | 0.2771 |

## age_band

| Group | n (avg) | Macro F1 | Sensitivity (macro) | ECE | Error rate |
|---|---|---|---|---|---|
| 40-59 | 283 | 0.4328 | 0.4972 | 0.1208 | 0.3072 |
| <40 | 87 | 0.3232 | 0.4516 | 0.1353 | 0.2694 |
| >=60 | 675 | 0.4306 | 0.4274 | 0.0925 | 0.2603 |
| unknown | 5 | 0.1481 | 0.8333 | 0.1126 | 0.0667 |

## skin_tone_class

| Group | n (avg) | Macro F1 | Sensitivity (macro) | ECE | Error rate |
|---|---|---|---|---|---|
| 1 | 23 | 0.3400 | 0.6833 | 0.2578 | 0.3505 |
| 2 | 99 | 0.4313 | 0.5282 | 0.1754 | 0.3681 |
| 3 | 646 | 0.4490 | 0.4396 | 0.0909 | 0.2601 |
| 4 | 207 | 0.4229 | 0.5167 | 0.0955 | 0.2666 |
| 5 | 73 | 0.3848 | 0.5295 | 0.1103 | 0.2583 |

