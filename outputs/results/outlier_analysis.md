# Outlier Analysis: seed2/A5 and seed3/Full vs. seed0 reference

## seed2_A5 (OUTLIER, F1=0.4427)

- Recomputed val macro F1: 0.4427 (checkpoint logged 0.4427 at epoch 9)
- ECE: 0.0838, mean entropy: 0.6718, mean confidence: 0.7684

| Class | F1 | Support |
|---|---|---|
| AKIEC | 0.427 | 30 |
| BCC | 0.852 | 252 |
| BEN_OTH | 0.286 | 5 |
| BKL | 0.408 | 54 |
| DF | 0.333 | 5 |
| INF | 0.000 | 5 |
| MAL_OTH | 0.000 | 1 |
| MEL | 0.457 | 45 |
| NV | 0.780 | 75 |
| SCCKA | 0.660 | 47 |
| VASC | 0.667 | 5 |

Confusion matrix (rows=true, cols=pred, order matches classes above):
```
       AKIEC    BCC  BEN_O    BKL     DF    INF  MAL_O    MEL     NV  SCCKA   VASC
AKIEC     16      7      0      3      0      0      0      0      0      4      0
  BCC     11    216      1      5      2      0      0      2      3     11      1
BEN_O      1      0      1      2      0      0      0      0      1      0      0
  BKL      8     15      0     20      0      0      0      2      6      3      0
   DF      0      1      0      1      2      0      0      0      0      1      0
  INF      1      1      0      1      1      0      0      0      0      1      0
MAL_O      0      1      0      0      0      0      0      0      0      0      0
  MEL      3      5      0      6      0      0      0     16     15      0      0
   NV      0      4      0      0      2      0      0      5     64      0      0
SCCKA      5      4      0      5      0      0      0      0      0     33      0
 VASC      0      1      0      1      0      0      0      0      0      0      3
```

## seed0_A5 (reference, F1=0.5957)

- Recomputed val macro F1: 0.5957 (checkpoint logged 0.5957 at epoch 18)
- ECE: 0.0753, mean entropy: 0.4656, mean confidence: 0.8319

| Class | F1 | Support |
|---|---|---|
| AKIEC | 0.554 | 30 |
| BCC | 0.888 | 252 |
| BEN_OTH | 0.727 | 5 |
| BKL | 0.395 | 54 |
| DF | 0.571 | 5 |
| INF | 0.500 | 5 |
| MAL_OTH | 0.000 | 1 |
| MEL | 0.592 | 45 |
| NV | 0.808 | 75 |
| SCCKA | 0.717 | 47 |
| VASC | 0.800 | 5 |

Confusion matrix (rows=true, cols=pred, order matches classes above):
```
       AKIEC    BCC  BEN_O    BKL     DF    INF  MAL_O    MEL     NV  SCCKA   VASC
AKIEC     18      7      0      0      0      0      0      1      0      4      0
  BCC      4    231      1      3      0      0      0      2      2      8      1
BEN_O      0      0      4      0      0      0      0      1      0      0      0
  BKL      9     13      0     15      0      0      0      6      3      8      0
   DF      0      3      0      0      2      0      0      0      0      0      0
  INF      0      2      0      0      0      2      0      0      1      0      0
MAL_O      0      0      0      0      0      0      0      1      0      0      0
  MEL      1      3      1      3      0      1      0     29      6      1      0
   NV      0      3      0      1      0      0      0     12     59      0      0
SCCKA      3      5      0      0      0      0      0      1      0     38      0
 VASC      0      1      0      0      0      0      0      0      0      0      4
```

## seed3_Full (OUTLIER, F1=0.4332)

- Recomputed val macro F1: 0.4332 (checkpoint logged 0.4332 at epoch 13)
- ECE: 0.0834, mean entropy: 0.6010, mean confidence: 0.7950

| Class | F1 | Support |
|---|---|---|
| AKIEC | 0.485 | 30 |
| BCC | 0.858 | 252 |
| BEN_OTH | 0.000 | 5 |
| BKL | 0.500 | 54 |
| DF | 0.000 | 5 |
| INF | 0.545 | 5 |
| MAL_OTH | 0.000 | 1 |
| MEL | 0.433 | 45 |
| NV | 0.749 | 75 |
| SCCKA | 0.695 | 47 |
| VASC | 0.500 | 5 |

Confusion matrix (rows=true, cols=pred, order matches classes above):
```
       AKIEC    BCC  BEN_O    BKL     DF    INF  MAL_O    MEL     NV  SCCKA   VASC
AKIEC     16      9      0      5      0      0      0      0      0      0      0
  BCC      7    221      0      7      1      1      0      1      6      8      0
BEN_O      0      2      0      1      0      0      0      0      1      0      1
  BKL     10     10      0     26      0      0      0      0      4      4      0
   DF      0      2      0      0      0      1      0      0      0      2      0
  INF      0      1      0      0      0      3      0      0      0      0      1
MAL_O      0      0      0      0      0      0      0      0      0      1      0
  MEL      0      4      1      7      0      1      0     13     19      0      0
   NV      0      4      0      3      1      0      0      1     64      0      2
SCCKA      3     10      0      1      0      0      0      0      0     33      0
 VASC      0      0      0      0      0      0      0      0      2      0      3
```

## seed0_Full (reference, F1=0.5549)

- Recomputed val macro F1: 0.5549 (checkpoint logged 0.5549 at epoch 18)
- ECE: 0.0564, mean entropy: 0.5017, mean confidence: 0.8175

| Class | F1 | Support |
|---|---|---|
| AKIEC | 0.588 | 30 |
| BCC | 0.901 | 252 |
| BEN_OTH | 0.571 | 5 |
| BKL | 0.602 | 54 |
| DF | 0.333 | 5 |
| INF | 0.400 | 5 |
| MAL_OTH | 0.000 | 1 |
| MEL | 0.602 | 45 |
| NV | 0.780 | 75 |
| SCCKA | 0.659 | 47 |
| VASC | 0.667 | 5 |

Confusion matrix (rows=true, cols=pred, order matches classes above):
```
       AKIEC    BCC  BEN_O    BKL     DF    INF  MAL_O    MEL     NV  SCCKA   VASC
AKIEC     20      7      0      0      0      0      0      2      0      1      0
  BCC      1    241      0      2      0      0      0      2      0      5      1
BEN_O      0      1      2      0      0      0      0      1      1      0      0
  BKL      8     10      0     25      0      1      0      4      2      4      0
   DF      2      2      0      0      1      0      0      0      0      0      0
  INF      1      1      0      0      0      2      0      0      1      0      0
MAL_O      0      0      0      0      0      0      0      1      0      0      0
  MEL      2      4      0      0      0      1      0     31      7      0      0
   NV      0      4      0      0      0      0      0     16     55      0      0
SCCKA      4     12      0      2      0      1      0      0      0     28      0
 VASC      0      1      0      0      0      0      0      1      0      0      3
```
