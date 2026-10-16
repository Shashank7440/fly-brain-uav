# Complete protocol-labelled results

All tables use mean ± sample SD across three training seeds. Validation and test are never pooled.

## original — val

| Model / graph | Macro-F1 | Balanced accuracy (%) | LEFT recall |
|---|---:|---:|---:|
| wireflynet / rewired:4 | 0.7627 ± 0.0248 | 83.169 ± 4.269 | 0.7753 ± 0.0710 |
| wireflynet / rewired:0 | 0.7623 ± 0.0109 | 86.579 ± 0.797 | 0.8134 ± 0.0204 |
| wireflynet / rewired:2 | 0.7619 ± 0.0052 | 85.044 ± 7.497 | 0.7869 ± 0.1233 |
| wireflynet / random:3 | 0.7592 ± 0.0275 | 83.830 ± 6.454 | 0.7537 ± 0.1246 |
| wireflynet / random:0 | 0.7576 ± 0.0231 | 88.092 ± 0.998 | 0.8275 ± 0.0112 |
| wireflynet / rewired:1 | 0.7558 ± 0.0152 | 81.345 ± 2.434 | 0.7222 ± 0.0476 |
| wireflynet / random:1 | 0.7553 ± 0.0141 | 82.654 ± 3.962 | 0.7603 ± 0.0481 |
| wireflynet / random:4 | 0.7522 ± 0.0184 | 85.282 ± 0.710 | 0.7620 ± 0.0166 |
| wireflynet / random:2 | 0.7509 ± 0.0150 | 82.809 ± 2.784 | 0.7595 ± 0.0785 |
| wireflynet / bio | 0.7484 ± 0.0098 | 84.582 ± 2.627 | 0.8027 ± 0.0162 |
| cnn_matched / none | 0.7474 ± 0.0142 | 85.681 ± 2.426 | 0.8308 ± 0.0549 |
| wireflynet / rewired:3 | 0.7460 ± 0.0274 | 84.591 ± 0.843 | 0.7737 ± 0.0293 |

## original — test

| Model / graph | Macro-F1 | Balanced accuracy (%) | LEFT recall |
|---|---:|---:|---:|
| wireflynet / random:1 | 0.7522 ± 0.0206 | 83.500 ± 7.942 | 0.6325 ± 0.1669 |
| cnn_matched / none | 0.7420 ± 0.0238 | 88.243 ± 1.815 | 0.7308 ± 0.0463 |
| wireflynet / random:4 | 0.7360 ± 0.0270 | 87.512 ± 4.427 | 0.7009 ± 0.1316 |
| wireflynet / random:2 | 0.7324 ± 0.0254 | 81.942 ± 6.025 | 0.5940 ± 0.2076 |
| wireflynet / random:3 | 0.7317 ± 0.0101 | 85.177 ± 7.976 | 0.6453 ± 0.2028 |
| wireflynet / bio | 0.7263 ± 0.0130 | 77.809 ± 9.399 | 0.4701 ± 0.2619 |
| wireflynet / rewired:3 | 0.7260 ± 0.0138 | 86.740 ± 6.631 | 0.6838 ± 0.1992 |
| wireflynet / rewired:4 | 0.7215 ± 0.0163 | 79.560 ± 7.400 | 0.4829 ± 0.2150 |
| wireflynet / random:0 | 0.7199 ± 0.0185 | 89.170 ± 4.559 | 0.7735 ± 0.0834 |
| wireflynet / rewired:0 | 0.7133 ± 0.0332 | 85.545 ± 4.651 | 0.6667 ± 0.1888 |
| wireflynet / rewired:2 | 0.7126 ± 0.0604 | 82.199 ± 6.670 | 0.5599 ± 0.1613 |
| wireflynet / rewired:1 | 0.7027 ± 0.0318 | 78.696 ± 12.509 | 0.4616 ± 0.3725 |

## weighted_loss — val

| Model / graph | Macro-F1 | Balanced accuracy (%) | LEFT recall |
|---|---:|---:|---:|
| wireflynet / bio | 0.7630 ± 0.0148 | 86.692 ± 1.006 | 0.8226 ± 0.0101 |
| wireflynet / random:4 | 0.7620 ± 0.0239 | 85.672 ± 3.961 | 0.8018 ± 0.0781 |
| wireflynet / random:1 | 0.7619 ± 0.0163 | 83.209 ± 3.147 | 0.7695 ± 0.0317 |
| wireflynet / rewired:2 | 0.7616 ± 0.0104 | 84.893 ± 2.622 | 0.8184 ± 0.0302 |
| wireflynet / random:3 | 0.7586 ± 0.0116 | 86.443 ± 1.292 | 0.8209 ± 0.0224 |
| wireflynet / rewired:0 | 0.7570 ± 0.0172 | 85.389 ± 2.891 | 0.8076 ± 0.0297 |
| wireflynet / rewired:4 | 0.7533 ± 0.0046 | 84.577 ± 3.383 | 0.8093 ± 0.0683 |
| cnn_matched / none | 0.7531 ± 0.0014 | 85.315 ± 3.765 | 0.8151 ± 0.0315 |
| wireflynet / random:2 | 0.7524 ± 0.0091 | 83.234 ± 0.771 | 0.7761 ± 0.0647 |
| wireflynet / rewired:3 | 0.7515 ± 0.0154 | 85.038 ± 3.547 | 0.7943 ± 0.1081 |
| wireflynet / rewired:1 | 0.7511 ± 0.0057 | 87.395 ± 4.313 | 0.8541 ± 0.0659 |
| wireflynet / random:0 | 0.7468 ± 0.0048 | 86.197 ± 2.854 | 0.8126 ± 0.0760 |
