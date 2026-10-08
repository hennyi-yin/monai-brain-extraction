A MONAI 3D U-Net achieves Dice **0.99** on **25** held-out NFBS T1w scans, **21.3 points above** fixed FSL BET (-R, uncropped scans; paired Wilcoxon p = 5.96046e-08).

| Method | Dice mean ± SD | Dice median | HD95 mm mean ± SD | p vs U-Net |
|---|---:|---:|---:|---:|
| MONAI 3D U-Net | 0.9863 ± 0.0018 | 0.9868 | 1.43 ± 0.19 | — |
| FSL BET (default, -f 0.5) | 0.7581 ± 0.1199 | 0.7735 | 46.54 ± 27.25 | 5.96046e-08 |
| FSL BET (-R, -f 0.5) (selected) | 0.7734 ± 0.3014 | 0.9301 | 32.92 ± 41.66 | 5.96046e-08 |

The selected BET setting has 7/25 cases with Dice <0.8 and 2 zero-overlap masks. Its median Dice is 0.9301; the median paired U-Net gain is 5.7 points. The larger mean gain is influenced by these BET failures. This comparison uses the two fixed commands on uncropped inputs, and is not a comparison against a neck-cropped or otherwise optimized FSL pipeline. The [NFBS paper](https://link.springer.com/article/10.1186/s13742-016-0150-5) reported BET Dice 0.893 ± 0.027 using FSL 5.0.7 with `bet -B` (bias-field and neck cleanup), not the default/-R settings evaluated here. That published reference is a different protocol.

SD is the sample standard deviation (ddof=1). Two-sided paired Wilcoxon tests align by subject ID; zero differences are excluded, and an all-zero difference returns p=1. HD95 is the maximum of the two directed 95th-percentile surface distances, in mm.

Both BET settings were fixed before testing. The BET variant with higher test mean Dice is selected as required by the project brief. This is test-based baseline selection; the selected comparison's p-value is descriptive and unadjusted for selection. Both variants are reported for transparency. The U-Net checkpoint and all model decisions use only training/validation data.

Resume wording (rounded mean Dice; delta computed from unrounded means):

> Achieved a Dice of 0.99 on NFBS brain extraction (25 held-out T1w scans), 21.3 points above FSL BET (best of default/-R on uncropped scans), by training a MONAI 3D U-Net with intensity normalization and spatial augmentation.
