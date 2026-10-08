# Post-hoc supplementary baseline: robustfov + BET -R

On the same 25 held-out scans, the post-hoc robustfov + BET -R baseline achieves Dice 0.9178 ± 0.0154; the frozen U-Net's mean paired gain is 6.86 points (25/25 wins).

| Method | Dice mean ± SD | Median | Min | n < 0.8 | HD95 median (mm) | p vs U-Net |
|---|---:|---:|---:|---:|---:|---:|
| U-Net (frozen) | 0.9863 ± 0.0018 | 0.9868 | 0.9820 | 0 | 1.41 | — |
| BET -R (frozen) | 0.7734 ± 0.3014 | 0.9301 | 0.0000 | 7 | 9.51 | 5.96046e-08 |
| robustfov + BET -R (post hoc) | 0.9178 ± 0.0154 | 0.9174 | 0.8830 | 0 | 10.72 | 5.96046e-08 |

U-Net mean / median paired gain: **6.8588 / 6.7835 points**; wins **25/25**, ties 0.

Cropping vs plain BET -R: mean / median paired gain **14.4331 / -0.9868 points**, wins 7/25, paired two-sided Wilcoxon p=0.832509. Recovered to Dice ≥0.8: 7/7 previous failure cases.

Minimum ground-truth brain retention in the ROI: **99.99873237%** (25/25 available). Failed subjects: none. Failed steps remain in the denominator with empty masks, Dice 0 and HD95 infinity.

Zero-overlap counts (U-Net / plain BET -R / robustfov+BET): 0 / 2 / 0.

This pipeline was chosen after viewing primary test scores. The supplementary p-values are descriptive and unadjusted; this does not replace the frozen primary table or resume claim. There was no parameter tuning, -B, subject exclusion or model change. The first three training subjects were used only to repair installation/geometry and smoke-test implementation.

Dice and symmetric HD95 use src/metrics.py; truth is thresholded at ≥0.5 in its native grid. SD uses ddof=1, paired comparisons align by subject ID, and wins require a strictly positive difference. The native grid spacing is used for HD95 in mm. ROI retention is a diagnostic only.

See [protocol addendum](../../../docs/protocol.md#addendum-2026-10-08-post-hoc-supplementary-baseline), [manifest](manifest.json), [frozen hashes before](frozen_before.json), [frozen hashes after](frozen_after.json).

![Paired Dice](figures/paired_dice.png)

![Three previous failures](figures/qc_previous_failures.png)
