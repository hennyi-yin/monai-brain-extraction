<p align="center">
  <img src="docs/assets/banner.svg" alt="MONAI Brain Extraction — reproducible 3D MRI segmentation" width="100%">
</p>

<p align="center">
  <a href="https://github.com/hennyi-yin/monai-brain-extraction/actions/workflows/tests.yml"><img src="https://github.com/hennyi-yin/monai-brain-extraction/actions/workflows/tests.yml/badge.svg" alt="Protocol tests"></a>
  <a href="https://github.com/hennyi-yin/monai-brain-extraction/releases/tag/v1.1.1"><img src="https://img.shields.io/badge/release-v1.1.1-1f9e9c" alt="Release v1.1.1"></a>
  <a href="splits/split.json"><img src="https://img.shields.io/badge/NFBS-125%20subjects-4263a8" alt="NFBS: 125 subjects"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-64748b" alt="MIT license"></a>
</p>

<p align="center">
  <a href="#results">Results</a> ·
  <a href="#external-validation-ixi-and-asl-adult-t1w">External validation</a> ·
  <a href="docs/reproduce.md">Reproduce</a> ·
  <a href="docs/methods.md">Methods</a> ·
  <a href="docs/protocol.md">Protocol</a> ·
  <a href="https://github.com/hennyi-yin/monai-brain-extraction/releases/tag/v1.1.1">Weights &amp; masks</a>
</p>

A reproducible brain-extraction benchmark comparing a **MONAI 3D U-Net** with fixed **FSL BET** and **FreeSurfer** baselines on native T1w MRI. The project includes completed training, subject-level scores, visual quality checks, verified weights and all prediction masks.

- **Fixed held-out evaluation:** 85 training / 15 validation / 25 test subjects, seed 42; checkpoint selected on validation data.
- **Traceable outputs:** native-grid masks, exact commands, dependency locks and SHA-256 manifests.
- **Transparent comparison:** frozen primary analysis plus two explicitly labeled post-hoc baselines: neck-cropped BET and FreeSurfer.

## Results

The U-Net achieves **0.9863 ± 0.0018 Dice** on the 25 held-out NFBS scans. This is the held-out NFBS result; see the separate external T1w validation below. Neither analysis establishes clinical performance.

![U-Net predictions on the best, median and worst held-out cases](results/figures/overlays.png)

<details>
<summary><strong>Frozen primary analysis — full results and BET failure caveats</strong></summary>

<!-- RESULTS:START -->
A MONAI 3D U-Net achieves Dice **0.99** on **25** held-out NFBS T1w scans, **21.3 points above** fixed FSL BET (-R, uncropped scans; paired Wilcoxon p = 5.96046e-08).

| Method | Dice mean ± SD | Dice median | HD95 mm mean ± SD | p vs U-Net |
|---|---:|---:|---:|---:|
| MONAI 3D U-Net | 0.9863 ± 0.0018 | 0.9868 | 1.43 ± 0.19 | — |
| FSL BET (default, -f 0.5) | 0.7581 ± 0.1199 | 0.7735 | 46.54 ± 27.25 | 5.96046e-08 |
| FSL BET (-R, -f 0.5) (selected) | 0.7734 ± 0.3014 | 0.9301 | 32.92 ± 41.66 | 5.96046e-08 |

The selected BET setting has 7/25 cases with Dice <0.8 and 2 zero-overlap masks. Its median Dice is 0.9301; the median paired U-Net gain is 5.7 points. The larger mean gain is influenced by these BET failures. This comparison uses the two fixed commands on uncropped inputs, and is not a comparison against a neck-cropped or otherwise optimized FSL pipeline. The [NFBS paper](https://link.springer.com/article/10.1186/s13742-016-0150-5) reported BET Dice 0.893 ± 0.027 using FSL 5.0.7 with `bet -B` (bias-field and neck cleanup), not the default/-R settings evaluated here. That published reference is a different protocol.
<!-- RESULTS:END -->

![Primary paired Dice scores](results/figures/paired_dice.png)

The original scores, model outputs and generated result block remain byte-identical. [Primary summary](results/summary.md) · [Per-subject scores](results/per_subject.csv) · [Original BET failure overlays](results/figures/bet_failure_qc.png).

</details>

## Supplementary baseline (post hoc): robustfov + BET -R

On the same 25 scans, the post-hoc neck-cropped baseline achieves **0.9178 ± 0.0154 Dice**, with a **6.86-point mean paired U-Net gain** and **25/25 U-Net wins**.

| Method | Dice mean ± SD | Median | Dice <0.8 | HD95 median (mm) |
|---|---:|---:|---:|---:|
| **MONAI 3D U-Net** | **0.9863 ± 0.0018** | **0.9868** | **0/25** | **1.41** |
| BET -R · primary, uncropped | 0.7734 ± 0.3014 | 0.9301 | 7/25 | 9.51 |
| robustfov + BET -R · post hoc | 0.9178 ± 0.0154 | 0.9174 | 0/25 | 10.72 |

Cropping recovers all seven previous failures, but lowers Dice in the other 18 cases: its mean gain over plain BET -R is +14.43 points and its median change is −0.99 points. The large primary mean gap is affected by severe neck failures. The supplementary pipeline was chosen **after viewing test scores**, pre-declared in a separate commit, and scored once without tuning or exclusions. It does not replace the primary analysis.

![Paired U-Net, original BET and post-hoc cropped BET Dice scores](results/supplementary/robustfov_bet/figures/paired_dice.png)

<details>
<summary><strong>Visual check: three previous neck-localized BET failures</strong></summary>

![Ground truth and robustfov plus BET sagittal overlays](results/supplementary/robustfov_bet/figures/qc_previous_failures.png)

All 25 map-back checks passed with exact ROI image identity and preserved mask voxel counts. Minimum ground-truth brain retention in the crop was **99.9987%**. No supplementary subject failed.

</details>

[Protocol addendum](docs/protocol.md#addendum-2026-10-08-post-hoc-supplementary-baseline) · [Full supplementary statistics](results/supplementary/robustfov_bet/summary.md) · [Commands & geometry manifest](results/supplementary/robustfov_bet/manifest.json)

## Supplementary baseline (post hoc): FreeSurfer autorecon1

On the same 25 scans, **FreeSurfer 7.4.1** achieves **0.8591 ± 0.1472 Dice**. The mean paired U-Net minus FreeSurfer difference is **+12.73 Dice points** (25/25 U-Net wins). Processing failures: **0/25**, retained in the comparison.

| Method | Dice mean ± SD | Dice median | Dice <0.8 | HD95 median (mm) |
|---|---:|---:|---:|---:|
| U-Net (frozen) | 0.9863 ± 0.0018 | 0.9868 | 0/25 | 1.41 |
| BET -R (frozen) | 0.7734 ± 0.3014 | 0.9301 | 7/25 | 9.51 |
| robustfov + BET -R (post hoc) | 0.9178 ± 0.0154 | 0.9174 | 0/25 | 10.72 |
| FreeSurfer 7.4.1 autorecon1 (post hoc) | 0.8591 ± 0.1472 | 0.8994 | 2/25 | 8.25 |

This independent post-hoc comparison uses the default watershed step in `recon-all -autorecon1` on uncropped T1w scans. Masks are mapped to the original T1w grid. The workflow was fixed before running the FreeSurfer test set; the original analysis and earlier supplement are unchanged.

Two FreeSurfer masks fell below Dice 0.8 despite successful processing. [Native overlays](results/supplementary/freesurfer_autorecon1/figures/low_dice_qc.png) show extensive extra foreground outside the brain in both cases.

[Protocol](docs/protocol.md#addendum-2026-11-10-post-hoc-freesurfer-baseline) · [Full results and native QC](results/supplementary/freesurfer_autorecon1/summary.md) · [Run and audit instructions](docs/freesurfer.md) · [Download the 25 FreeSurfer masks](https://github.com/hennyi-yin/monai-brain-extraction/releases/tag/v1.2.0)

## External validation: IXI and ASL adult T1w

The frozen NFBS checkpoint was applied without retraining to **89 external scans** (48 IXI and 41 ASL-cohort T1w) from the official SynthStrip v1.5 test split.

| Cohort | NFBS U-Net Dice | BET -R Dice | SynthStrip Dice | Cropped BET Dice (strict / corrected) |
|---|---:|---:|---:|---:|
| IXI (48) | 0.9560 | 0.9652 | 0.9701 | 0.4791 / 0.9558 |
| ASL (41) | 0.8950 | 0.9488 | 0.9731 | 0.2434 / 0.9036 |
| Pooled (89) | 0.9279 | 0.9577 | 0.9715 | 0.3705 / 0.9318 |

**The NFBS advantage did not transfer to these external comparisons.** Mean paired U-Net Dice was **2.98 points lower than BET -R** and **4.35 points lower than SynthStrip** (95% bootstrap CI for U-Net minus SynthStrip: **[-4.84, -3.91]**; SynthStrip was higher in all 89 pairs).

The pre-declared strict cropped BET comparison counted **54/89** map-back failures as empty masks, which drove its apparent +55.74-point U-Net difference. A separate post-hoc sensitivity recovered **54/54** from the original BET masks without rerunning extraction. The corrected paired difference was **-0.38 points [95% CI: -0.95, +0.22]**.

The source references were manually refined from a consensus of classical methods. These results describe adult T1w transfer in these two cohorts and do not establish clinical performance.

[Full external report](results/external/synthstrip_v15_t1/summary.md) · [Pre-declared protocol](docs/external_protocol.md) · [Geometry sensitivity](results/external/synthstrip_v15_t1/geometry_sensitivity/summary.md) · [Reproduce](docs/external_reproduce.md)

## Reproduce

The [v1.1.1 release](https://github.com/hennyi-yin/monai-brain-extraction/releases/tag/v1.1.1) provides the best checkpoint, inference weights, **100 native-grid prediction masks** and checksums. Existing masks can be audited on CPU; training and new U-Net inference use CUDA.

```bash
git clone https://github.com/hennyi-yin/monai-brain-extraction.git
cd monai-brain-extraction
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements-lock.txt
```

Follow the [reproduction guide](docs/reproduce.md) to download and verify the assets, restore masks, run the audits, or train independently. The primary CSV was reproduced byte for byte in a fresh locked environment. The completed supplementary evaluation is guarded against repeated scoring.

## Project map

| Start here | Contents |
|---|---|
| [Methods](docs/methods.md) | Dataset, architecture, preprocessing, inference, metrics and limitations |
| [Protocol](docs/protocol.md) | Frozen primary decisions and the dated post-hoc addendum |
| [Source](src/) / [tests](tests/) | MONAI pipeline, FSL runners, evaluation and native-grid roundtrip tests |
| [Primary results](results/summary.md) | 75 metric rows across three fixed methods; training and QC figures |
| [Supplementary results](results/supplementary/robustfov_bet/summary.md) | 25 extra metric rows, comparisons, commands and map-back evidence |
| [FreeSurfer comparison protocol](docs/freesurfer.md) | Fixed `autorecon1` setup, training checks and independent evaluation commands |
| [Training record](logs/) | 200-epoch history, selected checkpoint metadata and environment |

## Data & references

NFBS provides 125 defaced T1w scans and BEaST-derived brain masks, visually reviewed and corrected where needed (85 edited; 40 accepted without edits). The download script retains the source data. Code is [MIT-licensed](LICENSE); NFBS data and FSL retain their own terms.

- **NFBS:** Puccio et al. (2016), [manually corrected skull-stripped T1w MRI repository](https://doi.org/10.1186/s13742-016-0150-5). [Dataset](https://preprocessed-connectomes-project.org/NFB_skullstripped/).
- **BET:** Smith (2002), [Fast robust automated brain extraction](https://doi.org/10.1002/hbm.10062). [FSL documentation](https://fsl.fmrib.ox.ac.uk/fsl/docs/structural/bet.html).
- **MONAI:** MONAI Consortium, [An open-source framework for deep learning in healthcare](https://arxiv.org/abs/2211.02701).
