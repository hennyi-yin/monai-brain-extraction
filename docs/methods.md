# Methods and evidence

## Dataset and split

NFBS contains 125 defaced T1w scans with 1 mm voxel spacing and corresponding brain masks. The labels were produced with BEaST, visually reviewed, and corrected where required: 85 edited, 40 accepted without edits. All labels are supplied by the dataset; not every case was manually edited.

Subject IDs are sorted, shuffled with Python `random.Random(42)`, and split into **85 training / 15 validation / 25 test** subjects. Only source T1w images and masks are used. Fractional source labels are binarized at **≥0.5** consistently for training, validation and evaluation.

Evidence: [frozen split](../splits/split.json), [data audit](../results/data_audit.json), [experimental protocol](protocol.md).

## Model and training

| Component | Fixed setting |
|---|---|
| Network | MONAI 3D `UNet`, one input channel, two output classes |
| Architecture | Channels 16/32/64/128/256; strides 2/2/2/2; two residual units; InstanceNorm |
| Preprocessing | RAS orientation without interpolation; nonzero-voxel z-score normalization |
| Sampling | 96³ patches; foreground/background center weights 1:1 |
| Augmentation | Left/right flips; small affine rotation/scaling; nearest-neighbor label interpolation |
| Batch | Two subjects × two crops = four patches per optimizer step |
| Optimization | AdamW, learning rate 1e-4, weight decay 1e-5; DiceCE loss; CUDA mixed precision |
| Model selection | Full-volume validation every five epochs; best validation Dice |
| Stopping | At most 200 epochs; patience of 30 epochs without improvement |

Two crops per subject are the project brief's 12 GB memory fallback, fixed before training. Equal sampling weights specify probabilities, not an enforced 50/50 count in every batch. The completed run reached epoch 200 and selected epoch 190. Validation uses inference postprocessing. Checkpoints include optimizer/scaler and random states for resumption.

Evidence: [configuration](../configs/config.yaml), [training history](../logs/training.csv), [completion record](../logs/completed.json), [environment](../logs/environment.json), [training curves](../results/figures/training_curves.png).

## Inference and evaluation

Inference uses 96³ sliding windows with 0.5 overlap and constant blending. Window execution runs on GPU; full-volume output accumulates on CPU. Postprocessing keeps the largest foreground connected component with 6-connectivity and fills holes. Inverse axis permutations/flips restore the original shape and affine. Masks are saved as uint8 binary NIfTI; no resampling or image-guided trimming is used for the U-Net.

Foreground Dice and physical-space HD95 are computed in the native grid. HD95 is the maximum of the two directed 95th-percentile surface distances, using each scan's actual voxel spacing. Empty predictions score Dice 0 and HD95 infinity. Tables use sample SD (`ddof=1`); paired two-sided Wilcoxon tests align subjects by ID. All-zero paired differences yield p=1.

The primary analysis evaluates two fixed BET commands, `-f 0.5 -m -n` and `-f 0.5 -R -m -n`, on uncropped images. The higher test mean selects the reported BET variant as required by the original brief. This selection is disclosed; p-values are descriptive and unadjusted. U-Net checkpoint selection uses validation data exclusively.

The large primary mean advantage is affected by seven BET -R failures, including two nonempty masks localized in the neck with zero overlap. Native shape, affine, orientation, qform/sform and world-coordinate centroids were checked. No subjects were excluded. The [published NFBS BET reference](https://doi.org/10.1186/s13742-016-0150-5) used FSL 5.0.7 `bet -B` and is a different protocol.

Evidence: [metric definitions](../src/metrics.py), [primary scores](../results/per_subject.csv), [primary statistics](../results/summary.md), [BET failure QC](../results/baseline_qc.json), [original failure overlays](../results/figures/bet_failure_qc.png).

## Post-hoc supplementary comparison

Exactly one pipeline was pre-declared in a separate commit after seeing the primary test scores: `fslreorient2std → robustfov -b 170 → bet -f 0.5 -R -m -n`. It was smoke-tested on three training subjects and scored once on all 25 test subjects. No parameters, subjects, model decisions or primary artifacts were changed.

Native map-back checks require exact ROI image identity (maximum absolute difference 0), unchanged mask voxel count and unchanged ROI footprint voxel count. World-coordinate mapping uses nearest-neighbor interpolation; a pre-declared FSL matrix fallback handles header-boundary rounding. All 25 scans passed, using 18 direct mappings and seven matrix fallbacks. Minimum ground-truth ROI retention was 99.9987%; no subject failed.

Cropping recovered all seven previous Dice <0.8 cases, but reduced Dice in the other 18 cases. The mean paired improvement over plain BET -R was 14.43 points, while the median change was **−0.99 points**. It therefore improves severe neck failures without demonstrating uniform improvement. The U-Net's mean/median paired gain over this supplementary baseline was 6.86/6.78 points, with 25/25 wins. Post-hoc p-values are descriptive and unadjusted.

The modular FSL installation has no `$FSLDIR/etc/fslversion`; the manifest explicitly records its absence instead of inventing a suite version. Exact package builds, executable hashes, commands, geometry checks and input/output hashes are provided.

Evidence: [pre-registration addendum](protocol.md#addendum-2026-10-08-post-hoc-supplementary-baseline), [supplementary summary](../results/supplementary/robustfov_bet/summary.md), [manifest](../results/supplementary/robustfov_bet/manifest.json), [geometry notes](../results/supplementary/robustfov_bet/geometry_notes.md), [frozen-artifact verification](../results/supplementary/robustfov_bet/verification.json).

## Scope

This is an internal held-out evaluation on one dataset. It does not establish clinical performance or cross-dataset robustness. No external laboratory data, augmentation ablation or artificial degradation experiment is part of the reported results.
