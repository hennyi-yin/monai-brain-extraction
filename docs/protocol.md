# Frozen experimental protocol

The primary experiment implements requirements R1–R10 from the supplied brief. The test set is untouched for model choice. Data validation checks all file pairs but does not compute model performance. Preprocessing overlays and smoke checks use only training subjects. Validation chooses one checkpoint; after training ends the final test protocol is frozen.

| Requirement | Implementation / evidence |
|---|---|
| R1 | `python -m src.make_split`: all 125 pairs, shape, affine, 1 mm spacing, finite voxels and nonempty foreground after threshold 0.5; `results/data_audit.json` records fractional label voxels |
| R2 | Committed seed-42 `splits/split.json`, exact 85/15/25 split; duplicate IDs rejected |
| R3 | `src/data.py`, no interpolation during RAS orientation; `results/figures/preprocessing_qc.png` uses three training subjects |
| R4 | `src/model.py`, actual 96³ forward/backward check in `results/smoke.json` |
| R5 | `configs/config.yaml`, `logs/training.csv`, checkpoint saved by validation Dice; two crops per subject for 12 GB |
| R6 | `src/infer.py`, CPU output accumulation, connected component/hole filling and inverse orientation; native-grid roundtrip tests |
| R7 | `src/run_bet.py`, two fixed settings and 50 masks; `results/bet_manifest.json` records commands |
| R8 | `src/metrics.py`, physical-space symmetric HD95, 75 rows in `results/per_subject.csv` |
| R9 | ID-aligned paired tests, means, sample SD, medians, both baselines; `results/summary.md` |
| R10 | Pinned dependencies/config/split, full RNG and resume state, `python evaluate.py`, hashed frozen analysis |

The patch sampler's 1:1 weights are an equal probability of a foreground or background center; random sampling does not force exactly equal counts in each minibatch. DataLoader batch size 2 denotes subjects, and two crops per subject result in four 96³ patches.

The best checkpoint uses full validation volumes and inference postprocessing; no validation subject is used in optimization. Epoch loss is the weighted mean over all training patches. Early-stopping patience is measured in epochs, not in 30 validation events.

Two baseline settings are fixed in advance. Selecting the better BET variant on test data is retained to match the brief and explicitly disclosed. The resulting p-value is descriptive, unadjusted for that selection. Never modify the test split, retrain based on test overlays, or claim synthetic smoke scores as experimental results.

After scoring, the large BET discrepancy was investigated using the original shape, affine, orientation, qform/sform and world-coordinate centroids. The two zero-overlap -R masks are nonempty masks centered in the neck, rather than grid misalignment. All 25 cases remain in the analysis. The selected baseline's median Dice is 0.9301, while seven failures below 0.8 reduce its mean to 0.7734; the median paired gain is 5.7 points. The 21.3-point mean gain must retain the fixed-command, uncropped-input qualification.

The [NFBS article](https://link.springer.com/article/10.1186/s13742-016-0150-5) used FSL 5.0.7 `bet -B` for bias-field correction and neck cleanup when reporting 0.893 ± 0.027. This project implements the supplied brief's default and -R settings; it does not reproduce that historical baseline. A stronger baseline comparison would require a separate protocol chosen using training/validation data.

Determinism is enabled with seed 42. Training checkpoints record Python/NumPy/PyTorch/CUDA/DataLoader random states; worker transform seeds derive from the restored loader generator. Exact bitwise replay can still depend on hardware and CUDA library behavior. Checkpoints contain optimizer metadata and must be loaded only from a trusted local run or verified release hash.

## Addendum (2026-10-08): post-hoc supplementary baseline

The uncropped primary BET -R masks have Dice <0.8 in seven of 25 subjects, including two zero-overlap masks localized in the neck. This motivates exactly one stronger supplementary baseline: neck cropping before robust BET. It was chosen after viewing test scores and is **post hoc**, not a replacement for the primary analysis. The reorientation/cropping order follows [FSL fsl_anat](https://fsl.fmrib.ox.ac.uk/fsl/docs/structural/fsl_anat.html). This addendum is committed alone before any supplementary BET test-subject run.

The single pre-declared pipeline is, for every subject:

```bash
fslreorient2std <native_T1w> <reoriented_T1w>
robustfov -i <reoriented_T1w> -r <roi_T1w> -b 170 -m <roi_to_reoriented.mat>
bet <roi_T1w> <output> -f 0.5 -R -m -n
```

`-b 170` is robustfov's default brain extent in mm; `-m` records geometry only. No tuning, -B, per-subject settings, exclusions, extra postprocessing, or changes to the trained model are allowed. Only the first three training subjects in the frozen split are used for implementation smoke tests; no smoke scores guide parameter choices. After those checks pass, all 25 test subjects are processed and scored once. Subject failures produce native-grid empty masks, Dice 0 and HD95 infinity, and remain in all reports.

Map each ROI and BET mask to the native T1w grid with `nibabel.processing.resample_from_to(..., order=0)`. Verify exact ROI T1w identity inside the mapped ROI (maximum absolute difference 0), preservation of the ROI footprint voxel count, and preservation of the foreground mask voxel count. Save uint8 binary masks with the original affine. If world-coordinate mapping fails, use FSL's recorded transformations (`fslreorient2std -m`, `robustfov -m`, `convert_xfm`, `flirt -applyxfm -interp nearestneighbour -ref <native_T1w>`) and repeat the same checks; if geometry still fails, score the subject as empty. Record the fraction of native ground-truth brain voxels retained by each crop as a diagnostic, without using it to change the crop or exclude subjects.

The supplementary manifest records executable locations/hashes, the attempted `$FSLDIR/etc/fslversion` read (including an explicit unavailable status if absent from this modular FSL installation), exact package versions/builds, every external command/return code, input/intermediate/output SHA-256, ROI retention, map-back checks and failures. A synthetic axis-permutation/flip/crop roundtrip test must pass before test processing.

Evaluation uses the existing foreground Dice and symmetric physical-space HD95 definitions in `src/metrics.py`, binary truth threshold ≥0.5 and the native grid's voxel spacing. Align all subjects by ID against the frozen U-Net and plain BET -R rows in `results/per_subject.csv`. Report means with sample SD, medians, minima, Dice <0.8 counts, zero-overlap counts, HD95 medians, two-sided paired Wilcoxon p-values vs U-Net, mean/median paired U-Net gain in points and wins/25. Also report the paired cropping gain vs plain BET -R, failure-case recovery, minimum ROI retention and every failed subject. P-values are descriptive and unadjusted in this post-hoc analysis. Both improved and unsuccessful outcomes are retained without changing parameters.

Outputs are confined to `results/supplementary/robustfov_bet/` and a new ignored `results/bet_robustfov/` mask/work directory. The supplementary README section stays outside the primary results block. The split, configuration, checkpoints, primary U-Net/BET outputs, primary CSV/summaries/evaluation hash evidence and README results block are SHA-256 recorded before work and checked unchanged at completion. Exactly three local commits are made: this addendum; code and tests; results and README. Nothing is pushed.
