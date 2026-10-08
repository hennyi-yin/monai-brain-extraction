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

Determinism is enabled with seed 42. Training checkpoints record Python/NumPy/PyTorch/CUDA/DataLoader random states; worker transform seeds derive from the restored loader generator. Exact bitwise replay can still depend on hardware and CUDA library behavior. Checkpoints contain optimizer metadata and must be loaded only from a trusted local run or verified release hash.
