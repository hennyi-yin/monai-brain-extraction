# Reproduce the external T1w experiment

See the [frozen protocol](external_protocol.md), [primary report](../results/external/synthstrip_v15_t1/summary.md), and [geometry sensitivity](external_geometry_sensitivity.md). The external experiment is separate from the NFBS study. It uses the original NFBS checkpoint and leaves all prior results untouched.

1. Restore the earlier NFBS study artifacts using [v1.1.1](reproduce.md) and the FreeSurfer masks using [v1.2.0](freesurfer.md). Use a fresh workspace for independent external inference, keeping the published external JSON/CSV files as a reference. The runner refuses to overwrite a completed scoring record. In that workspace, move the published external predictions, smoke reports and evaluation outputs out of `results/external/synthstrip_v15_t1/`, retaining `dataset.json` and `frozen_before.json`.
2. Install the Python versions in `requirements-lock.txt` and the exact FSL executables/template recorded in the earlier robustfov manifest. Set `FSLDIR` and put its `bin` on `PATH`. Obtain the official FreeSurfer 7.4.1 package and set `FREESURFER_HOME`; the external run calls the original `python/scripts/mri_synthstrip` and `models/synthstrip.1.pt` with a CUDA-capable PyTorch installation and `surfa==0.6.3`, `xxhash==3.5.0`. It does not call `recon-all`. The pinned script and weights are checked by hash. The original FreeSurfer script may require a newer PyTorch build than its bundled interpreter to use a recent GPU.
3. Obtain the [official SynthStrip v1.5 3D archive](https://surfer.nmr.mgh.harvard.edu/docs/synthstrip/data/synthstrip_data_v1.5.tar). The preparer verifies the publisher's SHA-256, extracts the 89 locked pairs into ignored `data/`, checks native grids and binary labels, and compares exact image hashes to the NFBS inputs.

```bash
python -m scripts.prepare_external_data /path/to/synthstrip_data_v1.5.tar
for method in unet bet_r bet_crop synthstrip; do
  python -m src.run_external "$method" --smoke
done
python -m src.run_external bet_r --jobs 4
python -m src.run_external bet_crop --jobs 4
python -m src.run_external unet
python -m src.run_external synthstrip
python -m src.evaluate_external --jobs 3
python -m scripts.audit_external
python -m scripts.external_geometry_sensitivity
```

The audit does not recompute Dice or HD95. Prediction masks and intermediate ROI files stay under ignored `results/external_work/`. The native volume and mask hashes, exact commands, failures, all subject-level scores and the confidence interval calculation are recorded in the repository. `python -m src.evaluate_external --report-only` redraws the primary report and figure from the locked score file without rescoring.

For HD95 efficiency only, scoring removes the common zero background outside the union of prediction and reference foreground bounds before calling `src.metrics.score`. This leaves both surfaces and all distances unchanged. It does not crop input scans or predictions.

References: [Hoopes et al., SynthStrip](https://doi.org/10.1016/j.neuroimage.2022.119474), [IXI / Imperial College London](https://brain-development.org/ixi-dataset/), and [Juttukonda et al., ASL source study](https://doi.org/10.1016/j.neuroimage.2021.117807). Refer to the [publisher's archive license](https://surfer.nmr.mgh.harvard.edu/docs/synthstrip/data/README) for the image and annotation licenses. The repository distributes metrics and analysis artifacts, not the scans or reference masks.
