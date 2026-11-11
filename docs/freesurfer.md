# FreeSurfer 7.4.1 comparison

This is a separate post-hoc baseline for the published NFBS study. It evaluates the default watershed skull strip in `recon-all -autorecon1`, using the same native T1w inputs and fixed test split. It is not the full FreeSurfer reconstruction, the FreeSurfer 8 SynthStrip pipeline, or a new primary analysis. See the [pre-declared protocol](protocol.md#addendum-2026-11-10-post-hoc-freesurfer-baseline).

## Prepare the environment

1. Restore the frozen inputs and masks with the [v1.1.1 replay instructions](reproduce.md). The existing data and masks are read-only inputs to this analysis.
2. Use the [official FreeSurfer 7.4.1 Linux build](https://surfer.nmr.mgh.harvard.edu/fswiki/rel7downloads) for your environment. The local WSL setup uses the official Ubuntu 22 amd64 package and `tcsh`. Source the setup script. Use the [official registration form](https://surfer.nmr.mgh.harvard.edu/registration.html) to obtain the free license; keep the key outside version control.
3. Set `FS_LICENSE` to the absolute path to the license and activate the repository's Python environment:

```bash
export FREESURFER_HOME=/path/to/freesurfer/7.4.1
source "$FREESURFER_HOME/SetUpFreeSurfer.sh"
export FS_LICENSE=/path/to/license.txt
python -m src.run_freesurfer --preflight
python -m pytest -q
```

## Run in order

```bash
python -m src.run_freesurfer --smoke
# Inspect the printed training QC PNG: all three overlays must align spatially.
python -m src.run_freesurfer --test --smoke-reviewed
python -m src.evaluate_freesurfer
python -m scripts.audit_freesurfer
```

Smoke checks use the first three training subjects and produce no test scores. If setup or geometry fails, diagnose it from the console logs and repeat only the training smoke. The test runner requires a successful smoke with the same build and won't overwrite a started test run. The evaluator requires 25 masks and records a one-pass scoring marker.

Optional `--jobs N` runs independent subjects concurrently; each `recon-all` invocation still uses `-openmp 1`. The audit checks the saved masks, scoring record, and frozen inputs without re-running FreeSurfer or scoring any subject again.

The first local training attempt produced a valid native mask but failed while writing a NumPy dimension to JSON. After correcting serialization, all three training cases passed in fresh directories and the native overlays were reviewed. The repeated first case produced a byte-identical mask. The algorithm and its flags were unchanged. Before starting the test batch, the runner also checks every T1w and reference label hash against the frozen robustfov comparison.

Run outputs and FreeSurfer subject directories are ignored in `results/freesurfer_autorecon1/`. Reports, scores, hashes and QC figures are written separately to `results/supplementary/freesurfer_autorecon1/`. Existing primary and robustfov reports are checked by SHA-256 before and after each stage. FreeSurfer is not required to reproduce the earlier U-Net and BET analyses.
