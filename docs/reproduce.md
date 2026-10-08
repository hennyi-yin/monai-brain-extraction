# Reproduce the experiment

## Replay the published artifacts

Scoring and auditing existing masks work on **CPU**. Training and generating new U-Net predictions use the configured CUDA device. The recorded run used Linux/WSL, Python 3.12, MONAI 1.6.1, PyTorch 2.8.0 with CUDA 12.8, and an NVIDIA RTX 5070 with 12 GB VRAM.

Clone the repository, create an environment, and install the exact recorded Python dependencies:

```bash
git clone https://github.com/hennyi-yin/monai-brain-extraction.git
cd monai-brain-extraction
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements-lock.txt
```

Download the five assets from [v1.1.0](https://github.com/hennyi-yin/monai-brain-extraction/releases/tag/v1.1.0) into `artifacts/release/`. With GitHub CLI installed:

```bash
gh release download v1.1.0 --dir artifacts/release
python -m scripts.verify_release artifacts/release
python -m scripts.download_data
mkdir -p checkpoints
cp artifacts/release/best.pt checkpoints/best.pt
tar -xzf artifacts/release/test_predictions.tar.gz
tar -xzf artifacts/release/robustfov_predictions.tar.gz
python -m pytest -q
python -m scripts.audit_delivery
python -m scripts.audit_supplementary
```

The release includes the full best checkpoint, inference-only model state, all **75 primary + 25 supplementary** native-grid masks, and SHA-256 checksums. The primary archive also includes its mask manifests. The repository supplies the split, protocol, logs, per-subject scores and evaluation manifests. The raw NFBS scans are downloaded separately.

The two audits verify the saved analysis and native-grid masks without scoring the supplementary test set again. `audit_supplementary` also checks all 161 frozen primary file hashes and the original README results block. FSL is not required to audit the released masks.

For an independent replay of the **primary** metrics, `python evaluate.py` uses the frozen masks/checkpoint and rejects changed analysis inputs. It recomputes the 75 rows, statistics, figures and generated README block. A fresh environment installed from the lock file previously reproduced the primary CSV byte for byte: [reproduction evidence](../results/reproducibility.json). The supplementary `evaluation_once.json` deliberately prevents a second scoring pass in this completed workspace.

## Train the primary model independently

Use a **separate clone/workspace** for new training. Keep the committed split/config and the test set out of model selection. Clear previously recorded output metadata only in that separate workspace before generating a new run; retain this published repository as the reference record.

Install `requirements.txt` (the fixed direct dependencies) or `requirements-lock.txt` (the recorded transitive environment). An NVIDIA GPU with at least 12 GB VRAM is recommended for the recorded configuration. Set up FSL separately with `bet` on `PATH`, `FSLDIR` pointing to the FSL installation, and `FSLOUTPUTTYPE=NIFTI_GZ`. Ubuntu requires the `dc` calculator for robust BET (`sudo apt-get install dc`). The original FSL package builds are in [environment-fsl-explicit.txt](../environment-fsl-explicit.txt).

```bash
python -m scripts.download_data
python -m src.make_split
python -m src.qc
python -m pytest -q
python -m scripts.smoke
python train.py --config configs/config.yaml
bash scripts/run_bet.sh
python evaluate.py
python -m scripts.audit_delivery
```

`bash scripts/run_project.sh` runs the primary data audit, preprocessing QC, training, fixed BET baselines and evaluation after dependencies and data are installed. `python train.py --resume` resumes a stopped run. Relative paths resolve against the repository root. GPU, driver and library differences can affect a newly trained model even with seed 42.

## Reproduce the supplementary pipeline independently

The [post-hoc addendum](protocol.md#addendum-2026-10-08-post-hoc-supplementary-baseline) fixes the pipeline and evaluation policy. The completed supplementary test set was scored **once**; do not rerun its evaluation in the published workspace.

For an independent implementation run, use a separate workspace. Retain the frozen primary checkpoint, split, configuration and primary predictions/scores. Archive the published supplementary records and use fresh supplementary output directories. The runner's existing-manifest checks and evaluator's one-pass guard intentionally reject changed inputs or repeated scoring under the same record.

FSL must provide `fslreorient2std`, `robustfov`, `bet`, `convert_xfm` and `flirt`, including the standard MNI orientation reference. Exact modular package builds for the measured supplementary run are in [its FSL environment lock](../results/supplementary/robustfov_bet/environment-fsl-explicit.txt). Set:

```bash
export FSLDIR=/path/to/fsl
export PATH="$FSLDIR/bin:$PATH"
export FSLOUTPUTTYPE=NIFTI_GZ
which fslreorient2std robustfov bet
```

Each subject receives exactly:

```bash
fslreorient2std <native_T1w> <reoriented_T1w>
robustfov -i <reoriented_T1w> -r <roi_T1w> -b 170 -m <roi_to_reoriented.mat>
bet <roi_T1w> <output> -f 0.5 -R -m -n
```

Use `python -m src.run_bet_robustfov --help` for the runner's training-only smoke and test modes. Run implementation checks on the first three training subjects, then process the 25 test scans once. `python -m src.evaluate_supplementary` performs the one-pass supplementary evaluation. Native-map checks and failure policy are specified in the addendum; no crop/BET tuning or case exclusion is permitted.

## Where to look

| Artifact | Location |
|---|---|
| Primary per-subject metrics | [results/per_subject.csv](../results/per_subject.csv) |
| Primary summaries and hashes | [results/summary.md](../results/summary.md), [evaluation_inputs.json](../results/evaluation_inputs.json) |
| Supplementary metrics, commands, geometry | [results/supplementary/robustfov_bet/](../results/supplementary/robustfov_bet/) |
| Training history and environment | [logs/](../logs/) |
| Source and tests | [src/](../src/), [tests/](../tests/) |
| Weights and mask volumes | [GitHub release assets](https://github.com/hennyi-yin/monai-brain-extraction/releases/tag/v1.1.0) |

Raw scans, checkpoints, caches, mask volumes and local artifacts are excluded from Git. `python -m scripts.package_release` prepares a primary release from a completed run. `python -m scripts.package_supplementary_release` preserves those assets and adds the 25 verified supplementary masks in `artifacts/release-v1.1.0/`; neither command publishes or rescans the test set.
