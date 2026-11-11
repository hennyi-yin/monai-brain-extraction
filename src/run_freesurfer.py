"""Run the pre-declared FreeSurfer 7.4.1 autorecon1 baseline."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
import uuid

import nibabel as nib
from nibabel.processing import resample_from_to
import numpy as np

from src.common import ROOT, config, digest, load_split, write_json
from src.freesurfer_common import MASKS, METHOD, OUTPUT, VERSION, protocol_commit, read_json, relative, verify_frozen
from src.run_bet_robustfov import save_mask


def same_grid(a, b):
    return (len(a.shape) == 3 and a.shape == b.shape
            and np.allclose(a.affine, b.affine, atol=0.001, rtol=0))


def map_to_native(auto_mask, conformed, imported, native):
    if not same_grid(imported, native):
        raise ValueError("Imported scan shape or scanner-RAS affine differs from native input")
    if not same_grid(auto_mask, conformed):
        raise ValueError("FreeSurfer brainmask and conformed orig grids differ")
    values = np.asarray(auto_mask.dataobj)
    if not np.isfinite(values).all():
        raise ValueError("FreeSurfer brainmask contains nonfinite intensities")
    binary = nib.Nifti1Image((values != 0).astype(np.uint8), auto_mask.affine)
    result = resample_from_to(binary, (native.shape, native.affine), order=0, mode="constant", cval=0)
    native_mask = np.asarray(result.dataobj, dtype=np.uint8)
    checks = {"imported_grid_matches_native": True, "mask_grid_matches_conformed_orig": True,
              "native_shape": list(native.shape), "native_affine": native.affine.tolist(),
              "conformed_shape": list(auto_mask.shape), "conformed_affine": auto_mask.affine.tolist(),
              "conformed_foreground_voxels": int(np.count_nonzero(values)),
              "native_foreground_voxels": int(native_mask.sum())}
    if not native_mask.any():
        raise ValueError("Mapped FreeSurfer mask is empty")
    return native_mask, checks


def preflight():
    home = os.environ.get("FREESURFER_HOME")
    license_path = os.environ.get("FS_LICENSE")
    if not home or not license_path or not Path(license_path).is_file() or not Path(license_path).stat().st_size:
        raise RuntimeError("Source FreeSurfer 7.4.1 and set FS_LICENSE to a local license.txt file")
    bin_dir = Path(home).resolve() / "bin"
    tools = {}
    for name in ("recon-all", "mri_convert", "mri_watershed", "mri_normalize", "mri_em_register"):
        location = shutil.which(name)
        if not location or Path(location).resolve().parent != bin_dir:
            raise RuntimeError(f"{name} does not resolve to the selected FREESURFER_HOME/bin")
        tools[name] = {"path": str(Path(location).resolve()), "sha256": digest(location)}
    version = subprocess.check_output([tools["recon-all"]["path"], "-version"], text=True, stderr=subprocess.STDOUT).strip()
    if not re.search(rf"(?<![\d.]){re.escape(VERSION)}(?![\d.])", version):
        raise RuntimeError(f"Expected FreeSurfer {VERSION}; got {version!r}")
    stamp = Path(home) / "build-stamp.txt"
    return {"version": version, "build_stamp": stamp.read_text().strip(), "tools": tools,
            "protocol_commit": protocol_commit(), "omp_threads": 1,
            "split_sha256": digest(ROOT / config()["split_file"])}


def run_subject(item, subjects, masks, executable, fingerprint):
    sid = item["id"]
    native = nib.load(item["image"])
    command = [executable, "-i", item["image"], "-s", sid, "-sd", str(subjects), "-autorecon1", "-openmp", "1"]
    log = masks / f"{sid}.console.log"
    output = masks / f"{sid}_mask.nii.gz"
    mri = subjects / sid / "mri"
    raw = mri / "brainmask.auto.mgz"
    row = {"id": sid, "command": command, "console_log": relative(log), "inputs": fingerprint,
           "output": relative(output), "status": "ok", "failure": None, "mapping": None,
           "returncode": None, "auto_mask_sha256": None}
    env = {key: value for key, value in os.environ.items() if key not in
           {"FS_ALLOW_DEEP", "FS_USE_SYNTHSTRIP", "FS_USE_SYNTHSEG", "SUBJECTS_DIR", "FS_OMP_NUM_THREADS"}}
    env.update(OMP_NUM_THREADS="1", SUBJECTS_DIR=str(subjects))
    start = time.perf_counter()
    try:
        with log.open("wb") as stream:
            row["returncode"] = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, env=env).returncode
        if row["returncode"]:
            raise ValueError(f"recon-all exited with code {row['returncode']}")
        # Check the algorithm actually executed, not just that an MGZ exists.
        command_log = subjects / sid / "scripts/recon-all.cmd"
        lines = command_log.read_text(errors="replace").splitlines()
        watershed = [line.strip() for line in lines if re.search(r"(?:^|[/\s])mri_watershed\s", line)
                     and "brainmask.auto.mgz" in line]
        if not watershed:
            raise ValueError("No mri_watershed brainmask.auto.mgz command in recon-all.cmd")
        row["watershed_command"] = watershed[-1]
        row["auto_mask_sha256"] = digest(raw)
        prediction, row["mapping"] = map_to_native(nib.load(raw), nib.load(mri / "orig.mgz"),
                                                    nib.load(mri / "orig/001.mgz"), native)
    except (OSError, ValueError, nib.filebasedimages.ImageFileError) as error:
        row.update(status="failed", failure=str(error))
        prediction = np.zeros(native.shape, dtype=np.uint8)
    save_mask(prediction, native, output)
    row.update(elapsed_seconds=round(time.perf_counter() - start, 3), output_sha256=digest(output),
               console_log_sha256=digest(log))
    return row


def fingerprints(items):
    result = {}
    for item in items:
        native, truth = (nib.load(item[k]) for k in ("image", "label"))
        if not same_grid(native, truth):
            raise ValueError(f"Shared inputs do not match for {item['id']}")
        result[item["id"]] = {k: digest(item[k]) for k in ("image", "label")}
    return result


def check_smoke(signature, items):
    manifest = read_json(MASKS / "smoke_manifest.json")
    if (manifest["signature"] != signature or not manifest["complete"]
            or [row["id"] for row in manifest["subjects"]] != [item["id"] for item in items]
            or any(row["status"] != "ok" or digest(ROOT / row["output"]) != row["output_sha256"]
                   for row in manifest["subjects"])):
        raise RuntimeError("Run a successful three-subject smoke with this exact build first")
    if digest(ROOT / manifest["qc"]) != manifest["qc_sha256"]:
        raise RuntimeError("Training QC figure changed after generation")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--preflight", action="store_true")
    group.add_argument("--smoke", action="store_true", help="First three training subjects only")
    group.add_argument("--test", action="store_true", help="Run the 25 frozen test subjects once")
    parser.add_argument("--smoke-reviewed", action="store_true", help="Confirm inspection of training QC overlays")
    parser.add_argument("--jobs", type=int, default=1,
                        help="Concurrent subjects; each recon-all command still uses one thread")
    args = parser.parse_args()
    if not 1 <= args.jobs <= (os.cpu_count() or 1):
        parser.error("--jobs must be between 1 and the available logical CPU count")
    verify_frozen()
    signature = preflight()
    if args.preflight:
        print(signature["version"])
        return
    training = load_split(config(), ("train",))["train"][:3]
    if args.test:
        if not args.smoke_reviewed:
            parser.error("Review the three training overlays before test processing")
        reviewed_smoke = check_smoke(signature, training)
        items = load_split(config(), ("test",))["test"]
        masks = MASKS / "test"
        manifest_path = OUTPUT / "manifest.json"
        if manifest_path.exists() or masks.exists():
            raise RuntimeError("Test processing already started; refusing to rerun it")
    else:
        items = training
        masks = MASKS / "smoke" / uuid.uuid4().hex[:12]
        manifest_path = MASKS / "smoke_manifest.json"
    hashes = fingerprints(items)
    subjects = masks / "subjects"
    subjects.mkdir(parents=True, exist_ok=False)
    manifest = {"method": METHOD, "group": "test" if args.test else "train", "signature": signature,
                "concurrent_subjects": args.jobs,
                "complete": False, "subjects": []}
    if args.test:
        archived = OUTPUT / "training_smoke_manifest.json"
        shutil.copyfile(MASKS / "smoke_manifest.json", archived)
        figure = OUTPUT / "figures/training_qc.png"
        figure.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / reviewed_smoke["qc"], figure)
        manifest["reviewed_training_smoke_sha256"] = digest(archived)
    write_json(manifest_path, manifest)
    def process(item):
        return run_subject(item, subjects, masks, signature["tools"]["recon-all"]["path"], hashes[item["id"]])

    with ThreadPoolExecutor(max_workers=args.jobs) as workers:
        for row in workers.map(process, items):
            manifest["subjects"].append(row)
            write_json(manifest_path, manifest)
            print(f"{row['id']}: {row['status']} ({row['elapsed_seconds']:.1f}s) {row['failure'] or ''}", flush=True)
    if args.smoke:
        from src.evaluate_freesurfer import plot_overlays
        qc = masks / "qc.png"
        plot_overlays(items, manifest["subjects"], qc)
        manifest.update(qc=relative(qc), qc_sha256=digest(qc))
        print(f"Training QC: {qc}")
    manifest["complete"] = True
    verify_frozen()
    write_json(manifest_path, manifest)
    if args.smoke and any(row["status"] != "ok" for row in manifest["subjects"]):
        raise SystemExit("Training smoke did not pass; inspect the console logs before proceeding")


if __name__ == "__main__":
    main()
