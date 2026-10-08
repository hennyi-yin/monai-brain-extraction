"""One pre-declared post-hoc neck-cropping pipeline, with audited native mapping."""
import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to

from src.common import check_grid, config, digest, load_split, path, write_json
from src.supplementary_common import MASKS, OUTPUT, PROTOCOL_COMMIT, frozen_state


def command(argv, records):
    result = subprocess.run([str(a) for a in argv], env={**os.environ, "FSLOUTPUTTYPE": "NIFTI_GZ"},
                            capture_output=True, text=True)
    records.append({"argv": [str(a) for a in argv], "returncode": result.returncode,
                    "stdout": result.stdout, "stderr": result.stderr})
    if result.returncode:
        raise RuntimeError(f"Command failed ({result.returncode}): {argv}: {result.stderr}")


def map_back(roi, mask, native):
    """Nearest-neighbour world-coordinate mapping, including zero-valued ROI voxels."""
    footprint = nib.Nifti1Image(np.ones(roi.shape, dtype=np.uint8), roi.affine)
    target = (native.shape, native.affine)
    return tuple(resample_from_to(volume, target, order=0) for volume in (roi, mask, footprint))


def mapping_checks(roi, mask, native, mapped):
    image, prediction, footprint = (np.asarray(v.dataobj) for v in mapped)
    inside = footprint > 0.5
    difference = np.abs(image[inside].astype(np.float64) - np.asarray(native.dataobj)[inside])
    max_diff = float(difference.max()) if inside.any() else None
    source_count = int(np.count_nonzero(np.asarray(mask.dataobj) > 0.5))
    mapped_count = int(np.count_nonzero(prediction > 0.5))
    checks = {"roi_t1w_max_abs_diff": max_diff,
              "roi_footprint_voxels_original": int(np.prod(roi.shape)),
              "roi_footprint_voxels_native": int(inside.sum()),
              "mask_voxels_original": source_count, "mask_voxels_native": mapped_count}
    checks["passed"] = bool(max_diff == 0 and int(inside.sum()) == int(np.prod(roi.shape))
                            and source_count == mapped_count)
    return checks


def save_mask(values, native, filename):
    header = native.header.copy()
    header.set_data_dtype(np.uint8)
    image = nib.Nifti1Image(np.asarray(values, dtype=np.uint8), native.affine, header)
    image.set_qform(native.get_qform(), int(native.header["qform_code"]))
    image.set_sform(native.get_sform(), int(native.header["sform_code"]))
    nib.save(image, filename)
    check_grid(native, nib.load(filename))


def fsl_coordinate_flip(image):
    """Convert unflipped scaled voxels to FLIRT coordinates (FSL/fslpy convention)."""
    matrix = np.eye(4)
    if np.linalg.det(image.affine[:3, :3]) > 0:
        matrix[0, 0] = -1
        matrix[0, 3] = (image.shape[0] - 1) * float(image.header.get_zooms()[0])
    return matrix


def fallback(roi_file, mask_file, native_file, native, work, tools, records):
    for name in ("convert_xfm", "flirt"):
        if not tools.get(name):
            raise RuntimeError(f"FSL matrix fallback requires {name} on PATH")
    reorient_mat = work / "native_to_reoriented.mat"
    flirt_reorient_mat = work / "native_to_reoriented_flirt.mat"
    inverse = work / "reoriented_to_native.mat"
    combined = work / "roi_to_native.mat"
    command([tools["fslreorient2std"], "-m", reorient_mat, native_file], records)
    # fslswapdim's reported matrix uses unflipped scaled voxels; FLIRT uses an
    # additional X flip for positive-determinant images. Convert the basis,
    # without estimating or optimising a registration.
    reoriented = nib.load(work / "reoriented.nii.gz")
    converted = fsl_coordinate_flip(reoriented) @ np.loadtxt(reorient_mat) @ fsl_coordinate_flip(native)
    np.savetxt(flirt_reorient_mat, converted, fmt="%.12g")
    write_json(work / "matrix_coordinate_conversion.json", {
        "definition": "F_reoriented @ fslreorient2std_scaled_voxel_matrix @ F_native",
        "reference": "https://pages.fmrib.ox.ac.uk/fsl/fslpy/fsl.transform.flirt.html",
        "original": np.loadtxt(reorient_mat).tolist(), "FLIRT_basis": converted.tolist()})
    command([tools["convert_xfm"], "-omat", inverse, "-inverse", flirt_reorient_mat], records)
    command([tools["convert_xfm"], "-omat", combined, "-concat", inverse,
             work / "roi_to_reoriented.mat"], records)
    roi = nib.load(roi_file)
    footprint = work / "roi_footprint.nii.gz"
    nib.save(nib.Nifti1Image(np.ones(roi.shape, dtype=np.uint8), roi.affine), footprint)
    outputs = []
    for label, source in (("roi", roi_file), ("mask", mask_file), ("footprint", footprint)):
        destination = work / f"mapped_{label}.nii.gz"
        command([tools["flirt"], "-in", source, "-ref", native_file, "-applyxfm", "-init", combined,
                 "-interp", "nearestneighbour", "-setbackground", "0", "-noclamp", "-noresampblur",
                 "-out", destination], records)
        outputs.append(nib.load(destination))
    return tuple(outputs)


def environment():
    tools = {name: shutil.which(name) for name in ("fslreorient2std", "robustfov", "bet", "bet2", "convert_xfm", "flirt")}
    if not all(tools[name] for name in ("fslreorient2std", "robustfov", "bet", "bet2")):
        raise RuntimeError("Required FSL executables are unavailable")
    root = Path(os.environ["FSLDIR"])
    version_file = root / "etc" / "fslversion"
    version_read = subprocess.run(["cat", str(version_file)], capture_output=True, text=True)
    template = root / "data" / "standard" / "MNI152_T1_2mm_brain.nii.gz"
    if not template.exists():
        raise RuntimeError("fslreorient2std requires fsl-data_standard; install it before any subject run")
    result = subprocess.run(["which", "fslreorient2std", "robustfov", "bet"], capture_output=True, text=True)
    packages = []
    for filename in sorted((root / "conda-meta").glob("fsl-*.json")):
        item = json.loads(filename.read_text())
        packages.append({key: item[key] for key in ("name", "version", "build")})
    return {"FSLDIR": str(root), "fslversion_path": str(version_file),
            "fslversion": version_file.read_text().strip() if version_file.exists() else None,
            "fslversion_status": "available" if version_file.exists() else "missing in modular conda installation",
            "fslversion_read_command": {"argv": ["cat", str(version_file)], "returncode": version_read.returncode,
                                        "stdout": version_read.stdout, "stderr": version_read.stderr},
            "standard_orientation_reference": {"path": str(template), "sha256": digest(template)},
            "which_command": {"argv": ["which", "fslreorient2std", "robustfov", "bet"],
                              "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr},
            "executables": tools, "executable_sha256": {name: digest(file) for name, file in tools.items() if file},
            "fsl_packages": packages}


def run_subject(item, group, tools, existing=None):
    sid = item["id"]
    native = nib.load(item["image"])
    work = path(MASKS) / "work" / group / sid
    work.mkdir(parents=True, exist_ok=True)
    destination = path(MASKS) / ("smoke" if group == "train" else "") / f"{sid}_mask.nii.gz"
    destination.parent.mkdir(parents=True, exist_ok=True)
    record = {"id": sid, "group": group, "commands": [], "status": "failed",
              "input_sha256": {"image": digest(item["image"]), "truth": digest(item["label"])},
              "roi_brain_retention": None, "mapping": None, "failure": None}
    try:
        if existing:
            raise RuntimeError("Interrupted previous attempt; not rerunning the test pipeline")
        reoriented = work / "reoriented.nii.gz"
        roi_file = work / "roi.nii.gz"
        bet_out = work / "bet"
        mask_file = work / "bet_mask.nii.gz"
        command([tools["fslreorient2std"], item["image"], reoriented], record["commands"])
        command([tools["robustfov"], "-i", reoriented, "-r", roi_file, "-b", "170",
                 "-m", work / "roi_to_reoriented.mat"], record["commands"])
        command([tools["bet"], roi_file, bet_out, "-f", "0.5", "-R", "-m", "-n"], record["commands"])
        roi, mask = nib.load(roi_file), nib.load(mask_file)
        values = np.asarray(mask.dataobj)
        if not np.isfinite(values).all() or not set(np.unique(values)).issubset({0, 1}):
            raise ValueError("BET output is not a finite binary mask")
        mapped = map_back(roi, mask, native)
        checks = mapping_checks(roi, mask, native, mapped)
        record["nibabel_mapping"] = checks
        mapping_method = "nibabel.resample_from_to(order=0)"
        if not checks["passed"]:
            mapped = fallback(roi_file, mask_file, item["image"], native, work, tools, record["commands"])
            checks = mapping_checks(roi, mask, native, mapped)
            mapping_method = "FSL matrices and FLIRT nearestneighbour"
        record["mapping"] = {"method": mapping_method, **checks}
        if not checks["passed"]:
            raise ValueError(f"Native mapping verification failed: {checks}")
        truth_image = nib.load(item["label"])
        check_grid(native, truth_image)
        truth = np.asarray(truth_image.dataobj) >= 0.5
        footprint = np.asarray(mapped[2].dataobj) > 0.5
        record["roi_brain_retention"] = float(np.count_nonzero(truth & footprint) / truth.sum())
        save_mask(np.asarray(mapped[1].dataobj) > 0.5, native, destination)
        record["status"] = "ok"
    except Exception as error:
        record["failure"] = str(error)
        save_mask(np.zeros(native.shape, dtype=np.uint8), native, destination)
    record["output"] = destination.relative_to(path(".")).as_posix()
    record["output_sha256"] = digest(destination)
    record["intermediate_sha256"] = {file.name: digest(file) for file in sorted(work.iterdir()) if file.is_file()}
    if existing:
        record["commands"] = existing.get("commands", [])
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--group", choices=("train", "test"), default="test")
    args = parser.parse_args()
    cfg = config(args.config)
    frozen_state()
    env = environment()
    out = path(OUTPUT)
    filename = out / ("smoke_manifest.json" if args.group == "train" else "manifest.json")
    items = load_split(cfg, (args.group,))[args.group]
    if args.group == "train":
        items = items[:3]
    else:
        smoke = json.loads((out / "smoke_manifest.json").read_text())
        if len(smoke["subjects"]) != 3 or any(item["status"] != "ok" for item in smoke["subjects"]):
            raise RuntimeError("All three training smoke subjects must pass before test processing")
    signature = {"split_sha256": digest(path(cfg["split_file"])), "config_sha256": digest(path(args.config)),
                 "group": args.group, "pipeline": "fslreorient2std -> robustfov -b 170 -> bet -f 0.5 -R -m -n",
                 "executable_sha256": env["executable_sha256"], "protocol_commit": PROTOCOL_COMMIT}
    manifest = {"post_hoc": True, "signature": signature, "environment": env,
                "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path(".")).decode().strip(),
                "source_sha256": {name: digest(path(name)) for name in ("src/run_bet_robustfov.py", "src/supplementary_common.py")},
                "subjects": [], "complete": False}
    if filename.exists():
        manifest = json.loads(filename.read_text())
        if manifest["signature"] != signature:
            raise RuntimeError("Supplementary outputs have a different signature; refusing reuse")
    for item in items:
        prior = next((row for row in manifest["subjects"] if row["id"] == item["id"]), None)
        if prior and prior["status"] in ("ok", "failed"):
            if digest(path(prior["output"])) != prior["output_sha256"]:
                raise ValueError("A previously generated supplementary mask changed")
            if any(digest(item[key]) != prior["input_sha256"][label] for label, key in (("image", "image"), ("truth", "label"))):
                raise ValueError("A previously processed supplementary input changed")
            continue
        if not prior:
            manifest["subjects"].append({"id": item["id"], "status": "attempt_started"})
            write_json(filename, manifest)
        record = run_subject(item, args.group, env["executables"], prior)
        manifest["subjects"] = [record if row["id"] == item["id"] else row for row in manifest["subjects"]]
        write_json(filename, manifest)
        print(item["id"], record["status"], record["mapping"], flush=True)
    manifest["complete"] = True
    write_json(filename, manifest)
    frozen_state()


if __name__ == "__main__":
    main()
