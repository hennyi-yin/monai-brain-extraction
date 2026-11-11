import json

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from src.evaluate_freesurfer import summarize
from src.freesurfer_common import METHOD
from src.run_freesurfer import map_to_native, save_mask


def test_mgz_map_back_with_flip_and_different_voxel_sizes(tmp_path):
    native_affine = np.eye(4)
    native_affine[:3, 3] = [-3, 15, -7]
    native = nib.Nifti1Image(np.zeros((9, 9, 10), dtype=np.int16), native_affine)
    conformed_to_native_voxels = np.array([[0, -2, 0, 10], [0, 0, 1, 1], [1, 0, 0, 2], [0, 0, 0, 1]])
    conform_affine = native_affine @ conformed_to_native_voxels
    intensity = np.zeros((7, 8, 9), dtype=np.int16)
    intensity[1:3, 2:5, 1:4] = 37
    intensity[1, 2, 1] = -13
    mgz = tmp_path / "brainmask.auto.mgz"
    nib.save(nib.MGHImage(intensity, conform_affine), mgz)
    raw = nib.load(mgz)
    imported = nib.MGHImage(np.asarray(native.dataobj), native_affine)
    values, checks = map_to_native(raw, raw, imported, native)
    assert json.loads(json.dumps(checks, allow_nan=False))["conformed_shape"] == [7, 8, 9]
    expected = np.zeros(native.shape, dtype=np.uint8)
    expected[2:8, 2:5, 3:5] = 1
    np.testing.assert_array_equal(values, expected)
    assert checks["conformed_foreground_voxels"] == 18
    assert checks["native_foreground_voxels"] == 36
    output = tmp_path / "native.nii.gz"
    save_mask(values, native, output)
    saved = nib.load(output)
    assert saved.get_data_dtype() == np.uint8
    np.testing.assert_array_equal(saved.affine, native.affine)
    np.testing.assert_array_equal(np.asarray(saved.dataobj), expected)


@pytest.mark.parametrize(("shifted", "message"), [("import", "Imported scan"), ("mask", "conformed orig")])
def test_reject_same_shape_with_wrong_header(shifted, message):
    data = np.ones((4, 5, 6), dtype=np.int16)
    native = nib.Nifti1Image(data, np.eye(4))
    affine = np.eye(4)
    affine[0, 3] = 1
    incorrect = nib.MGHImage(data, affine)
    with pytest.raises(ValueError, match=message):
        map_to_native(incorrect if shifted == "mask" else native, native,
                      incorrect if shifted == "import" else native, native)


def test_summary_keeps_failed_subject_and_pairs_shuffled_ids():
    records = []
    for i in range(25):
        for method, dice in {"unet": .95, "bet_robust": .70, "bet_robustfov": .75 + i / 200,
                             METHOD: .85 if i else 0}.items():
            records.append({"id": f"s{i:02d}", "method": method, "dice": dice,
                            "hd95": 2 if dice else float("inf")})
    result, _ = summarize(pd.DataFrame(records).sample(frac=1, random_state=42))
    assert result["methods"][METHOD]["zero_overlap_count"] == 1
    assert result["methods"][METHOD]["hd95_infinite_count"] == 1
    assert result["methods"][METHOD]["dice_mean"] == pytest.approx(.85 * 24 / 25)
    assert result["freesurfer_minus_comparator"]["bet_robustfov"]["freesurfer_wins"] == 19
    assert result["freesurfer_minus_comparator"]["bet_robustfov"]["ties"] == 1
