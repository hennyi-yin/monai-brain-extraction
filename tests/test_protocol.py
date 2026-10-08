import nibabel as nib
import numpy as np
import pandas as pd
import pytest
import torch
from monai.metrics import compute_hausdorff_distance

from src.common import check_grid
from src.infer import postprocess, save_native
from src.metrics import paired_test, score
from src.report import summarize


def test_hd95_matches_monai_and_uses_mm():
    a = np.zeros((20, 20, 20), dtype=np.uint8)
    a[4:12, 4:12, 4:12] = 1
    b = np.roll(a, 2, axis=0)
    spacing = (2, 1, 1)
    _, hd = score(a, b, spacing)
    onehot = lambda m: torch.from_numpy(np.stack([1 - m, m])).unsqueeze(0)
    expected = compute_hausdorff_distance(onehot(a), onehot(b), include_background=False,
                                         percentile=95, spacing=spacing).item()
    assert hd == pytest.approx(expected)
    assert score(a, a) == (1.0, 0.0)


def test_empty_masks():
    empty = np.zeros((4, 4, 4))
    full = np.ones_like(empty)
    assert score(empty, empty) == (1.0, 0.0)
    assert score(empty, full) == (0.0, float("inf"))


@pytest.mark.parametrize("codes", [("L", "P", "S"), ("S", "R", "A"), ("I", "P", "L")])
def test_native_orientation_roundtrip(tmp_path, codes):
    ras = np.zeros((7, 9, 11), dtype=np.uint8)
    ras[1:4, 3:6, 2:8] = 1
    ras_image = nib.Nifti1Image(ras, np.diag([1., 1., 1., 1.]))
    change = nib.orientations.ornt_transform(nib.orientations.axcodes2ornt(("R", "A", "S")),
                                            nib.orientations.axcodes2ornt(codes))
    native = ras_image.as_reoriented(change)
    source, destination = tmp_path / "scan.nii.gz", tmp_path / "mask.nii.gz"
    nib.save(native, source)
    save_native(ras, source, destination)
    saved = nib.load(destination)
    check_grid(native, saved)
    np.testing.assert_array_equal(saved.get_fdata(), native.get_fdata())
    np.testing.assert_array_equal(nib.as_closest_canonical(saved).get_fdata(), ras)
    assert saved.get_data_dtype() == np.dtype("uint8")


def test_component_and_hole_cleanup():
    mask = np.zeros((10, 10, 10), dtype=np.uint8)
    mask[2:8, 2:8, 2:8] = 1
    mask[4, 4, 4] = 0
    mask[0, 0, 0] = 1
    result = postprocess(mask)
    assert result[4, 4, 4] == 1 and result[0, 0, 0] == 0
    assert result.sum() == 216


def test_paired_stats_align_subjects_and_unrounded_delta():
    rows = []
    for sid, u, default, robust in (("A", .965, .910, .920), ("B", .975, .940, .950), ("C", .960, .900, .915)):
        for method, dice in (("unet", u), ("bet_default", default), ("bet_robust", robust)):
            rows.append({"id": sid, "method": method, "dice": dice, "hd95": 2.0})
    stats = summarize(pd.DataFrame(rows).sample(frac=1, random_state=42))
    assert stats["selected_bet"] == "bet_robust"
    assert stats["resume"]["delta_points_unrounded"] == pytest.approx((.965 + .975 + .960 - .920 - .950 - .915) / 3 * 100)
    assert paired_test([.9, .9], [.9, .9]) == 1.0


def test_same_shape_different_affine_rejected():
    data = np.zeros((3, 3, 3))
    shifted = np.eye(4)
    shifted[0, 3] = 1
    with pytest.raises(ValueError, match="affine"):
        check_grid(nib.Nifti1Image(data, np.eye(4)), nib.Nifti1Image(data, shifted))
