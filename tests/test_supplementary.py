import nibabel as nib
import numpy as np

from src.run_bet_robustfov import map_back, mapping_checks, save_mask, fsl_coordinate_flip
from src.common import check_grid


def test_permutation_flip_crop_maps_back_exactly(tmp_path):
    values = np.arange(9 * 11 * 13, dtype=np.int16).reshape(9, 11, 13)
    native = nib.Nifti1Image(values, np.diag([2., 1., 3., 1.]))
    change = nib.orientations.ornt_transform(nib.orientations.axcodes2ornt(("R", "A", "S")),
                                            nib.orientations.axcodes2ornt(("I", "P", "R")))
    reoriented = native.as_reoriented(change)
    # A reported reorientation transform must change coordinate bases before
    # use by FLIRT; verify the converted transform still represents world identity.
    native_scale = np.diag([*native.header.get_zooms(), 1.])
    reoriented_scale = np.diag([*reoriented.header.get_zooms(), 1.])
    scaled_transform = reoriented_scale @ np.linalg.inv(reoriented.affine) @ native.affine @ np.linalg.inv(native_scale)
    fsl_transform = fsl_coordinate_flip(reoriented) @ scaled_transform @ fsl_coordinate_flip(native)
    fsl_to_world = reoriented.affine @ np.linalg.inv(reoriented_scale) @ fsl_coordinate_flip(reoriented)
    world_to_fsl = fsl_coordinate_flip(native) @ native_scale @ np.linalg.inv(native.affine)
    np.testing.assert_allclose(fsl_to_world @ fsl_transform @ world_to_fsl, np.eye(4), atol=1e-12)
    roi = reoriented.slicer[2:11, 1:9, 2:7]
    mask_values = (np.asarray(roi.dataobj) % 3 == 0).astype(np.uint8)
    mask = nib.Nifti1Image(mask_values, roi.affine)
    mapped = map_back(roi, mask, native)
    checks = mapping_checks(roi, mask, native, mapped)
    assert checks["passed"] and checks["roi_t1w_max_abs_diff"] == 0
    footprint = np.asarray(mapped[2].dataobj) > 0
    np.testing.assert_array_equal(np.asarray(mapped[0].dataobj)[footprint], values[footprint])
    expected = np.zeros_like(values, dtype=np.uint8)
    expected[footprint] = (values[footprint] % 3 == 0).astype(np.uint8)
    np.testing.assert_array_equal(np.asarray(mapped[1].dataobj), expected)
    filename = tmp_path / "native_mask.nii.gz"
    save_mask(expected, native, filename)
    saved = nib.load(filename)
    check_grid(native, saved)
    np.testing.assert_array_equal(np.asarray(saved.dataobj), expected)
    assert saved.get_data_dtype() == np.dtype("uint8")
