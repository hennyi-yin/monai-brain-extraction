import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from scripts.prepare_external_data import select_ids
from scripts.external_geometry_sensitivity import integer_map
from src.evaluate_external import hd_quantile, paired_interval, summarize
from src.external_common import COHORTS, METHODS


def test_select_entire_publisher_test_cohorts_and_reject_duplicate():
    names = [f"{cohort}_t1_{i:03d}" for cohort, count in COHORTS.items() for i in range(count)]
    text = "\n".join(names + ["ixi_t2_002", "qin_t1_003", "fsm_t1_001", "asl_epi_010"])
    assert select_ids(text) == sorted(names)
    with pytest.raises(ValueError, match="distinct test IDs"):
        select_ids(text + "\n" + names[0])


def test_pooled_bootstrap_weights_subjects_not_cohort_means():
    assert paired_interval([np.zeros(48), np.ones(41)]) == pytest.approx([41 / 89, 41 / 89])
    assert hd_quantile([0, 1, float("inf")], .5) == 1
    assert hd_quantile([0, 1, float("inf")], .75) == "inf"


def test_paired_summary_keeps_empty_mask_in_denominator():
    rows = []
    for cohort, n in COHORTS.items():
        for i in range(n):
            for method in METHODS:
                failed = (cohort, i, method) == ("asl", 0, "bet_crop")
                rows.append({"cohort": cohort, "id": f"{cohort}_{i}", "method": method, "dice": 0 if failed else .9,
                             "hd95_mm": float("inf") if failed else 1., "status": "failed" if failed else "ok",
                             "crop_reference_retention": None})
    frame = pd.DataFrame(rows).sample(frac=1, random_state=12)
    result, _ = summarize(frame)
    pooled = result["groups"]["pooled"]
    assert pooled["methods"]["bet_crop"]["processing_failures"] == 1
    assert pooled["methods"]["bet_crop"]["hd95_infinite"] == 1
    assert pooled["unet_minus_comparator"]["bet_crop"]["mean_points"] == pytest.approx(90 / 89)
    assert pooled["unet_minus_comparator"]["bet_crop"]["ties"] == 88
    with pytest.raises(ValueError, match="full valid paired design"):
        summarize(frame.iloc[:-1])


def test_exact_crop_mapping_preserves_boundary_and_rejects_misalignment():
    values = np.arange(7 * 8 * 9, dtype=np.int16).reshape(7, 8, 9)
    native = nib.Nifti1Image(values, np.diag([1., 1.2, .9, 1.]))
    crop = native.slicer[0:6, 1:7, 2:9]
    orientation = np.array([[1, -1], [2, 1], [0, -1]])
    roi = crop.as_reoriented(orientation)
    raw = nib.Nifti1Image((np.asarray(roi.dataobj) % 4 == 0).astype(np.uint8), roi.affine)
    mapped, _, evidence = integer_map(roi, raw, native)
    expected = np.zeros(values.shape, dtype=np.uint8)
    expected[:6, 1:7, 2:] = values[:6, 1:7, 2:] % 4 == 0
    np.testing.assert_array_equal(mapped, expected)
    assert evidence["roi_intensities_exact"]
    altered = nib.Nifti1Image(np.asarray(roi.dataobj) + 1, roi.affine)
    with pytest.raises(ValueError, match="does not match native intensities"):
        integer_map(altered, raw, native)


def test_crop_mapping_accepts_only_exact_float32_representation():
    native = nib.Nifti1Image(np.arange(64, dtype=np.float64).reshape(4, 4, 4) * 1.835897445678711, np.eye(4))
    original = np.asarray(native.dataobj)[1:3]
    affine = native.slicer[1:3].affine
    roi = nib.Nifti1Image(original.astype(np.float32), affine)
    raw = nib.Nifti1Image(np.ones(roi.shape, dtype=np.uint8), affine)
    _, _, evidence = integer_map(roi, raw, native)
    assert not evidence["roi_intensities_exact"] and evidence["roi_matches_after_native_float32_cast"]
    altered = np.asarray(roi.dataobj).copy()
    altered[0, 0, 0] = np.nextafter(altered[0, 0, 0], np.float32(np.inf))
    with pytest.raises(ValueError, match="stored precision"):
        integer_map(nib.Nifti1Image(altered, affine), raw, native)
