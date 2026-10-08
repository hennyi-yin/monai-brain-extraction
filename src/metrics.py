import numpy as np
from scipy import ndimage
from scipy.stats import wilcoxon


def score(prediction, truth, spacing=(1, 1, 1)):
    """Foreground Dice and symmetric HD95 (max of directed 95th percentiles)."""
    p, g = np.asarray(prediction, dtype=bool), np.asarray(truth, dtype=bool)
    if p.shape != g.shape:
        raise ValueError("Metric grids differ")
    size = int(p.sum()) + int(g.sum())
    dice = 2 * np.count_nonzero(p & g) / size if size else 1.0
    if not p.any() or not g.any():
        return float(dice), 0.0 if not size else float("inf")
    structure = ndimage.generate_binary_structure(3, 1)
    ps = p ^ ndimage.binary_erosion(p, structure=structure, border_value=0)
    gs = g ^ ndimage.binary_erosion(g, structure=structure, border_value=0)
    distance = ndimage.distance_transform_edt(~gs, sampling=spacing)
    forward = np.percentile(distance[ps], 95)
    del distance
    distance = ndimage.distance_transform_edt(~ps, sampling=spacing)
    backward = np.percentile(distance[gs], 95)
    return float(dice), float(max(forward, backward))


def paired_test(unet, baseline):
    difference = np.asarray(unet) - np.asarray(baseline)
    if np.all(difference == 0):
        return 1.0
    return float(wilcoxon(difference, alternative="two-sided", zero_method="wilcox", method="auto").pvalue)
