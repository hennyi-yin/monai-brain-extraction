# Native-grid mapping checks

The initial training smoke attempt found that the modular FSL environment lacked the standard MNI orientation reference. The official `fsl-data_standard` package and FLIRT were installed without changing any existing package builds. The same three training subjects were used; no test subject had been processed. The failed installation attempt is retained in `smoke_initial_missing_template.json`.

The next training run passed for two subjects. In A00037511, direct NiBabel nearest-neighbour mapping preserved ROI intensities (max difference 0) and BET foreground count, but floating-point header translation caused one edge plane of the ROI footprint to be omitted. The stricter footprint-count check therefore triggered the pre-declared FSL fallback. This intermediate smoke manifest is retained in `smoke_initial_mapping.json`.

FLIRT requires zero exterior background for the all-one ROI footprint; its default background extrapolation otherwise fills the exterior. The fallback therefore explicitly sets `-setbackground 0`. `-noclamp -noresampblur` ensure that the mapping step never clamps or blurs the T1w intensities. These are fixed mapping options, not cropping or BET parameter changes.

The matrix printed by `fslreorient2std -m` / `fslswapdim` uses unflipped scaled voxel coordinates. [FLIRT coordinates](https://pages.fmrib.ox.ac.uk/fsl/fslpy/fsl.transform.flirt.html) additionally flip X when the affine determinant is positive. The recorded matrix is converted to that basis as `F_reoriented @ M @ F_native`, then inverted/concatenated with `convert_xfm`. No registration is estimated. Each fallback saves both matrices and its coordinate-conversion JSON, with hashes in the subject manifest.

The training repair reused the existing ROI and BET mask; it did not rerun cropping or BET or choose parameters from Dice. After the coordinate correction, A00037511 passed with maximum intensity difference 0, 8,355,840 ROI voxels preserved and 1,246,737 mask voxels preserved. All three training smoke subjects passed before the code commit and test-subject processing. Every test subject must satisfy the same checks, or receive an empty failure mask under the addendum.

`training_geometry_debug.json` records the additional training-only matrix experiments, reconstructed from retained debugging scripts and terminal results. Production commands and verified output hashes are in `smoke_manifest.json` and `manifest.json`.
