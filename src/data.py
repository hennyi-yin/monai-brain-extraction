from monai.transforms import (
    AsDiscreted, Compose, EnsureChannelFirstd, EnsureTyped, LoadImaged,
    NormalizeIntensityd, Orientationd, RandAffined, RandCropByPosNegLabeld,
    RandFlipd, SpatialPadd,
)


def transforms(cfg, training=False, labels=True):
    keys = ["image", "label"] if labels else ["image"]
    base = [LoadImaged(keys), EnsureChannelFirstd(keys),
            Orientationd(keys, axcodes="RAS"),
            NormalizeIntensityd("image", nonzero=True, channel_wise=True)]
    if labels:
        base.append(AsDiscreted("label", threshold=0.5))
    if training:
        base.extend([
            SpatialPadd(keys, spatial_size=cfg["train"]["roi_size"]),
            RandCropByPosNegLabeld(keys, label_key="label", spatial_size=cfg["train"]["roi_size"],
                                  pos=1, neg=1, num_samples=cfg["train"]["num_samples"]),
            RandFlipd(keys, prob=0.5, spatial_axis=0),
            RandAffined(keys, prob=0.3, rotate_range=(0.1,) * 3, scale_range=(0.1,) * 3,
                        mode=("bilinear", "nearest"), padding_mode="zeros"),
        ])
    return Compose(base + [EnsureTyped(keys, dtype="float32")])
