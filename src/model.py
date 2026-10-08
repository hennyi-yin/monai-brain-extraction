from monai.networks.nets import UNet


def make_model(cfg):
    spec = cfg["model"]
    return UNet(spatial_dims=3, in_channels=1, out_channels=2,
                channels=tuple(spec["channels"]), strides=tuple(spec["strides"]),
                num_res_units=spec["num_res_units"], norm="INSTANCE")
