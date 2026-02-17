"""Model module — registry and builder.

Usage:
    from src.models import build_model
    model = build_model(config, dataset_info)
"""

from src.models.unet_lomtev import AutoEncoder1D
from src.models.unet_raw import AutoEncoder1DRaw
from src.models.tcn import TCN


def _build_unet_lomtev(model_cfg, n_channels, n_input_features, n_targets):
    return AutoEncoder1D(
        n_electrodes=n_channels,
        n_freqs=n_input_features,
        n_channels_out=n_targets,
        channels=model_cfg["channels"],
        kernel_sizes=model_cfg["kernel_sizes"],
        strides=model_cfg["strides"],
        dilation=model_cfg["dilation"],
    )


def _build_unet_raw(model_cfg, n_channels, n_input_features, n_targets):
    return AutoEncoder1DRaw(
        n_channels_in=n_channels,
        n_channels_out=n_targets,
        channels=model_cfg["channels"],
        kernel_sizes=model_cfg["kernel_sizes"],
        strides=model_cfg["strides"],
        dilation=model_cfg["dilation"],
    )


def _build_tcn(model_cfg, n_channels, n_input_features, n_targets):
    return TCN(
        n_channels_in=n_channels * n_input_features,
        n_channels_out=n_targets,
        hidden_channels=model_cfg["hidden_channels"],
        kernel_size=model_cfg["kernel_size"],
        n_blocks=model_cfg["n_blocks"],
        dilation_base=model_cfg.get("dilation_base", 2),
        dropout=model_cfg.get("dropout", 0.1),
    )


MODEL_REGISTRY = {
    "unet_lomtev": _build_unet_lomtev,
    "unet_raw": _build_unet_raw,
    "tcn": _build_tcn,
}


def build_model(config, dataset_info):
    """Construct a model from config and dataset metadata.

    Parameters
    ----------
    config : dict
        Full YAML config. Must have config["model"]["name"].
    dataset_info : dict
        Return value from build_data(). Must have 'n_channels' and
        'n_input_features'.

    Returns
    -------
    nn.Module
    """
    model_cfg = config["model"]
    name = model_cfg["name"]

    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model {name!r}. Available: {list(MODEL_REGISTRY.keys())}"
        )

    n_channels = dataset_info["n_channels"]
    n_input_features = dataset_info.get("n_input_features", 1)
    n_targets = config["data"]["n_targets"]

    return MODEL_REGISTRY[name](model_cfg, n_channels, n_input_features, n_targets)
