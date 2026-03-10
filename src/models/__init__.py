"""Model module — registry and builder.

Usage:
    from src.models import build_model
    model = build_model(config, dataset_info)
"""

from src.models.unet_lomtev import AutoEncoder1D
from src.models.unet_raw import AutoEncoder1DRaw
from src.models.tcn import TCN
from src.models.transformer import TransformerECoG, MultiscaleTransformerECoG, HybridTransformerECoG
from src.models.nested_unet import NestedUNet
from src.models.dtcnet import DTCNet


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


def _build_transformer(model_cfg, n_channels, n_input_features, n_targets):
    return TransformerECoG(
        n_channels_in=n_channels * n_input_features,
        n_channels_out=n_targets,
        d_model=model_cfg["d_model"],
        n_layers=model_cfg["n_layers"],
        n_heads=model_cfg["n_heads"],
        dim_feedforward=model_cfg["dim_feedforward"],
        spatial_kernel_size=model_cfg.get("spatial_kernel_size", 3),
        spatial_bottleneck_dim=model_cfg.get("spatial_bottleneck_dim", 0),
        ffn_type=model_cfg.get("ffn_type", "gelu"),
        dropout=model_cfg.get("dropout", 0.1),
        eval_window=model_cfg.get("eval_window", 256),
    )


def _build_multiscale_transformer(model_cfg, n_channels, n_input_features, n_targets):
    return MultiscaleTransformerECoG(
        n_channels_in=n_channels * n_input_features,
        n_channels_out=n_targets,
        d_model=model_cfg["d_model"],
        n_layers=model_cfg["n_layers"],
        n_heads=model_cfg["n_heads"],
        dim_feedforward=model_cfg["dim_feedforward"],
        spatial_kernel_size=model_cfg.get("spatial_kernel_size", 1),
        dropout=model_cfg.get("dropout", 0.1),
        downsample_factor=model_cfg.get("downsample_factor", 4),
        eval_window=model_cfg.get("eval_window", 256),
    )


def _build_hybrid_transformer(model_cfg, n_channels, n_input_features, n_targets):
    return HybridTransformerECoG(
        n_channels_in=n_channels * n_input_features,
        n_channels_out=n_targets,
        d_model=model_cfg["d_model"],
        n_layers=model_cfg["n_layers"],
        n_heads=model_cfg["n_heads"],
        dim_feedforward=model_cfg["dim_feedforward"],
        spatial_kernel_size=model_cfg.get("spatial_kernel_size", 1),
        dropout=model_cfg.get("dropout", 0.1),
        eval_window=model_cfg.get("eval_window", 256),
    )


def _build_nested_unet(model_cfg, n_channels, n_input_features, n_targets):
    return NestedUNet(
        n_channels_in=n_channels * n_input_features,
        n_channels_out=n_targets,
        base_ch=model_cfg.get("base_ch", 32),
        kernel_size=model_cfg.get("kernel_size", 3),
    )


def _build_dtcnet(model_cfg, n_channels, n_input_features, n_targets):
    return DTCNet(
        n_channels_in=n_channels * n_input_features,
        n_channels_out=n_targets,
        dropout=model_cfg.get("dropout", 0.1),
    )


MODEL_REGISTRY = {
    "unet_lomtev": _build_unet_lomtev,
    "unet_raw": _build_unet_raw,
    "tcn": _build_tcn,
    "transformer": _build_transformer,
    "multiscale_transformer": _build_multiscale_transformer,
    "hybrid_transformer": _build_hybrid_transformer,
    "nested_unet": _build_nested_unet,
    "dtcnet": _build_dtcnet,
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
