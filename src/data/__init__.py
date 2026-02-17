"""Data module — dispatches to the correct dataset builder.

Usage:
    from src.data import build_data
    ds = build_data(config)
"""

from src.data.dataset import build_datasets
from src.data.dataset_lomtev import build_lomtev_datasets, build_bci4_lomtev_datasets


SPECTROGRAM_MODELS = {"unet_lomtev", "tcn"}


def build_data(config):
    """Build train/val/test datasets from a config dict.

    Routing logic:
    - task == "bci4"  → Lomtev spectrogram pipeline (BCI-IV)
    - task == "fingerflex" + spectrogram model → Lomtev spectrogram pipeline (Miller)
    - task == "fingerflex" + other models → raw ECoG pipeline

    Parameters
    ----------
    config : dict
        Full YAML config.

    Returns
    -------
    dict with 'train', 'val', 'test' datasets plus metadata.
    """
    task = config["data"]["task"]
    model_name = config.get("model", {}).get("name", "")

    if task == "bci4":
        subject = config["data"]["subject"]
        return build_bci4_lomtev_datasets(subject, config)

    if task == "fingerflex":
        patient = config["data"]["patient"]
        if model_name in SPECTROGRAM_MODELS:
            return build_lomtev_datasets(patient, config)
        else:
            return build_datasets(patient, config)

    raise ValueError(f"Unknown task {task!r}. Expected 'fingerflex' or 'bci4'.")
