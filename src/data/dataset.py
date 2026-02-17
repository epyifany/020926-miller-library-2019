"""PyTorch Dataset and pipeline orchestrator for fingerflex decoding.

Follows FingerFlex's on-the-fly slicing pattern: stores full continuous
arrays and computes windows in __getitem__. This is memory efficient
(~112 MB/patient vs ~1.1 GB if pre-computing all windows).
"""

import os

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.load_fingerflex import load_fingerflex
from src.data.preprocessing import (
    filter_ecog, zscore, apply_car, minmax_scale, normalize_and_car,
)
from src.data.splits import temporal_split


class FingerFlexDataset(Dataset):
    """On-the-fly windowed dataset for continuous ECoG → finger flexion.

    Parameters
    ----------
    ecog : np.ndarray, shape (time, channels)
        Preprocessed ECoG signal for one split.
    flex : np.ndarray, shape (time, 5)
        Preprocessed finger flexion targets for one split.
    window_size : int
        Samples per window.
    hop_size : int
        Samples between consecutive window starts.
    """

    def __init__(self, ecog, flex, window_size, hop_size):
        self.ecog = torch.from_numpy(ecog.astype(np.float32))
        self.flex = torch.from_numpy(flex.astype(np.float32))
        self.window_size = window_size
        self.hop_size = hop_size

        n_samples = self.ecog.shape[0]
        self.n_windows = (n_samples - window_size) // hop_size + 1
        if self.n_windows <= 0:
            raise ValueError(
                f"Split length {n_samples} too short for "
                f"window_size={window_size}, hop_size={hop_size}"
            )

    def __len__(self):
        return self.n_windows

    def __getitem__(self, idx):
        start = idx * self.hop_size
        end = start + self.window_size
        # Return channels-first: (channels, time), (n_targets, time)
        return self.ecog[start:end].T, self.flex[start:end].T

    def get_full_signal(self):
        """Return full continuous signal for full-signal evaluation.

        Returns
        -------
        input : torch.Tensor, shape (channels, time)
        target : np.ndarray, shape (n_targets, time)
        """
        return self.ecog.T, self.flex.T.numpy()


def _build_ours(ecog, flex, sr, prep_cfg, split_cfg):
    """Pipeline mode 'ours': filter → split → zscore → CAR → zscore targets."""
    # 1. Filter full signal (avoids edge artifacts at split boundaries)
    ecog = filter_ecog(
        ecog, sr=sr,
        l_freq=prep_cfg.get("bandpass_lo", 1.0),
        h_freq=prep_cfg.get("bandpass_hi", 200.0),
        powerline_freq=prep_cfg.get("notch_freqs", [60, 120, 180])[0],
    )

    # 2. Temporal split
    n_samples = ecog.shape[0]
    splits = temporal_split(
        n_samples,
        train_frac=split_cfg.get("train_frac", 0.70),
        val_frac=split_cfg.get("val_frac", 0.15),
        test_frac=split_cfg.get("test_frac", 0.15),
    )

    train_ecog = ecog[splits["train"][0] : splits["train"][1]]
    val_ecog = ecog[splits["val"][0] : splits["val"][1]]
    test_ecog = ecog[splits["test"][0] : splits["test"][1]]

    train_flex = flex[splits["train"][0] : splits["train"][1]]
    val_flex = flex[splits["val"][0] : splits["val"][1]]
    test_flex = flex[splits["test"][0] : splits["test"][1]]

    # 3. Z-score ECoG: fit on train, apply to all
    train_ecog, ecog_stats = zscore(train_ecog)
    val_ecog, _ = zscore(val_ecog, mean=ecog_stats["mean"], std=ecog_stats["std"])
    test_ecog, _ = zscore(test_ecog, mean=ecog_stats["mean"], std=ecog_stats["std"])

    # 4. CAR per-split independently
    train_ecog = apply_car(train_ecog)
    val_ecog = apply_car(val_ecog)
    test_ecog = apply_car(test_ecog)

    # 5. Z-score targets: fit on train, apply to all
    train_flex, flex_stats = zscore(train_flex)
    val_flex, _ = zscore(val_flex, mean=flex_stats["mean"], std=flex_stats["std"])
    test_flex, _ = zscore(test_flex, mean=flex_stats["mean"], std=flex_stats["std"])

    return (splits,
            (train_ecog, val_ecog, test_ecog), ecog_stats,
            (train_flex, val_flex, test_flex), flex_stats)


def _build_lomtev(ecog, flex, sr, prep_cfg, split_cfg):
    """Pipeline mode 'lomtev': zscore+CAR → filter → split → minmax targets.

    Matches FingerFlex (Lomtev, 2023) preprocessing order on raw ECoG.
    """
    # 1. Z-score + CAR on full signal (pre-filter, matching Lomtev)
    ecog, ecog_stats = normalize_and_car(ecog)

    # 2. Filter full signal
    ecog = filter_ecog(
        ecog, sr=sr,
        l_freq=prep_cfg.get("bandpass_lo", 40.0),
        h_freq=prep_cfg.get("bandpass_hi", 300.0),
        powerline_freq=prep_cfg.get("notch_freqs", [60, 120, 180])[0],
    )

    # 3. Temporal split
    n_samples = ecog.shape[0]
    splits = temporal_split(
        n_samples,
        train_frac=split_cfg.get("train_frac", 0.70),
        val_frac=split_cfg.get("val_frac", 0.15),
        test_frac=split_cfg.get("test_frac", 0.15),
    )

    train_ecog = ecog[splits["train"][0] : splits["train"][1]]
    val_ecog = ecog[splits["val"][0] : splits["val"][1]]
    test_ecog = ecog[splits["test"][0] : splits["test"][1]]

    train_flex = flex[splits["train"][0] : splits["train"][1]]
    val_flex = flex[splits["val"][0] : splits["val"][1]]
    test_flex = flex[splits["test"][0] : splits["test"][1]]

    # 4. MinMax [0,1] targets: fit on train, apply to all
    train_flex, flex_stats = minmax_scale(train_flex)
    val_flex, _ = minmax_scale(val_flex, dmin=flex_stats["min"], dmax=flex_stats["max"])
    test_flex, _ = minmax_scale(test_flex, dmin=flex_stats["min"], dmax=flex_stats["max"])

    return (splits,
            (train_ecog, val_ecog, test_ecog), ecog_stats,
            (train_flex, val_flex, test_flex), flex_stats)


def build_datasets(patient_id, config):
    """Full preprocessing pipeline with configurable mode.

    Two pipeline modes controlled by ``preprocessing.pipeline_mode``:

    **'ours'** (default):
        filter → split → zscore(train-fit) → CAR → zscore targets

    **'lomtev'** (faithful to Lomtev, 2023):
        zscore+CAR(full signal) → filter(40-300Hz) → split → minmax targets

    Parameters
    ----------
    patient_id : str
        One of: bp, cc, ht, jc, jp, mv, wc, wm, zt.
    config : dict
        Loaded YAML config (see configs/fingerflex_default.yaml).

    Returns
    -------
    dict with keys:
        'train', 'val', 'test' : FingerFlexDataset
        'stats' : dict with 'ecog' and 'flex' normalization stats
        'split_indices' : dict with 'train', 'val', 'test' (start, end) tuples
        'n_channels' : int
        'pipeline_mode' : str
    """
    data_cfg = config["data"]
    prep_cfg = config["preprocessing"]
    win_cfg = config["windowing"]
    split_cfg = config["split"]
    pipeline_mode = prep_cfg.get("pipeline_mode", "ours")

    # Load raw data
    base_dir = data_cfg.get("data_dir", "data/fingerflex")
    mat_dir = os.path.join(base_dir, "data")
    raw = load_fingerflex(patient_id, data_dir=mat_dir)
    ecog = raw["ecog"]  # (time, channels)
    flex = raw["flex"]  # (time, 5)
    sr = raw["sr"]
    n_channels = ecog.shape[1]

    # Dispatch to pipeline mode
    if pipeline_mode == "ours":
        (splits, ecog_splits, ecog_stats,
         flex_splits, flex_stats) = _build_ours(ecog, flex, sr, prep_cfg, split_cfg)
    elif pipeline_mode == "lomtev":
        (splits, ecog_splits, ecog_stats,
         flex_splits, flex_stats) = _build_lomtev(ecog, flex, sr, prep_cfg, split_cfg)
    else:
        raise ValueError(f"Unknown pipeline_mode: {pipeline_mode!r} (expected 'ours' or 'lomtev')")

    train_ecog, val_ecog, test_ecog = ecog_splits
    train_flex, val_flex, test_flex = flex_splits

    # Create datasets
    window_size = win_cfg.get("window_size", 1000)
    hop_size = win_cfg.get("hop_size", 100)

    datasets = {}
    for split_name, ecog_split, flex_split in [
        ("train", train_ecog, train_flex),
        ("val", val_ecog, val_flex),
        ("test", test_ecog, test_flex),
    ]:
        datasets[split_name] = FingerFlexDataset(
            ecog_split, flex_split, window_size, hop_size
        )

    datasets["stats"] = {"ecog": ecog_stats, "flex": flex_stats}
    datasets["split_indices"] = splits
    datasets["n_channels"] = n_channels
    datasets["n_input_features"] = 1
    datasets["pipeline_mode"] = pipeline_mode

    return datasets
