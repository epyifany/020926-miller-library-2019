"""Lomtev (2023) spectrogram dataset and pipeline.

Supports both Miller Library (9 patients) and BCI Competition IV (3 subjects).
Produces wavelet spectrogram inputs for AutoEncoder1D.

Miller adaptation: variable channel counts (38-64), finger data at 1000 Hz
(downsample instead of interpolate).

BCI-IV: finger data at 25 Hz (interpolate to 100 Hz), pre-split train/test,
50 Hz powerline.
"""

import os

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.load_fingerflex import load_fingerflex
from src.data.load_bci4 import load_bci4
from src.data.preprocessing import normalize_and_car, filter_ecog
from src.data.preprocessing_lomtev import (
    compute_spectrograms, downsample_spectrograms,
    crop_for_time_delay, robust_scale_spectrograms, minmax_scale_flex,
    interpolate_fingerflex,
)
from src.data.splits import temporal_split


class LomtevDataset(Dataset):
    """Stride-1 windowed dataset on wavelet spectrograms.

    Matches the original EcogFingerflexDataset from the FingerFlex repo.

    Parameters
    ----------
    spec : np.ndarray, shape (channels, wavelets, time)
    flex : np.ndarray, shape (fingers, time)
    sample_len : int — window size in samples (default 256 @ 100 Hz = 2.56s)
    stride : int — hop between windows (default 1)
    """

    def __init__(self, spec, flex, sample_len=256, stride=1):
        self.spec = torch.from_numpy(spec.astype(np.float32))
        self.flex = torch.from_numpy(flex.astype(np.float32))
        self.sample_len = sample_len
        self.stride = stride

        duration = self.spec.shape[-1]
        self.n_windows = (duration - sample_len) // stride
        if self.n_windows <= 0:
            raise ValueError(
                f"Duration {duration} too short for sample_len={sample_len}"
            )

    def __len__(self):
        return self.n_windows

    def __getitem__(self, idx):
        start = idx * self.stride
        end = start + self.sample_len
        return self.spec[..., start:end], self.flex[..., start:end]

    def get_full_signal(self):
        """Return full continuous signal for full-signal evaluation.

        Returns
        -------
        input : torch.Tensor, shape (channels, wavelets, time)
        target : np.ndarray, shape (n_targets, time)
        """
        return self.spec, self.flex.numpy()


def build_lomtev_datasets(patient_id, config):
    """Full Lomtev spectrogram pipeline for one Miller patient.

    Pipeline (matches Lomtev, 2023):
        1. Load raw data
        2. Z-score + CAR (median) — on full signal, pre-filter
        3. Bandpass 40-300 Hz + notch 60 Hz
        4. Morlet wavelets → (channels, 40, time)
        5. Downsample to 100 Hz
        6. Downsample flex to 100 Hz
        7. Temporal split 70/15/15
        8. RobustScaler on train spectrograms, apply to all
        9. MinMaxScaler on train flex, apply to all
       10. Time delay crop (200ms)
       11. Wrap in LomtevDataset (stride=1, sample_len=256)

    Parameters
    ----------
    patient_id : str
    config : dict

    Returns
    -------
    dict with 'train', 'val', 'test' LomtevDataset, plus metadata.
    """
    data_cfg = config["data"]
    prep_cfg = config.get("preprocessing_lomtev", config.get("preprocessing", {}))
    win_cfg = config.get("windowing_lomtev", config.get("windowing", {}))
    split_cfg = config["split"]

    # 1. Load raw data
    base_dir = data_cfg.get("data_dir", "data/fingerflex")
    mat_dir = os.path.join(base_dir, "data")
    raw = load_fingerflex(patient_id, data_dir=mat_dir)
    ecog = raw["ecog"]  # (time, channels)
    flex = raw["flex"]  # (time, 5)
    sr = raw["sr"]      # 1000

    n_channels = ecog.shape[1]

    # 2. Z-score + CAR on full signal (Lomtev does this pre-filter)
    ecog_norm, ecog_stats = normalize_and_car(ecog)
    # Back to (channels, time) for spectrogram pipeline
    ecog_ct = ecog_norm.T  # (channels, time)

    # 3. Bandpass + notch (on channels-first data)
    l_freq = prep_cfg.get("bandpass_lo", 40.0)
    h_freq = prep_cfg.get("bandpass_hi", 300.0)
    powerline = prep_cfg.get("notch_freqs", [60])[0]

    # filter_ecog expects (time, channels), so transpose in/out
    ecog_filtered = filter_ecog(
        ecog_ct.T, sr=sr, l_freq=l_freq, h_freq=h_freq,
        powerline_freq=powerline
    ).T  # back to (channels, time)

    # 4. Morlet wavelets
    n_wavelets = prep_cfg.get("n_wavelets", 40)
    spec = compute_spectrograms(
        ecog_filtered, sr=sr, l_freq=l_freq, h_freq=h_freq,
        n_wavelets=n_wavelets, n_jobs=prep_cfg.get("n_jobs", 1)
    )  # (channels, wavelets, time @ 1000 Hz)

    # 5. Downsample spectrograms to 100 Hz
    ds_fs = prep_cfg.get("downsample_fs", 100)
    spec = downsample_spectrograms(spec, cur_fs=sr, new_fs=ds_fs)
    # (channels, wavelets, time @ 100 Hz)

    # 6. Downsample flex to 100 Hz (Miller data is at 1000 Hz)
    flex_ct = flex.T  # (5, time @ 1000 Hz)
    ds_ratio = sr // ds_fs
    flex_ds = flex_ct[:, ::ds_ratio]  # (5, time @ 100 Hz)

    # Ensure same length
    min_len = min(spec.shape[-1], flex_ds.shape[-1])
    spec = spec[..., :min_len]
    flex_ds = flex_ds[..., :min_len]

    # 7. Temporal split (on the 100 Hz time axis)
    n_time = spec.shape[-1]
    splits = temporal_split(
        n_time,
        train_frac=split_cfg.get("train_frac", 0.70),
        val_frac=split_cfg.get("val_frac", 0.15),
        test_frac=split_cfg.get("test_frac", 0.15),
    )

    def _slice(arr, s, e):
        return arr[..., s:e]

    train_spec = _slice(spec, *splits["train"])
    val_spec = _slice(spec, *splits["val"])
    test_spec = _slice(spec, *splits["test"])

    train_flex = _slice(flex_ds, *splits["train"])
    val_flex = _slice(flex_ds, *splits["val"])
    test_flex = _slice(flex_ds, *splits["test"])

    # 8. RobustScaler on spectrograms (fit on train)
    train_spec, val_spec, test_spec, spec_scaler = robust_scale_spectrograms(
        train_spec, val_spec, test_spec
    )

    # 9. MinMaxScaler on flex (fit on train)
    train_flex, val_flex, test_flex, flex_scaler = minmax_scale_flex(
        train_flex, val_flex, test_flex
    )

    # 10. Time delay crop
    delay_sec = prep_cfg.get("time_delay_sec", 0.2)
    train_flex, train_spec = crop_for_time_delay(train_flex, train_spec, delay_sec, ds_fs)
    val_flex, val_spec = crop_for_time_delay(val_flex, val_spec, delay_sec, ds_fs)
    test_flex, test_spec = crop_for_time_delay(test_flex, test_spec, delay_sec, ds_fs)

    # 11. Create datasets
    sample_len = win_cfg.get("sample_len", 256)
    stride = win_cfg.get("stride", 1)

    datasets = {
        "train": LomtevDataset(train_spec, train_flex, sample_len, stride),
        "val": LomtevDataset(val_spec, val_flex, sample_len, stride),
        "test": LomtevDataset(test_spec, test_flex, sample_len, stride),
        "stats": {
            "ecog": ecog_stats,
            "spec_scaler": spec_scaler,
            "flex_scaler": flex_scaler,
        },
        "split_indices": splits,
        "n_channels": n_channels,
        "n_input_features": n_wavelets,
        "n_wavelets": n_wavelets,
        "downsample_fs": ds_fs,
        "pipeline_mode": "lomtev_spectrogram",
    }
    return datasets


def build_bci4_lomtev_datasets(subject_id, config):
    """Lomtev spectrogram pipeline for BCI Competition IV data.

    Key differences from build_lomtev_datasets (Miller):
        - Pre-split: train and test are separate recordings (no temporal_split)
        - Validation is carved from end of training data
        - Finger data at 25 Hz → cubic interpolation to 100 Hz
        - 50 Hz powerline (European)

    Pipeline:
        1. Load raw data (pre-split train/test)
        2. Z-score + CAR on train and test independently
        3. Bandpass 40-300 Hz + notch 50 Hz
        4. Morlet wavelets → (channels, 40, time)
        5. Downsample spectrograms to 100 Hz
        6. Interpolate finger data 25 Hz → 100 Hz (cubic)
        7. Carve validation from end of train (val_frac)
        8. RobustScaler on train spectrograms, apply to all
        9. MinMaxScaler on train flex, apply to all
       10. Time delay crop (200ms)
       11. Wrap in LomtevDataset

    Parameters
    ----------
    subject_id : int — 1, 2, or 3
    config : dict

    Returns
    -------
    dict with 'train', 'val', 'test' LomtevDataset, plus metadata.
    """
    data_cfg = config["data"]
    prep_cfg = config.get("preprocessing_lomtev", config.get("preprocessing", {}))
    win_cfg = config.get("windowing_lomtev", config.get("windowing", {}))
    split_cfg = config["split"]

    # 1. Load raw data (pre-split)
    data_dir = data_cfg.get("bci4_data_dir", "data/bci4")
    raw = load_bci4(subject_id, data_dir=data_dir)
    sr = raw["sr"]  # 1000
    n_channels = raw["n_channels"]

    # 2. Z-score + CAR on each split independently
    train_ecog_norm, train_stats = normalize_and_car(raw["train_ecog"])
    test_ecog_norm, _ = normalize_and_car(raw["test_ecog"])

    # To (channels, time)
    train_ecog_ct = train_ecog_norm.T
    test_ecog_ct = test_ecog_norm.T

    # 3. Bandpass + notch
    l_freq = prep_cfg.get("bandpass_lo", 40.0)
    h_freq = prep_cfg.get("bandpass_hi", 300.0)
    powerline = prep_cfg.get("notch_freqs", [50])[0]  # 50 Hz for BCI-IV

    train_ecog_filt = filter_ecog(
        train_ecog_ct.T, sr=sr, l_freq=l_freq, h_freq=h_freq,
        powerline_freq=powerline
    ).T
    test_ecog_filt = filter_ecog(
        test_ecog_ct.T, sr=sr, l_freq=l_freq, h_freq=h_freq,
        powerline_freq=powerline
    ).T

    # 4. Morlet wavelets
    n_wavelets = prep_cfg.get("n_wavelets", 40)
    n_jobs = prep_cfg.get("n_jobs", 1)
    train_spec = compute_spectrograms(
        train_ecog_filt, sr=sr, l_freq=l_freq, h_freq=h_freq,
        n_wavelets=n_wavelets, n_jobs=n_jobs
    )
    test_spec = compute_spectrograms(
        test_ecog_filt, sr=sr, l_freq=l_freq, h_freq=h_freq,
        n_wavelets=n_wavelets, n_jobs=n_jobs
    )

    # 5. Downsample spectrograms to 100 Hz
    ds_fs = prep_cfg.get("downsample_fs", 100)
    train_spec = downsample_spectrograms(train_spec, cur_fs=sr, new_fs=ds_fs)
    test_spec = downsample_spectrograms(test_spec, cur_fs=sr, new_fs=ds_fs)

    # 6. Interpolate finger data 25 Hz → 100 Hz (cubic)
    train_flex = interpolate_fingerflex(
        raw["train_flex"].T, cur_fs=sr, true_fs=raw["flex_sr"], needed_hz=ds_fs
    )  # (5, time @ 100 Hz)
    test_flex = interpolate_fingerflex(
        raw["test_flex"].T, cur_fs=sr, true_fs=raw["flex_sr"], needed_hz=ds_fs
    )  # (5, time @ 100 Hz)

    # Ensure same length
    train_min = min(train_spec.shape[-1], train_flex.shape[-1])
    train_spec = train_spec[..., :train_min]
    train_flex = train_flex[..., :train_min]

    test_min = min(test_spec.shape[-1], test_flex.shape[-1])
    test_spec = test_spec[..., :test_min]
    test_flex = test_flex[..., :test_min]

    # 7. Carve validation from end of training data
    val_frac = split_cfg.get("val_frac", 0.15)
    n_train_total = train_spec.shape[-1]
    n_val = int(n_train_total * val_frac)
    n_train = n_train_total - n_val

    val_spec = train_spec[..., n_train:]
    val_flex = train_flex[..., n_train:]
    train_spec = train_spec[..., :n_train]
    train_flex = train_flex[..., :n_train]

    # 8. RobustScaler on spectrograms (fit on train)
    train_spec, val_spec, test_spec, spec_scaler = robust_scale_spectrograms(
        train_spec, val_spec, test_spec
    )

    # 9. MinMaxScaler on flex (fit on train)
    train_flex, val_flex, test_flex, flex_scaler = minmax_scale_flex(
        train_flex, val_flex, test_flex
    )

    # 10. Time delay crop
    delay_sec = prep_cfg.get("time_delay_sec", 0.2)
    train_flex, train_spec = crop_for_time_delay(train_flex, train_spec, delay_sec, ds_fs)
    val_flex, val_spec = crop_for_time_delay(val_flex, val_spec, delay_sec, ds_fs)
    test_flex, test_spec = crop_for_time_delay(test_flex, test_spec, delay_sec, ds_fs)

    # 11. Create datasets
    sample_len = win_cfg.get("sample_len", 256)
    stride = win_cfg.get("stride", 1)

    datasets = {
        "train": LomtevDataset(train_spec, train_flex, sample_len, stride),
        "val": LomtevDataset(val_spec, val_flex, sample_len, stride),
        "test": LomtevDataset(test_spec, test_flex, sample_len, stride),
        "stats": {
            "ecog": train_stats,
            "spec_scaler": spec_scaler,
            "flex_scaler": flex_scaler,
        },
        "n_channels": n_channels,
        "n_input_features": n_wavelets,
        "n_wavelets": n_wavelets,
        "downsample_fs": ds_fs,
        "pipeline_mode": "lomtev_spectrogram_bci4",
    }
    return datasets
