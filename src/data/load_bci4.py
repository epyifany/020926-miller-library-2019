"""Load BCI Competition IV Dataset 4 (fingerflex).

Data source: https://www.bbci.de/competition/iv/#dataset4
This is the same data used in the original FingerFlex (Lomtev, 2023) paper.

File structure after extraction:
    data/bci4/sub{1,2,3}_comp.mat      — training ECoG + training finger data + test ECoG
    data/bci4/sub{1,2,3}_testlabels.mat — test finger data (ground truth labels)

3 subjects with variable channel counts:
    sub1: 62 channels, sub2: 48 channels, sub3: 64 channels

Key differences from Miller Library fingerflex:
    - 3 subjects (vs 9 in Miller)
    - Variable channels (48–64)
    - Finger data at 25 Hz (vs 1000 Hz in Miller) — already upsampled to 1000 Hz in .mat
    - European: 50 Hz powerline (vs 60 Hz)
    - Pre-split: train (400s) and test (200s) are separate recordings
    - ECoG at 1000 Hz
"""

import os

import numpy as np
import scipy.io

BCI4_SUBJECTS = [1, 2, 3]


def load_bci4(subject_id=1, data_dir="data/bci4"):
    """Load BCI Competition IV Dataset 4.

    Parameters
    ----------
    subject_id : int
        Subject number (1, 2, or 3).
    data_dir : str
        Directory containing sub{N}_comp.mat and sub{N}_testlabels.mat.

    Returns
    -------
    dict with keys:
        'train_ecog' : np.ndarray, shape (time_train, n_channels), float64
        'train_flex' : np.ndarray, shape (time_train, 5), float64
        'test_ecog'  : np.ndarray, shape (time_test, n_channels), float64
        'test_flex'  : np.ndarray, shape (time_test, 5), float64
        'sr'         : int — 1000 Hz (ECoG sampling rate)
        'flex_sr'    : int — 25 Hz (finger data native rate, upsampled to 1000 in .mat)
        'n_channels' : int — varies per subject (62, 48, 64)
    """
    if subject_id not in BCI4_SUBJECTS:
        raise ValueError(f"subject_id must be one of {BCI4_SUBJECTS}, got {subject_id}")

    comp_path = os.path.join(data_dir, f"sub{subject_id}_comp.mat")
    labels_path = os.path.join(data_dir, f"sub{subject_id}_testlabels.mat")

    if not os.path.isfile(comp_path):
        raise FileNotFoundError(
            f"BCI-IV data not found: {comp_path}\n"
            f"Download from https://www.bbci.de/competition/iv/#dataset4"
        )
    if not os.path.isfile(labels_path):
        raise FileNotFoundError(
            f"BCI-IV test labels not found: {labels_path}\n"
            f"Download from https://www.bbci.de/competition/iv/results/ds4/"
        )

    comp = scipy.io.loadmat(comp_path, squeeze_me=False)
    labels = scipy.io.loadmat(labels_path, squeeze_me=False)

    train_ecog = comp["train_data"].astype(np.float64)
    n_channels = train_ecog.shape[1]

    return {
        "train_ecog": train_ecog,                                # (time, n_ch)
        "train_flex": comp["train_dg"].astype(np.float64),       # (time, 5)
        "test_ecog": comp["test_data"].astype(np.float64),       # (time, n_ch)
        "test_flex": labels["test_dg"].astype(np.float64),       # (time, 5)
        "sr": 1000,
        "flex_sr": 25,
        "n_channels": n_channels,
    }
