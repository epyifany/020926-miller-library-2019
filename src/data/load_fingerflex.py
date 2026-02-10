"""Load fingerflex data from the Miller ECoG Library (2019).

Data structure after unzipping fingerflex.zip:
    fingerflex/data/{patient}/{patient}_fingerflex.mat
    fingerflex/data/{patient}/{patient}_stim.mat

Each _fingerflex.mat contains:
    data : (time, channels) — ECoG recordings (int16 or int32)
    flex : (time, 5) — 5-finger flexion targets (uint16)
    cue  : (time, 1) — cue codes (uint8)
    locs : (channels, 3) — electrode locations
    elec_regions : (channels, 1) — region labels

Each _stim.mat contains:
    stim : (time, 1) — stimulus codes (int16)

All sampled at 1000 Hz.
"""

import os

import numpy as np
import scipy.io

# All 9 fingerflex patients
PATIENTS = ["bp", "cc", "ht", "jc", "jp", "mv", "wc", "wm", "zt"]


def load_fingerflex(patient_id: str, data_dir: str = "data/fingerflex/data"):
    """Load a single patient's fingerflex data.

    Parameters
    ----------
    patient_id : str
        Patient identifier (one of: bp, cc, ht, jc, jp, mv, wc, wm, zt).
    data_dir : str
        Directory containing patient subdirectories with .mat files.

    Returns
    -------
    dict with keys:
        ecog : np.ndarray, shape (time, channels), float64
        flex : np.ndarray, shape (time, 5), float64
        cue  : np.ndarray, shape (time,), float64 — cue codes from fingerflex.mat
        stim : np.ndarray, shape (time,), float64 — stimulus codes from stim.mat
        locs : np.ndarray, shape (channels, 3), float64 — electrode xyz locations
        sr   : int — sampling rate (1000 Hz)
    """
    patient_dir = os.path.join(data_dir, patient_id)
    flex_path = os.path.join(patient_dir, f"{patient_id}_fingerflex.mat")
    stim_path = os.path.join(patient_dir, f"{patient_id}_stim.mat")

    if not os.path.isfile(flex_path):
        raise FileNotFoundError(f"Data file not found: {flex_path}")

    mat = scipy.io.loadmat(flex_path, squeeze_me=False)

    ecog = mat["data"].astype(np.float64)     # (time, channels)
    flex = mat["flex"].astype(np.float64)      # (time, 5)
    cue = mat["cue"].squeeze().astype(np.float64)  # (time,)
    locs = mat["locs"].astype(np.float64)      # (channels, 3)

    # Load stim from separate file if available
    if os.path.isfile(stim_path):
        stim_mat = scipy.io.loadmat(stim_path, squeeze_me=False)
        stim = stim_mat["stim"].squeeze().astype(np.float64)
    else:
        stim = cue  # fallback to cue codes

    return {
        "ecog": ecog,
        "flex": flex,
        "cue": cue,
        "stim": stim,
        "locs": locs,
        "sr": 1000,
    }
