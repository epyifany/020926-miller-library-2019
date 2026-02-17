"""Preprocessing functions for ECoG signals.

Adapted from the FingerFlex pipeline (prepare_data.ipynb) which uses MNE
for filtering. We reuse MNE's filter_data and notch_filter since they are
proven to work on this data.

Our data is (time, channels). MNE expects (channels, time), so we transpose
in/out inside each function — the caller always works with (time, channels).

Pipeline order: normalize (z-score + CAR) -> bandpass -> notch -> scale targets
"""

import numpy as np
import mne


def zscore(data: np.ndarray, mean=None, std=None):
    """Z-score normalization per feature (column).

    If mean/std are not provided, compute them from data (use for train split).
    If provided, apply them directly (use for val/test splits with train stats).

    Parameters
    ----------
    data : np.ndarray, shape (time, features)
    mean : np.ndarray, shape (features,), optional
    std : np.ndarray, shape (features,), optional

    Returns
    -------
    normalized : np.ndarray, shape (time, features)
    stats : dict with 'mean' and 'std', each shape (features,)
    """
    if mean is None:
        mean = data.mean(axis=0)
    if std is None:
        std = data.std(axis=0)
    std = std.copy()
    std[std == 0] = 1.0

    return (data - mean) / std, {"mean": mean, "std": std}


def apply_car(data: np.ndarray):
    """Common Average Reference — subtract median across channels per timestep.

    Parameters
    ----------
    data : np.ndarray, shape (time, channels)

    Returns
    -------
    np.ndarray, shape (time, channels)
    """
    common_avg = np.median(data, axis=1, keepdims=True)
    return data - common_avg


def minmax_scale(data: np.ndarray, dmin=None, dmax=None):
    """MinMax scaling to [0, 1] per feature (column).

    Matches FingerFlex (Lomtev, 2023) target scaling: sklearn MinMaxScaler.
    If dmin/dmax not provided, compute from data (train). If provided, apply (val/test).

    Parameters
    ----------
    data : np.ndarray, shape (time, features)
    dmin : np.ndarray, shape (features,), optional
    dmax : np.ndarray, shape (features,), optional

    Returns
    -------
    scaled : np.ndarray, shape (time, features)
    stats : dict with 'min' and 'max', each shape (features,)
    """
    if dmin is None:
        dmin = data.min(axis=0)
    if dmax is None:
        dmax = data.max(axis=0)
    drange = dmax - dmin
    drange[drange == 0] = 1.0

    return (data - dmin) / drange, {"min": dmin, "max": dmax}


def normalize_and_car(ecog: np.ndarray):
    """Z-score per channel, then subtract common average (median across channels).

    Matches FingerFlex normalize(): z-score each channel independently,
    then subtract the median across channels at each time point.

    Parameters
    ----------
    ecog : np.ndarray, shape (time, channels)

    Returns
    -------
    out : np.ndarray, shape (time, channels)
    stats : dict with 'mean' and 'std', each shape (channels,), for inverse transform
    """
    mean = ecog.mean(axis=0)  # per-channel mean
    std = ecog.std(axis=0)
    std[std == 0] = 1.0  # guard against constant channels

    out = (ecog - mean) / std

    # CAR via median (matches FingerFlex)
    common_avg = np.median(out, axis=1, keepdims=True)
    out = out - common_avg

    return out, {"mean": mean, "std": std}


def filter_ecog(ecog: np.ndarray, sr: int = 1000,
                l_freq: float = 1.0, h_freq: float = 200.0,
                powerline_freq: float = 60.0):
    """Bandpass filter + notch filter using MNE (same approach as FingerFlex).

    FingerFlex uses l_freq=40, h_freq=300, powerline=50 (European).
    We use l_freq=1, h_freq=200, powerline=60 (US/Miller data) by default.
    These are configurable.

    Parameters
    ----------
    ecog : np.ndarray, shape (time, channels)
    sr : int — sampling rate
    l_freq : float — lower bandpass edge (Hz)
    h_freq : float — upper bandpass edge (Hz)
    powerline_freq : float — power line frequency (60 Hz in US)

    Returns
    -------
    np.ndarray, shape (time, channels) — filtered signal
    """
    # MNE expects (channels, time)
    data = ecog.T.astype(np.float64)

    # Bandpass (same as FingerFlex's mne.filter.filter_data call)
    data = mne.filter.filter_data(data, sr, l_freq=l_freq, h_freq=h_freq,
                                  verbose=False)

    # Notch filter at powerline harmonics (FingerFlex: all harmonics up to Nyquist)
    harmonics = np.arange(powerline_freq, sr / 2, powerline_freq)
    if len(harmonics) > 0:
        data = mne.filter.notch_filter(data, sr, freqs=harmonics, verbose=False)

    return data.T  # back to (time, channels)


def normalize_targets(flex: np.ndarray):
    """Z-score finger flexion targets per finger.

    Parameters
    ----------
    flex : np.ndarray, shape (time, n_fingers)

    Returns
    -------
    out : np.ndarray, shape (time, n_fingers)
    stats : dict with 'mean' and 'std', each shape (n_fingers,)
    """
    mean = flex.mean(axis=0)
    std = flex.std(axis=0)
    std[std == 0] = 1.0
    return (flex - mean) / std, {"mean": mean, "std": std}
