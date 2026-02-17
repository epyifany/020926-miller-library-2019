"""Lomtev (2023) preprocessing functions for the spectrogram pipeline.

Ported from ../FingerFlex/prepare_data.ipynb.
These are used by build_lomtev_datasets() to produce wavelet spectrogram
inputs for the AutoEncoder1D model.

All functions operate on (channels, time) or (channels, freqs, time) arrays
to match the original Lomtev convention (channel-first).

Key adaptation for Miller data: finger flexion is already at 1000 Hz
(not 25 Hz like BCI-IV), so we just downsample instead of interpolate.
"""

import numpy as np
import mne
import scipy.interpolate
from sklearn.preprocessing import RobustScaler, MinMaxScaler


# ── Finger data interpolation (BCI-IV) ───────────────────────────────

def interpolate_fingerflex(finger_flex, cur_fs=1000, true_fs=25, needed_hz=100,
                           interp_type='cubic'):
    """Interpolate finger flexion from native rate to target rate.

    Ported from FingerFlex/prepare_data.ipynb. BCI-IV finger data is recorded
    at 25 Hz but stored upsampled to 1000 Hz (zero-order hold). This function
    extracts the true 25 Hz samples, then cubic-interpolates to the target rate.

    Parameters
    ----------
    finger_flex : np.ndarray, shape (fingers, time @ cur_fs)
    cur_fs : int — sampling rate of the stored data (1000 Hz)
    true_fs : int — actual recording rate (25 Hz)
    needed_hz : int — target sampling rate (100 Hz)
    interp_type : str — interpolation kind for scipy.interpolate.interp1d

    Returns
    -------
    np.ndarray, shape (fingers, time @ needed_hz)
    """
    # Extract true 25 Hz samples
    downscaling_ratio = cur_fs // true_fs
    flex_true = finger_flex[:, ::downscaling_ratio]
    # Add last value as edge for interpolation
    flex_true = np.concatenate([flex_true, flex_true[:, -1:]], axis=1)

    upscaling_ratio = needed_hz // true_fs
    n_true = flex_true.shape[1]
    ts = np.arange(n_true) * upscaling_ratio
    ts_target = np.arange(n_true * upscaling_ratio)[:-upscaling_ratio]

    result = np.zeros((finger_flex.shape[0], len(ts_target)))
    for i in range(finger_flex.shape[0]):
        f = scipy.interpolate.interp1d(ts, flex_true[i], kind=interp_type)
        result[i] = f(ts_target)
    return result


# ── Spectrogram computation ──────────────────────────────────────────

def compute_spectrograms(ecog, sr=1000, l_freq=40.0, h_freq=300.0,
                         n_wavelets=40, n_jobs=1):
    """Morlet wavelet time-frequency decomposition.

    Parameters
    ----------
    ecog : np.ndarray, shape (channels, time)
    sr : int
    l_freq, h_freq : float — frequency range for log-spaced wavelets
    n_wavelets : int — number of wavelet frequencies
    n_jobs : int — parallel jobs for MNE

    Returns
    -------
    np.ndarray, shape (channels, n_wavelets, time) — wavelet power
    """
    freqs = np.logspace(np.log10(l_freq), np.log10(h_freq), n_wavelets)
    n_ch = ecog.shape[0]
    # MNE expects (n_epochs, n_channels, n_times)
    spec = mne.time_frequency.tfr_array_morlet(
        ecog.reshape(1, n_ch, -1), sfreq=sr,
        freqs=freqs, output='power', verbose=False, n_jobs=n_jobs
    )[0]  # (channels, n_wavelets, time)
    return spec


def downsample_spectrograms(spec, cur_fs=1000, new_fs=100):
    """Stride-based downsampling of spectrograms.

    Parameters
    ----------
    spec : np.ndarray, shape (channels, wavelets, time @ cur_fs)
    cur_fs, new_fs : int

    Returns
    -------
    np.ndarray, shape (channels, wavelets, time @ new_fs)
    """
    ratio = cur_fs // new_fs
    return spec[:, :, ::ratio]


# ── Time delay ───────────────────────────────────────────────────────

def crop_for_time_delay(flex, spec, delay_sec=0.2, fs=100):
    """Shift targets forward to account for neural response latency.

    Parameters
    ----------
    flex : np.ndarray, shape (fingers, time)
    spec : np.ndarray, shape (channels, wavelets, time)
    delay_sec : float — delay in seconds
    fs : int — sampling rate of spec/flex

    Returns
    -------
    flex_cropped : np.ndarray, shape (fingers, time - delay)
    spec_cropped : np.ndarray, shape (channels, wavelets, time - delay)
    """
    delay = int(delay_sec * fs)
    return flex[..., delay:], spec[..., :spec.shape[-1] - delay]


# ── Scaling ──────────────────────────────────────────────────────────

def robust_scale_spectrograms(train_spec, val_spec=None, test_spec=None,
                              quantile_range=(0.1, 0.9)):
    """RobustScaler on flattened spectrograms (fit on train only).

    Matches Lomtev: flatten (channels × wavelets) per timestep, fit
    RobustScaler(unit_variance=True), then reshape back.

    Parameters
    ----------
    train_spec : np.ndarray, shape (channels, wavelets, time)
    val_spec, test_spec : optional, same shape layout

    Returns
    -------
    Tuple of scaled arrays (same shapes), plus the fitted scaler.
    """
    n_ch, n_wv, _ = train_spec.shape
    n_features = n_ch * n_wv

    scaler = RobustScaler(unit_variance=True, quantile_range=quantile_range)

    def _reshape_fit_transform(spec, fit=False):
        # (ch, wv, time) → (time, ch*wv)
        flat = spec.transpose(2, 0, 1).reshape(-1, n_features)
        if fit:
            scaler.fit(flat)
        scaled = scaler.transform(flat)
        return scaled.reshape(-1, n_wv, n_ch).transpose(2, 1, 0)

    train_out = _reshape_fit_transform(train_spec, fit=True)
    val_out = _reshape_fit_transform(val_spec) if val_spec is not None else None
    test_out = _reshape_fit_transform(test_spec) if test_spec is not None else None

    return train_out, val_out, test_out, scaler


def minmax_scale_flex(train_flex, val_flex=None, test_flex=None):
    """MinMaxScaler [0,1] on finger data (fit on train only).

    Parameters
    ----------
    train_flex : np.ndarray, shape (fingers, time)
    val_flex, test_flex : optional

    Returns
    -------
    Tuple of scaled arrays, plus the fitted scaler.
    """
    scaler = MinMaxScaler()
    # (fingers, time) → (time, fingers) for sklearn
    scaler.fit(train_flex.T)
    train_out = scaler.transform(train_flex.T).T

    val_out = scaler.transform(val_flex.T).T if val_flex is not None else None
    test_out = scaler.transform(test_flex.T).T if test_flex is not None else None

    return train_out, val_out, test_out, scaler
