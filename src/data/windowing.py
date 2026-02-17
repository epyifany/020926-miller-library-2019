"""Windowing utilities for continuous ECoG signals.

Provides functions to slice continuous recordings into overlapping windows
and to reconstruct continuous signals from overlapping window predictions
(by averaging in overlap regions).
"""

import numpy as np


def create_windows(signal, window_size, hop_size):
    """Slice a continuous signal into overlapping windows.

    Parameters
    ----------
    signal : np.ndarray, shape (n_samples, features)
        Continuous signal (e.g., ECoG channels or finger flexion).
    window_size : int
        Number of samples per window.
    hop_size : int
        Number of samples between window starts.

    Returns
    -------
    np.ndarray, shape (n_windows, window_size, features)
    """
    n_samples, n_features = signal.shape
    n_windows = (n_samples - window_size) // hop_size + 1
    if n_windows <= 0:
        raise ValueError(
            f"Signal length {n_samples} too short for window_size={window_size}"
        )

    windows = np.empty((n_windows, window_size, n_features), dtype=signal.dtype)
    for i in range(n_windows):
        start = i * hop_size
        windows[i] = signal[start : start + window_size]
    return windows


def reconstruct_from_windows(windows, hop_size, n_samples):
    """Reconstruct a continuous signal by averaging overlapping windows.

    This is the inverse of create_windows — overlapping regions are averaged.
    Needed for evaluation: reconstruct continuous predictions from windowed
    model outputs, then compute Pearson r on the full sequence.

    Parameters
    ----------
    windows : np.ndarray, shape (n_windows, window_size, features)
        Windowed signal (e.g., model predictions).
    hop_size : int
        Hop size used when creating windows.
    n_samples : int
        Length of the original continuous signal.

    Returns
    -------
    np.ndarray, shape (n_samples, features)
        Reconstructed continuous signal (overlap regions averaged).
    """
    n_windows, window_size, n_features = windows.shape

    output = np.zeros((n_samples, n_features), dtype=np.float64)
    counts = np.zeros(n_samples, dtype=np.float64)

    for i in range(n_windows):
        start = i * hop_size
        end = start + window_size
        output[start:end] += windows[i]
        counts[start:end] += 1.0

    # Avoid division by zero for samples not covered by any window
    mask = counts > 0
    output[mask] /= counts[mask, np.newaxis]

    return output
