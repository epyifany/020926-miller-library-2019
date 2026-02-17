"""Temporal train/val/test splitting for time-series data.

Critical: no shuffling. Splits are contiguous blocks to prevent
future information from leaking into training.
"""


def temporal_split(n_samples, train_frac=0.70, val_frac=0.15, test_frac=0.15):
    """Compute contiguous train/val/test split boundaries.

    Parameters
    ----------
    n_samples : int
        Total number of time samples.
    train_frac : float
        Fraction of data for training (default 0.70).
    val_frac : float
        Fraction of data for validation (default 0.15).
    test_frac : float
        Fraction of data for testing (default 0.15).

    Returns
    -------
    dict with keys 'train', 'val', 'test', each mapping to (start, end) tuples.
        Indices are half-open: data[start:end].
    """
    total = train_frac + val_frac + test_frac
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"Fractions must sum to 1.0, got {total:.6f}"
        )

    train_end = int(n_samples * train_frac)
    val_end = int(n_samples * (train_frac + val_frac))

    return {
        "train": (0, train_end),
        "val": (train_end, val_end),
        "test": (val_end, n_samples),
    }
