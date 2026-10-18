"""EEG data loading and preprocessing.

Raw EEG:    (N, 17 channels, 1500 timepoints) at 500 Hz over 3 s.
Decimated:  (N, 17,          300 timepoints)  at 100 Hz.
EEG_WINDOW = slice(5, 60)  →  50–600 ms post-stimulus onset.
"""

import numpy as np
from scipy.signal import decimate

# ── Sampling constants ──────────────────────────────────────────────────────
SFREQ     = 500          # Hz  — original recording rate
DEC       = 5            # decimation factor  →  100 Hz
SFREQ_DEC = SFREQ // DEC

N_CHANNELS   = 17
N_TIME_RAW   = 1500      # samples @ 500 Hz
N_TIME_DEC   = 300       # samples @ 100 Hz

# 50–600 ms post-onset is where visual ERP signal peaks (P1/N1 and N170/P2).
EEG_WINDOW = slice(5, 60)


# ── Loaders ─────────────────────────────────────────────────────────────────

def load_train_eeg(npz_path: str):
    """Load and sort training EEG from a subject's .npz file.

    Returns
    -------
    ids : (N,) int ndarray   — stimulus IDs (1-indexed)
    eeg : (N, 17, 1500) float ndarray — raw EEG at 500 Hz
    """
    data = np.load(npz_path)
    ids  = np.array(sorted(data.files, key=int), dtype=int)
    eeg  = np.stack([data[str(i)] for i in ids])
    return ids, eeg


def load_test_eeg(npy_path: str) -> np.ndarray:
    """Load held-out test EEG from .npy.

    Returns
    -------
    eeg : (102, 17, 1500) float ndarray — raw EEG at 500 Hz, unsorted order
    """
    return np.load(npy_path)


# ── Preprocessing ────────────────────────────────────────────────────────────

def decimate_eeg(eeg: np.ndarray) -> np.ndarray:
    """Downsample EEG from 500 Hz to 100 Hz with anti-alias FIR filter.

    Parameters
    ----------
    eeg : (..., 1500) ndarray

    Returns
    -------
    (..., 300) ndarray
    """
    return decimate(eeg, DEC, axis=-1, ftype='fir', zero_phase=True)


def zscore_per_subject(eeg: np.ndarray) -> np.ndarray:
    """Z-score EEG across trials per (channel, timepoint).

    Removes subject-level amplitude offsets caused by skull thickness,
    electrode impedance, and recording noise — without using label info.

    Parameters
    ----------
    eeg : (N, C, T) ndarray

    Returns
    -------
    (N, C, T) ndarray  — zero mean, unit std across the N trial dimension.
    """
    mu    = eeg.mean(axis=0, keepdims=True)
    sigma = eeg.std(axis=0,  keepdims=True).clip(min=1e-8)
    return (eeg - mu) / sigma


def normalize_eeg_trials(eeg: np.ndarray) -> np.ndarray:
    """Trial-wise z-score normalization (matches organizer baseline correlation).

    Standardizes each trial's (C, T) slice individually: zero mean and unit std
    across its channels and timepoints. Preserves cross-trial ERP amplitude differences
    without subtracting common stimulus-evoked potentials.

    Parameters
    ----------
    eeg : (N, C, T) ndarray

    Returns
    -------
    (N, C, T) ndarray with mean=0 and std=1 per trial.
    """
    mu = eeg.mean(axis=(-2, -1), keepdims=True)
    sigma = eeg.std(axis=(-2, -1), keepdims=True).clip(min=1e-8)
    return (eeg - mu) / sigma

