"""Per-trial, unsupervised EEG feature representations for the simple-MLP.

The baseline model flattens the raw windowed voltages (17 x T). This module adds
BAND-POWER features as an alternative/complementary EEG representation, to test
whether an oscillatory view of the posterior (visual-cortex) EEG carries retrieval
signal that the raw time-domain window misses.

Everything here is computed PER TRIAL and is fully unsupervised (no labels, no
cross-trial fitting), so it is compliant with the challenge rules and introduces
no leakage.

Montage note: all 17 electrodes are parietal / parieto-occipital / occipital
(Pz P3 P7 O1 Oz O2 P4 P8 P1 P5 PO7 PO3 POz PO4 PO8 P6 P2). Occipito-parietal
ALPHA (8-13 Hz) is the classic marker of visual engagement, so band-power is a
physiologically motivated representation for this task.

Caveat: the ERP window is short (~0.55 s at 100 Hz = 55 samples), so band-power
here is a coarse estimate. We compute it on exactly the SAME windowed EEG the raw
model sees, to keep the comparison apples-to-apples.
"""

from typing import List, Tuple

import numpy as np

# Default bands (Hz). Gamma is excluded: the data is band-passed to 0.1-40 Hz and
# decimated to 100 Hz, so there is little usable signal above ~40 Hz.
DEFAULT_BANDS: List[Tuple[float, float]] = [
    (4.0, 8.0),    # theta
    (8.0, 13.0),   # alpha  (occipito-parietal alpha = visual engagement)
    (13.0, 30.0),  # beta
]

# "Rich" 5-band set used by the 221-feature representation. Adds delta (slow
# evoked-potential envelope, available since the data is band-passed from 0.1 Hz)
# and splits beta into low/high. 17 channels x 5 bands = 85 spectral features.
RICH_BANDS: List[Tuple[float, float]] = [
    (0.1, 4.0),    # delta
    (4.0, 8.0),    # theta
    (8.0, 13.0),   # alpha
    (13.0, 20.0),  # low beta
    (20.0, 30.0),  # high beta
]


def band_power(eeg: np.ndarray, sfreq: float = 100.0,
               bands: List[Tuple[float, float]] = None) -> np.ndarray:
    """Per-channel log band-power for each trial, via the real FFT power spectrum.

    Parameters
    ----------
    eeg : (N, C, T) ndarray
        Windowed EEG (e.g. the 50-600 ms ERP window at 100 Hz).
    sfreq : float
        Sampling rate of `eeg` in Hz (100 after decimation).
    bands : list of (lo, hi) Hz, optional
        Frequency bands. Defaults to theta/alpha/beta.

    Returns
    -------
    feats : (N, C * n_bands) ndarray
        log(1 + mean power) per channel per band, flattened to one vector per trial.
        Channel-major order: [c0_b0, c0_b1, ..., c0_bk, c1_b0, ...].
    """
    if bands is None:
        bands = DEFAULT_BANDS

    N, C, T = eeg.shape
    # Detrend per trial/channel (remove DC/mean) so band-power reflects oscillations.
    x = eeg - eeg.mean(axis=-1, keepdims=True)

    # Real FFT power spectrum.
    freqs = np.fft.rfftfreq(T, d=1.0 / sfreq)        # (F,)
    spec = np.abs(np.fft.rfft(x, axis=-1)) ** 2      # (N, C, F)

    out = np.empty((N, C, len(bands)), dtype=np.float64)
    for bi, (lo, hi) in enumerate(bands):
        mask = (freqs >= lo) & (freqs < hi)
        if not mask.any():
            # Band outside the resolvable range for this window length → zero.
            out[:, :, bi] = 0.0
        else:
            out[:, :, bi] = spec[:, :, mask].mean(axis=-1)

    out = np.log1p(out)                               # compress dynamic range
    return out.reshape(N, C * len(bands)).astype(np.float32)


def connectivity(eeg: np.ndarray) -> np.ndarray:
    """Per-trial channel-pair connectivity = upper-triangle of the time-domain
    Pearson correlation matrix across the 17 channels.

    For C=17 channels this gives C*(C-1)/2 = 136 features per trial — how strongly
    each electrode pair co-activates over the window. This is a spatial/functional
    view distinct from per-channel power, and well suited to the dense posterior
    montage. Uses correlation (not coherence) to avoid the frequency-resolution
    problem of a short (~0.55 s) window.

    Parameters
    ----------
    eeg : (N, C, T) ndarray

    Returns
    -------
    feats : (N, C*(C-1)/2) ndarray float32
    """
    N, C, T = eeg.shape
    # Per-trial, per-channel z-score over time so correlation is well-defined.
    x = eeg - eeg.mean(axis=-1, keepdims=True)
    sd = x.std(axis=-1, keepdims=True)
    x = x / (sd + 1e-8)
    # Correlation matrix per trial: (N, C, C) = (x @ x^T) / T
    corr = np.einsum("nct,ndt->ncd", x, x) / T
    iu = np.triu_indices(C, k=1)
    return corr[:, iu[0], iu[1]].astype(np.float32)  # (N, 136 for C=17)


def build_eeg_representation(eeg: np.ndarray, repr_mode: str,
                             sfreq: float = 100.0,
                             bands: List[Tuple[float, float]] = None) -> np.ndarray:
    """Return the flat per-trial EEG feature vector for the chosen representation.

    repr_mode:
      - "raw"        : flattened windowed voltages (C * T).            [baseline]
      - "bandpower"  : per-channel log band-power    (C * 3 bands).
      - "both"       : concat of raw and 3-band band-power.
      - "rich"       : 5-band power (85) + channel-pair connectivity (136) = 221,
                       the classic hand-crafted EEG feature vector. No raw voltages.
      - "both_rich"  : raw voltages + the 221-dim rich vector.

    Returns (N, D) float32, where D depends on repr_mode.
    """
    N = eeg.shape[0]
    if repr_mode == "raw":
        return eeg.reshape(N, -1).astype(np.float32)
    if repr_mode == "bandpower":
        return band_power(eeg, sfreq=sfreq, bands=bands)
    if repr_mode == "both":
        raw = eeg.reshape(N, -1).astype(np.float32)
        bp = band_power(eeg, sfreq=sfreq, bands=bands)
        return np.concatenate([raw, bp], axis=1)
    if repr_mode in ("rich", "both_rich"):
        bp5 = band_power(eeg, sfreq=sfreq, bands=RICH_BANDS)  # (N, 85)
        conn = connectivity(eeg)                              # (N, 136)
        rich = np.concatenate([bp5, conn], axis=1)            # (N, 221)
        if repr_mode == "rich":
            return rich
        raw = eeg.reshape(N, -1).astype(np.float32)
        return np.concatenate([raw, rich], axis=1)
    raise ValueError(f"Unknown repr_mode: {repr_mode}")
