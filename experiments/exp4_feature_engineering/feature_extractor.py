"""Feature extraction and engineering for EEG in Exp4 Tuned MLP pipeline.

Provides:
1. Channel subset slicing (e.g. all 17, core-8, top-12, occipital, etc.)
2. Feature representations:
   - 'raw': standard flattened windowed voltages (C * T)
   - 'bandpower': per-channel log spectral band-power (Delta, Theta, Alpha, Beta)
   - 'both': raw voltages + band-power
   - 'erp_stats': ERP temporal window averages (P100, N170, P300, Late) + peak-to-peak
   - 'rich': 5-band power + channel connectivity
   - 'full': raw voltages + band-power + ERP temporal statistics
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

ALL_CHANNELS: list[str] = [
    "Pz", "P3", "P7", "O1", "Oz", "O2", "P4", "P8",
    "P1", "P5", "PO7", "PO3", "POz", "PO4", "PO8", "P6", "P2"
]
CHANNEL_TO_IDX: dict[str, int] = {name: i for i, name in enumerate(ALL_CHANNELS)}

PREDEFINED_CHANNEL_SETS: dict[str, list[str]] = {
    "all17": ALL_CHANNELS,
    "core8": ["Oz", "P8", "PO8", "O2", "PO7", "Pz", "P7", "P2"],
    "top12": ["Oz", "P8", "PO8", "O2", "PO7", "Pz", "P7", "P2", "POz", "O1", "P4", "P3"],
    "occipital": ["O1", "Oz", "O2", "PO7", "PO3", "POz", "PO4", "PO8"],
    "parietal": ["Pz", "P3", "P7", "P4", "P8", "P1", "P5", "P6", "P2"],
}

DEFAULT_BANDS: list[tuple[float, float]] = [
    (0.5, 4.0),   # delta (slow visual evoked envelope)
    (4.0, 8.0),   # theta (memory & semantic retrieval)
    (8.0, 13.0),  # alpha (occipital visual engagement & attention)
    (13.0, 30.0), # beta (active visual processing)
]

RICH_BANDS: list[tuple[float, float]] = [
    (0.1, 4.0),
    (4.0, 8.0),
    (8.0, 13.0),
    (13.0, 20.0),
    (20.0, 30.0),
]


def resolve_channel_indices(channels_spec: str | Sequence[str]) -> tuple[list[int], list[str]]:
    """Resolve channel specification into sorted indices and electrode names."""
    if isinstance(channels_spec, str):
        spec = channels_spec.strip().lower()
        if spec in PREDEFINED_CHANNEL_SETS:
            names = PREDEFINED_CHANNEL_SETS[spec]
        elif spec in ("all", "*"):
            names = ALL_CHANNELS
        else:
            names = [item.strip() for item in channels_spec.split(",") if item.strip()]
    else:
        names = list(channels_spec)

    # Standardize case matching ALL_CHANNELS
    canonical_lookup = {name.lower(): name for name in ALL_CHANNELS}
    resolved_names: list[str] = []
    resolved_indices: list[int] = []
    seen = set()
    for name in names:
        canon = canonical_lookup.get(name.lower())
        if canon is None:
            raise ValueError(f"Unknown EEG electrode: '{name}'. Available: {ALL_CHANNELS}")
        if canon not in seen:
            seen.add(canon)
            resolved_names.append(canon)
            resolved_indices.append(CHANNEL_TO_IDX[canon])

    if not resolved_indices:
        raise ValueError("At least one electrode must be selected")
    return resolved_indices, resolved_names


def compute_band_power(
    eeg: np.ndarray,
    sfreq: float = 100.0,
    bands: list[tuple[float, float]] | None = None,
) -> np.ndarray:
    """Compute per-channel log band-power via real FFT."""
    if bands is None:
        bands = DEFAULT_BANDS

    N, C, T = eeg.shape
    # Detrend per channel/trial
    x = eeg - eeg.mean(axis=-1, keepdims=True)
    freqs = np.fft.rfftfreq(T, d=1.0 / sfreq)
    spec = np.abs(np.fft.rfft(x, axis=-1)) ** 2

    out = np.empty((N, C, len(bands)), dtype=np.float32)
    for bi, (lo, hi) in enumerate(bands):
        mask = (freqs >= lo) & (freqs < hi)
        if not mask.any():
            out[:, :, bi] = 0.0
        else:
            out[:, :, bi] = spec[:, :, mask].mean(axis=-1)

    out = np.log1p(out)
    return out.reshape(N, C * len(bands))


def compute_erp_statistics(eeg: np.ndarray) -> np.ndarray:
    """Compute temporal ERP statistics per channel:
    - P100 window mean (samples 0..7, ~50-120ms)
    - N170 window mean (samples 7..17, ~120-220ms)
    - P300 window mean (samples 17..35, ~220-400ms)
    - Late window mean (samples 35..55, ~400-600ms)
    - Peak-to-peak amplitude (max - min)
    """
    N, C, T = eeg.shape
    p100 = eeg[:, :, :min(7, T)].mean(axis=-1)
    n170 = eeg[:, :, min(7, T):min(17, T)].mean(axis=-1) if T > 7 else p100
    p300 = eeg[:, :, min(17, T):min(35, T)].mean(axis=-1) if T > 17 else n170
    late = eeg[:, :, min(35, T):].mean(axis=-1) if T > 35 else p300
    p2p = eeg.max(axis=-1) - eeg.min(axis=-1)

    stats = np.stack([p100, n170, p300, late, p2p], axis=-1)
    return stats.reshape(N, C * 5).astype(np.float32)


def compute_connectivity(eeg: np.ndarray) -> np.ndarray:
    """Per-trial channel-pair Pearson correlation matrix across electrodes."""
    N, C, T = eeg.shape
    if C < 2:
        return np.zeros((N, 0), dtype=np.float32)
    x = eeg - eeg.mean(axis=-1, keepdims=True)
    sd = x.std(axis=-1, keepdims=True)
    x = x / (sd + 1e-8)
    corr = np.einsum("nct,ndt->ncd", x, x) / max(T, 1)
    iu = np.triu_indices(C, k=1)
    return corr[:, iu[0], iu[1]].astype(np.float32)


def extract_features(
    eeg: np.ndarray,
    feature_mode: str,
    channel_indices: list[int] | None = None,
    sfreq: float = 100.0,
) -> np.ndarray:
    """Extract flat per-trial feature vectors for the given EEG and feature mode.

    Parameters:
    - eeg: (N, 17, T) or (N, C, T) float32 array
    - feature_mode: 'raw', 'bandpower', 'both', 'erp_stats', 'rich', 'full'
    - channel_indices: optional channel subset indices to slice along axis 1

    Returns:
    - (N, feature_dim) float32 array
    """
    if channel_indices is not None:
        eeg = eeg[:, channel_indices, :]

    N, C, T = eeg.shape
    mode = feature_mode.lower()

    if mode == "raw":
        return eeg.reshape(N, C * T).astype(np.float32)
    elif mode == "bandpower":
        return compute_band_power(eeg, sfreq=sfreq, bands=DEFAULT_BANDS)
    elif mode == "both":
        raw = eeg.reshape(N, C * T).astype(np.float32)
        bp = compute_band_power(eeg, sfreq=sfreq, bands=DEFAULT_BANDS)
        return np.concatenate([raw, bp], axis=1)
    elif mode == "erp_stats":
        raw = eeg.reshape(N, C * T).astype(np.float32)
        erp = compute_erp_statistics(eeg)
        return np.concatenate([raw, erp], axis=1)
    elif mode == "rich":
        bp = compute_band_power(eeg, sfreq=sfreq, bands=RICH_BANDS)
        conn = compute_connectivity(eeg)
        return np.concatenate([bp, conn], axis=1)
    elif mode == "full":
        raw = eeg.reshape(N, C * T).astype(np.float32)
        bp = compute_band_power(eeg, sfreq=sfreq, bands=DEFAULT_BANDS)
        erp = compute_erp_statistics(eeg)
        return np.concatenate([raw, bp, erp], axis=1)
    else:
        raise ValueError(
            f"Unknown feature mode: '{feature_mode}'. "
            "Supported: 'raw', 'bandpower', 'both', 'erp_stats', 'rich', 'full'"
        )


def compute_feature_dim(n_channels: int, n_timepoints: int, feature_mode: str) -> int:
    """Compute the expected feature dimensionality for the model."""
    mode = feature_mode.lower()
    if mode == "raw":
        return n_channels * n_timepoints
    elif mode == "bandpower":
        return n_channels * len(DEFAULT_BANDS)
    elif mode == "both":
        return n_channels * (n_timepoints + len(DEFAULT_BANDS))
    elif mode == "erp_stats":
        return n_channels * (n_timepoints + 5)
    elif mode == "rich":
        n_pairs = n_channels * (n_channels - 1) // 2
        return n_channels * len(RICH_BANDS) + n_pairs
    elif mode == "full":
        return n_channels * (n_timepoints + len(DEFAULT_BANDS) + 5)
    else:
        raise ValueError(f"Unknown feature mode: {feature_mode}")
