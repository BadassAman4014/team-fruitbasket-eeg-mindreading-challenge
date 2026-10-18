"""PyTorch Dataset utilities for EEG-Stimulus multi-subject contrastive training.
"""

import os
from typing import List, Tuple, Optional
import numpy as np
import torch
from torch.utils.data import Dataset

from src.data_utils import (
    load_train_eeg,
    load_test_eeg,
    decimate_eeg,
    zscore_per_subject,
    normalize_eeg_trials,
)


class EEGStimulusDataset(Dataset):
    """Multi-subject EEG and multimodal stimulus dataset.

    Parameters
    ----------
    subjects : list of int
        Subject numbers to include (e.g. [1, 2, 3, 4] or [1, 2, 3, 4, 5, 6]).
    stim_feats : np.ndarray
        Array of shape (1001, feat_dim) indexed by 1-based stimulus ID.
    trn_dir : str
        Directory containing 'sub-XX_trn.npz' files.
    augment : bool, default=False
        Whether to apply light temporal jitter and Gaussian noise during training.
    """

    def __init__(
        self,
        subjects: List[int],
        stim_feats: np.ndarray,
        trn_dir: str = "./trn",
        augment: bool = False,
        stim_range: Optional[Tuple[int, int]] = None,
        eeg_window: Optional[Tuple[int, int]] = (5, 60),
        norm_mode: str = "trial",
    ):
        self.stim_feats = torch.as_tensor(stim_feats, dtype=torch.float32)
        self.augment = augment
        self.eeg_window = eeg_window
        self.norm_mode = norm_mode

        all_eeg = []
        all_stims = []
        all_subjects = []

        range_desc = f" (stimuli {stim_range[0]}-{stim_range[1]})" if stim_range else ""
        win_desc = f" | Window={eeg_window}" if eeg_window else " | Window=Full(300)"
        norm_desc = f" | Norm={norm_mode}"
        print(f"Loading EEG data for subjects: {subjects}{range_desc}{win_desc}{norm_desc} ...")
        for sub in subjects:
            npz_path = os.path.join(trn_dir, f"sub-{sub:02d}_trn.npz")
            if not os.path.exists(npz_path):
                print(f"  Warning: {npz_path} not found, skipping subject {sub}")
                continue

            ids, raw_eeg = load_train_eeg(npz_path)  # (N, 17, 1500)
            dec_eeg = decimate_eeg(raw_eeg)         # (N, 17, 300)

            if eeg_window is not None:
                dec_eeg = dec_eeg[..., eeg_window[0]:eeg_window[1]]

            if norm_mode == "trial":
                norm_eeg = normalize_eeg_trials(dec_eeg)
            else:
                norm_eeg = zscore_per_subject(dec_eeg)

            if stim_range is not None:
                mask = (ids >= stim_range[0]) & (ids <= stim_range[1])
                ids = ids[mask]
                norm_eeg = norm_eeg[mask]

            all_eeg.append(norm_eeg)
            all_stims.append(ids)
            all_subjects.extend([sub] * len(ids))
            print(f"  sub-{sub:02d}: loaded {len(ids)} trials, shape={norm_eeg.shape}")

        self.eeg = torch.as_tensor(np.concatenate(all_eeg, axis=0), dtype=torch.float32)
        self.stim_ids = np.concatenate(all_stims, axis=0)
        self.subject_ids = np.array(all_subjects)

        print(f"Total dataset size: {len(self.eeg)} trials across {len(subjects)} subjects.")

    def __len__(self) -> int:
        return len(self.eeg)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        eeg_sample = self.eeg[idx]  # (17, 300)
        stim_id = int(self.stim_ids[idx])
        sub_id = int(self.subject_ids[idx])
        stim_feat = self.stim_feats[stim_id]  # (feat_dim,)

        if self.augment:
            # 1. Random small temporal crop / roll (+/- 5 timepoints, ~50 ms)
            shift = np.random.randint(-5, 6)
            if shift != 0:
                eeg_sample = torch.roll(eeg_sample, shifts=shift, dims=-1)

            # 2. Add subtle Gaussian noise (SNR > 25 dB)
            noise = torch.randn_like(eeg_sample) * 0.02
            eeg_sample = eeg_sample + noise

        return eeg_sample, stim_feat, stim_id, sub_id


def load_eval_test_data(
    sub_num: int,
    test_dir: str = "./tst",
    gt_dir: str = "./tst_stim",
    eeg_window: Optional[Tuple[int, int]] = (5, 60),
    norm_mode: str = "trial",
) -> Tuple[np.ndarray, np.ndarray]:
    """Load and preprocess test EEG and ground-truth stimulus IDs for evaluation.

    Returns
    -------
    test_eeg : (102, 17, T) sorted by ground truth stimulus ID
    gt_stim_ids : (102,) sorted ground truth IDs
    """
    raw_test = load_test_eeg(os.path.join(test_dir, f"sub-{sub_num:02d}_tst.npy"))
    dec_test = decimate_eeg(raw_test)

    if eeg_window is not None:
        dec_test = dec_test[..., eeg_window[0]:eeg_window[1]]

    if norm_mode == "trial":
        norm_test = normalize_eeg_trials(dec_test)
    else:
        norm_test = zscore_per_subject(dec_test)

    gt_path = os.path.join(gt_dir, f"sub-{sub_num:02d}_tst_stim.npy")
    gt_stim_ids = np.load(gt_path)

    # Sort rows by ground truth stimulus ID so diagonal corresponds to correct match
    order = np.argsort(gt_stim_ids)
    sorted_eeg = norm_test[order]
    sorted_gt = gt_stim_ids[order]

    return sorted_eeg, sorted_gt


def load_val_retrieval_data(
    subjects: List[int],
    stim_feats: np.ndarray,
    trn_dir: str = "./trn",
    val_range: Tuple[int, int] = (901, 1000),
    eeg_window: Optional[Tuple[int, int]] = (5, 60),
    norm_mode: str = "trial",
) -> dict:
    """Load validation trials and candidate stimulus features for held-out training stimuli.

    Extracts all trials from the specified subjects where stimulus ID is within val_range.
    Candidate stimuli for the retrieval task are exactly the stimuli in val_range.

    Returns
    -------
    dict with keys:
        'val_eeg': (N_val, 17, T) float32 tensor
        'val_stims': (N_val,) int array
        'val_subs': (N_val,) int tensor (1-based subject numbers)
        'cand_feats': (N_cands, feat_dim) float32 tensor
        'cand_ids': (N_cands,) int array
    """
    val_eeg_list = []
    val_stims_list = []
    val_subs_list = []

    for sub in subjects:
        npz_path = os.path.join(trn_dir, f"sub-{sub:02d}_trn.npz")
        if not os.path.exists(npz_path):
            continue
        ids, raw_eeg = load_train_eeg(npz_path)
        dec_eeg = decimate_eeg(raw_eeg)

        if eeg_window is not None:
            dec_eeg = dec_eeg[..., eeg_window[0]:eeg_window[1]]

        if norm_mode == "trial":
            norm_eeg = normalize_eeg_trials(dec_eeg)
        else:
            norm_eeg = zscore_per_subject(dec_eeg)

        mask = (ids >= val_range[0]) & (ids <= val_range[1])
        if mask.any():
            val_eeg_list.append(norm_eeg[mask])
            val_stims_list.append(ids[mask])
            val_subs_list.extend([sub] * int(mask.sum()))

    val_eeg = torch.as_tensor(np.concatenate(val_eeg_list, axis=0), dtype=torch.float32)
    val_stims = np.concatenate(val_stims_list, axis=0)
    val_subs = torch.as_tensor(val_subs_list, dtype=torch.long)

    cand_ids = np.arange(val_range[0], val_range[1] + 1)
    cand_feats = torch.as_tensor(stim_feats[cand_ids], dtype=torch.float32)

    return {
        "val_eeg": val_eeg,
        "val_stims": val_stims,
        "val_subs": val_subs,
        "cand_feats": cand_feats,
        "cand_ids": cand_ids,
    }

