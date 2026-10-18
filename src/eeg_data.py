"""
src/eeg_data.py - EEG Data Loading, Preprocessing, and Batch Sampling
=====================================================================
Implements WP2 of the Contrastive Retrieval Plan:
- Preprocesses raw (17, 1500) 500Hz EEG to 100Hz (17, 300) zero-phase FIR.
- Per-subject, per-channel normalization fit on training trials and applied to test.
- UniqueVideoSampler: batches of B distinct video IDs across pooled subjects.
- Robust data augmentation (small temporal shift +-2, channel dropout, Gaussian noise).
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
from model_pipeline import preprocess_eeg


class EEGBank:
    """
    In-memory preprocessed and normalized EEG bank for training and test trials
    across all subjects.
    """
    def __init__(self, subjects=(1, 2, 3, 4, 5, 6), dec=5, sfreq=500, clip_sd=5.0):
        self.dec = dec
        self.sfreq = sfreq
        self.clip_sd = clip_sd
        self.train_data = {}  # sub -> (N_trials, 17, 300)
        self.train_ids = {}   # sub -> list of int video IDs
        self.test_data = {}   # sub -> (102, 17, 300)
        self.sub_stats = {}   # sub -> (mean, std)

        for s in subjects:
            self._load_subject(s)

    def _load_subject(self, sub):
        trn_path = f'./trn/sub-{sub:02d}_trn.npz'
        tst_path = f'./tst/sub-{sub:02d}_tst.npy'

        if not os.path.exists(trn_path) or not os.path.exists(tst_path):
            return

        # 1. Load train EEG
        z = np.load(trn_path)
        vids = np.array(sorted(z.files, key=int), dtype=int)
        raw_trn = np.stack([z[str(v)] for v in vids])
        eeg_trn, _ = preprocess_eeg(raw_trn, dec=self.dec, sfreq=self.sfreq)

        # Compute per-channel mean and std strictly on training trials
        # eeg_trn is (N, 17, 300)
        ch_mean = np.mean(eeg_trn, axis=(0, 2), keepdims=True)  # (1, 17, 1)
        ch_std = np.std(eeg_trn, axis=(0, 2), keepdims=True) + 1e-6
        self.sub_stats[sub] = (ch_mean, ch_std)

        # Normalize train EEG
        eeg_trn_norm = np.clip((eeg_trn - ch_mean) / ch_std, -self.clip_sd, self.clip_sd).astype(np.float32)
        self.train_data[sub] = eeg_trn_norm
        self.train_ids[sub] = vids

        # 2. Load test EEG
        raw_tst = np.load(tst_path)
        eeg_tst, _ = preprocess_eeg(raw_tst, dec=self.dec, sfreq=self.sfreq)
        eeg_tst_norm = np.clip((eeg_tst - ch_mean) / ch_std, -self.clip_sd, self.clip_sd).astype(np.float32)
        self.test_data[sub] = eeg_tst_norm


class UniqueVideoSampler(Sampler):
    """
    Samples batches where each batch holds B strictly distinct video IDs.
    For each selected video ID, one random trial (from any subject that watched it) is drawn.
    """
    def __init__(self, trial_table, batch_size=64, steps_per_epoch=None, seed=42):
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)

        # Map video_id -> list of trial indices in trial_table
        self.vid_to_indices = {}
        for idx, (sub, vid, row) in enumerate(trial_table):
            self.vid_to_indices.setdefault(vid, []).append(idx)

        self.unique_vids = np.array(list(self.vid_to_indices.keys()))
        self.total_trials = len(trial_table)
        self.steps = steps_per_epoch or (self.total_trials // batch_size)

    def __iter__(self):
        for _ in range(self.steps):
            selected_vids = self.rng.choice(self.unique_vids, size=self.batch_size, replace=False)
            batch = [self.rng.choice(self.vid_to_indices[v]) for v in selected_vids]
            yield batch

    def __len__(self):
        return self.steps


class EEGTrialDataset(Dataset):
    """
    Dataset of multimodal EEG trials.
    Returns:
      eeg_tensor: (17, T) float32
      sub_idx: int (0 to 5 for subject 1 to 6)
      vid_id: int
      stim_features: dict of {block: tensor}
    """
    def __init__(self, trial_table, eeg_bank, stim_store, blocks=('vis_dino', 'aud_clap'), augment=False):
        self.trials = trial_table  # list of (sub, vid, row_in_sub)
        self.eeg_bank = eeg_bank
        self.stim_store = stim_store
        self.blocks = blocks
        self.augment = augment

    def __len__(self):
        return len(self.trials)

    def __getitem__(self, idx):
        sub, vid, row = self.trials[idx]
        eeg = self.eeg_bank.train_data[sub][row].copy()  # (17, 300)

        # Data augmentation
        if self.augment:
            # 1. Temporal jitter: roll +- 2 samples
            shift = np.random.randint(-2, 3)
            if shift != 0:
                eeg = np.roll(eeg, shift, axis=-1)
            # 2. Channel dropout: randomly zero 1 or 2 channels with prob 0.1
            if np.random.rand() < 0.10:
                drop_ch = np.random.choice(17, size=np.random.randint(1, 3), replace=False)
                eeg[drop_ch, :] = 0.0
            # 3. Additive Gaussian noise
            if np.random.rand() < 0.20:
                eeg += np.random.randn(*eeg.shape).astype(np.float32) * 0.05

        stim_tensors = self.stim_store.get([vid], self.blocks)
        # Squeeze batch dimension for dataset item
        stim_item = {k: v.squeeze(0) for k, v in stim_tensors.items()}

        sub_idx = sub - 1  # 0-indexed
        return torch.from_numpy(eeg).float(), sub_idx, vid, stim_item


def collate_contrastive_batch(batch):
    eegs, sub_idxs, vids, stim_dicts = zip(*batch)
    eeg_batch = torch.stack(eegs, dim=0)           # (B, 17, 300)
    sub_batch = torch.tensor(sub_idxs, dtype=torch.long) # (B,)
    vid_batch = torch.tensor(vids, dtype=torch.long)     # (B,)

    blocks = list(stim_dicts[0].keys())
    stim_batch = {}
    for b in blocks:
        stim_batch[b] = torch.stack([d[b] for d in stim_dicts], dim=0)

    return eeg_batch, sub_batch, vid_batch, stim_batch
