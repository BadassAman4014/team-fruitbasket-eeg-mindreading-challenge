"""
src/stim_store.py - Multimodal Stimulus Feature Store
=====================================================
Loads, pre-processes, caches, and indexes multimodal stimulus feature blocks
(DINOv2 ViT-B/14, CLAP, and AST) for training and candidate retrieval.

Complies with WP1/WP10 of the Contrastive Retrieval Plan:
- Explicit ID-to-index mapping (never position-based).
- Normalization (StandardScaler) and optional FastPCA fit strictly on training IDs.
- Clean tensor lookup API: StimStore.get(ids, block_names)
"""

import os
import glob
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from model_pipeline import FastPCA


class StimStore:
    def __init__(self, data_dict, id_map, dims):
        """
        data_dict: dict of {block_name: np.ndarray of shape (N, D)}
        id_map: dict mapping video_id (int) -> row index in data_dict arrays
        dims: dict of {block_name: feature_dim}
        """
        self.data = data_dict
        self.id_map = id_map
        self.dims = dims

    def get(self, ids, block_names=None, device='cpu'):
        """
        Returns tensors for the given IDs and block names.
        ids: list or 1D array of integer video IDs
        block_names: list of blocks (default: all blocks)
        Returns: dict of {block_name: torch.FloatTensor of shape (len(ids), D)}
        """
        if block_names is None:
            block_names = list(self.data.keys())

        indices = [self.id_map[int(vid)] for vid in ids]
        out = {}
        for b in block_names:
            arr = self.data[b][indices]
            out[b] = torch.from_numpy(arr).float().to(device)
        return out

    def get_all_candidate_ids(self):
        """Returns sorted list of test candidate IDs (1001 to 1102)."""
        return [i for i in range(1001, 1103) if i in self.id_map]


def build_stim_store(blocks=('vis_dino', 'aud_clap', 'aud_ast'), pca_dim=None, device='cpu'):
    """
    Builds and returns a unified StimStore containing normalized features for all
    training videos (1-1000) and test candidate videos (1001-1102).
    """
    cache_dir = './cache/stim'
    os.makedirs(cache_dir, exist_ok=True)

    # 1. Determine all available train IDs from .npz files or stimulus directory
    train_ids = set()
    for sub in range(1, 7):
        trn_path = f'./trn/sub-{sub:02d}_trn.npz'
        if os.path.exists(trn_path):
            z = np.load(trn_path)
            train_ids.update([int(k) for k in z.files])
    train_ids = sorted(list(train_ids))
    test_ids = [i for i in range(1001, 1103)]
    all_ids = train_ids + test_ids
    id_map = {vid: idx for idx, vid in enumerate(all_ids)}

    data_dict = {}
    dims = {}

    # -------------------------------------------------------------
    # Block 1: DINOv2 (ViT-B/14) Visual Features
    # -------------------------------------------------------------
    if 'vis_dino' in blocks:
        trn_feat_path = './features_dinov2/consolidated_encoder.layer.11.npz'
        tst_feat_path = './features_test_dinov2/consolidated_encoder.layer.11.npz'

        if not os.path.exists(trn_feat_path) or not os.path.exists(tst_feat_path):
            raise FileNotFoundError(
                f"DINOv2 features not found at {trn_feat_path} or {tst_feat_path}. "
                "Run `python benchmark_dinov2.py --sub 1` to extract them."
            )

        z_trn = np.load(trn_feat_path)
        z_tst = np.load(tst_feat_path)

        # Pre-index all available keys in z_trn and z_tst
        dino_train_dict = {}
        for k in z_trn.files:
            try:
                prefix = int(k.split('_')[0])
                dino_train_dict.setdefault(prefix, []).append(z_trn[k].flatten())
            except Exception:
                pass
        dino_test_dict = {}
        for k in z_tst.files:
            try:
                prefix = int(k.split('_')[0])
                dino_test_dict.setdefault(prefix, []).append(z_tst[k].flatten())
            except Exception:
                pass

        all_vecs = [np.mean(v, axis=0) for v in dino_train_dict.values()]
        global_mean_vis = np.mean(all_vecs, axis=0) if all_vecs else np.zeros(1536, dtype=np.float32)

        def get_dino_vec(d_dict, vid):
            if int(vid) in d_dict:
                return np.mean(d_dict[int(vid)], axis=0)
            return global_mean_vis

        raw_trn_vis = np.stack([get_dino_vec(dino_train_dict, vid) for vid in train_ids])
        raw_tst_vis = np.stack([get_dino_vec(dino_test_dict, vid) for vid in test_ids])

        # Fit StandardScaler and PCA strictly on training IDs
        sc1 = StandardScaler().fit(raw_trn_vis)
        trn_norm = sc1.transform(raw_trn_vis)
        tst_norm = sc1.transform(raw_tst_vis)

        if pca_dim is not None:
            pca = FastPCA(n_components=pca_dim, device=device).fit(trn_norm)
            trn_proj = pca.transform(trn_norm)
            tst_proj = pca.transform(tst_norm)
            sc2 = StandardScaler().fit(trn_proj)
            vis_trn = sc2.transform(trn_proj).astype(np.float32)
            vis_tst = sc2.transform(tst_proj).astype(np.float32)
        else:
            vis_trn = trn_norm.astype(np.float32)
            vis_tst = tst_norm.astype(np.float32)

        data_dict['vis_dino'] = np.vstack([vis_trn, vis_tst])
        dims['vis_dino'] = data_dict['vis_dino'].shape[1]
        print(f"[StimStore] vis_dino loaded: shape {data_dict['vis_dino'].shape} (dim={dims['vis_dino']})")

    # -------------------------------------------------------------
    # Block 2: CLAP Contrastive Audio Features
    # -------------------------------------------------------------
    if 'aud_clap' in blocks:
        trn_aud_path = './features_audio/train_clap_features.npz'
        tst_aud_path = './features_audio/test_clap_features.npz'

        if not os.path.exists(trn_aud_path) or not os.path.exists(tst_aud_path):
            raise FileNotFoundError(f"CLAP audio cache not found at {trn_aud_path} or {tst_aud_path}")

        z_trn_aud = np.load(trn_aud_path)
        z_tst_aud = np.load(tst_aud_path)

        all_clap = [z_trn_aud[k] for k in z_trn_aud.files]
        global_mean_clap = np.mean(all_clap, axis=0) if all_clap else np.zeros(512, dtype=np.float32)

        def get_clap_vec(z_file, vid):
            k = f"stim_{int(vid):05d}"
            return z_file[k] if k in z_file else global_mean_clap

        raw_trn_clap = np.stack([get_clap_vec(z_trn_aud, vid) for vid in train_ids])
        raw_tst_clap = np.stack([get_clap_vec(z_tst_aud, vid) for vid in test_ids])

        sc_clap = StandardScaler().fit(raw_trn_clap)
        clap_trn = sc_clap.transform(raw_trn_clap).astype(np.float32)
        clap_tst = sc_clap.transform(raw_tst_clap).astype(np.float32)

        data_dict['aud_clap'] = np.vstack([clap_trn, clap_tst])
        dims['aud_clap'] = data_dict['aud_clap'].shape[1]
        print(f"[StimStore] aud_clap loaded: shape {data_dict['aud_clap'].shape} (dim={dims['aud_clap']})")

    # -------------------------------------------------------------
    # Block 3: AST (Audio Spectrogram Transformer) Features
    # -------------------------------------------------------------
    if 'aud_ast' in blocks:
        trn_ast_path = './features_audio/train_ast_features.npz'
        tst_ast_path = './features_audio/test_ast_features.npz'

        if not os.path.exists(trn_ast_path) or not os.path.exists(tst_ast_path):
            raise FileNotFoundError(f"AST audio cache not found at {trn_ast_path} or {tst_ast_path}")

        z_trn_ast = np.load(trn_ast_path)
        z_tst_ast = np.load(tst_ast_path)

        all_ast = [z_trn_ast[k] for k in z_trn_ast.files]
        global_mean_ast = np.mean(all_ast, axis=0) if all_ast else np.zeros(768, dtype=np.float32)

        def get_ast_vec(z_file, vid):
            k = f"stim_{int(vid):05d}"
            return z_file[k] if k in z_file else global_mean_ast

        raw_trn_ast = np.stack([get_ast_vec(z_trn_ast, vid) for vid in train_ids])
        raw_tst_ast = np.stack([get_ast_vec(z_tst_ast, vid) for vid in test_ids])

        sc_ast = StandardScaler().fit(raw_trn_ast)
        ast_trn = sc_ast.transform(raw_trn_ast).astype(np.float32)
        ast_tst = sc_ast.transform(raw_tst_ast).astype(np.float32)

        data_dict['aud_ast'] = np.vstack([ast_trn, ast_tst])
        dims['aud_ast'] = data_dict['aud_ast'].shape[1]
        print(f"[StimStore] aud_ast loaded: shape {data_dict['aud_ast'].shape} (dim={dims['aud_ast']})")

    return StimStore(data_dict, id_map, dims)
