"""PCA + Ridge encoding model — same architecture as the organiser baseline.

What changes vs. the baseline:
  - Video features are injected externally (extractor-agnostic).
  - Works with any feature dimensionality (ViCLIP 768-d or ImageBind 1024-d).
  - Supports concatenated features (Phase 1C hybrid).
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
import joblib

from .data_utils import EEG_WINDOW


class EncodingModel:
    """Forward model: video features  →  predicted EEG  →  ranked candidates.

    Pipeline:  X  →  StandardScaler  →  PCA(0.95)  →  StandardScaler
                   →  Ridge(CV)  →  predicted EEG  →  cosine similarity

    Parameters
    ----------
    pca_variance : float
        Fraction of variance retained by PCA. Default 0.95.
    ridge_alphas : array-like, optional
        Alpha search grid for RidgeCV. Defaults to log-space (1e-3, 1e6).
    """

    def __init__(self, pca_variance: float = 0.95, ridge_alphas=None):
        self.pca_variance = pca_variance
        self.ridge_alphas = ridge_alphas if ridge_alphas is not None \
                            else np.logspace(-3, 6, 20)
        self._sc1   = StandardScaler()
        self._pca   = PCA(n_components=pca_variance, svd_solver='full')
        self._sc2   = StandardScaler()
        self._ridge = RidgeCV(alphas=self.ridge_alphas)
        self._C = self._T = None

    # ── Fitting ─────────────────────────────────────────────────────────────

    def fit(self, X: np.ndarray, eeg: np.ndarray) -> "EncodingModel":
        """Train the encoding model.

        Parameters
        ----------
        X   : (N, D)    video feature matrix — one row per stimulus
        eeg : (N, C, T) preprocessed EEG at 100 Hz (after decimation)
        """
        N, C, T = eeg.shape
        self._C, self._T = C, T

        Xs = self._sc1.fit_transform(X)
        Xp = self._pca.fit_transform(Xs)
        Xp = self._sc2.fit_transform(Xp)

        self._ridge.fit(Xp, eeg.reshape(N, C * T))
        print(f"  PCA kept {self._pca.n_components_} components "
              f"(of {X.shape[1]})  |  Ridge α={self._ridge.alpha_:.4g}")
        return self

    # ── Inference ────────────────────────────────────────────────────────────

    def predict_eeg(self, X: np.ndarray) -> np.ndarray:
        """Predict EEG for M stimuli.

        Parameters
        ----------
        X : (M, D) feature matrix

        Returns
        -------
        (M, C, T) predicted EEG
        """
        Xs = self._sc1.transform(X)
        Xp = self._pca.transform(Xs)
        Xp = self._sc2.transform(Xp)
        return self._ridge.predict(Xp).reshape(-1, self._C, self._T)

    def similarity_matrix(
        self,
        measured_eeg: np.ndarray,
        candidate_features: np.ndarray,
    ) -> np.ndarray:
        """Pearson-correlation similarity: measured vs. predicted EEG.

        Parameters
        ----------
        measured_eeg       : (M, C, T)  — recorded EEG for M test trials
        candidate_features : (K, D)     — features for K candidate videos

        Returns
        -------
        S : (M, K) similarity matrix  (higher = more likely match)
        """
        pred = self.predict_eeg(candidate_features)   # (K, C, T)

        def _zrow(a: np.ndarray) -> np.ndarray:
            mu = a.mean(axis=1, keepdims=True)
            sd = a.std(axis=1,  keepdims=True).clip(min=1e-8)
            return (a - mu) / sd

        # Slice informative window and flatten per trial
        M_win = measured_eeg[:, :, EEG_WINDOW].reshape(len(measured_eeg), -1)
        P_win = pred[:,         :, EEG_WINDOW].reshape(len(candidate_features), -1)

        return _zrow(M_win) @ _zrow(P_win).T / M_win.shape[1]  # (M, K)

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self, path: str):
        joblib.dump(self, path, compress=3)
        print(f"  Model saved → {path}")

    @classmethod
    def load(cls, path: str) -> "EncodingModel":
        return joblib.load(path)
