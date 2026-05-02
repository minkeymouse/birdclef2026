import numpy as np
import torch
import torch.nn as nn
import timm
from typing import Optional, Union


def logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Numerically stable logit: ln(p / (1-p)), clipping p to [eps, 1-eps].
    """
    p_clipped = np.clip(p, eps, 1 - eps)
    return np.log(p_clipped / (1 - p_clipped))


def sigmoid(y: np.ndarray) -> np.ndarray:
    """
    Elementwise sigmoid back to probability domain.
    """
    return 1.0 / (1.0 + np.exp(-y))


class InferenceModel(nn.Module):
    """
    Inference wrapper that directly uses a timm model with its built-in head.
    """
    def __init__(self, arch: str, in_chans: int, num_classes: int):
        super().__init__()
        # timm model includes its own classifier layer
        self.backbone = timm.create_model(
            arch,
            pretrained=False,
            in_chans=in_chans,
            num_classes=num_classes
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # timm backbone outputs raw logits (B, C)
        return torch.sigmoid(self.backbone(x))


class MultivariateBernoulliKalmanFilter:
    """
    Vector Kalman filter for multivariate Bernoulli smoothing on probabilities.
    - Maintains internal state in logit-space (y = logit(p)).
    - Converts back to probability after each step.
    """
    def __init__(
        self,
        num_states: int,
        Q_method: str = "kappa",    # 'kappa', 'empirical', or 'constant'
        kappa: float = 0.05,
        empirical_q: Optional[np.ndarray] = None,
        constant_q: float = 1e-3,
        include_offdiag: bool = False,
        rho_matrix: Optional[np.ndarray] = None,
        rho_thresh: float = 0.0,
        r_min: float = 1e-3,
        missing_tau: float = 0.01,
    ):
        self.n = num_states
        self.Q_method = Q_method
        self.kappa = kappa
        self.empirical_q = empirical_q
        self.constant_q = constant_q
        self.include_offdiag = include_offdiag
        self.rho_matrix = rho_matrix
        self.rho_thresh = rho_thresh
        self.r_min = r_min
        self.missing_tau = missing_tau

        # state placeholders
        self.y = np.zeros(self.n, dtype=np.float32)
        self.Py = np.eye(self.n, dtype=np.float32)

        # placeholders for covariances, to be set per recording
        self.Qy = np.eye(self.n, dtype=np.float32)
        self.Ry = np.eye(self.n, dtype=np.float32)

    def initialize(self, p0: np.ndarray, P0: float = 1.0):
        """
        Initialize filter state from initial probabilities p0 and variance P0.
        """
        self.y = logit(p0)
        self.Py = np.eye(self.n, dtype=np.float32) * P0

    def compute_R(self, preds: np.ndarray) -> None:
        """
        Estimate measurement covariance Ry from per-chunk preds (num_chunks x num_states).
        """
        var_diag = np.var(preds, axis=0, ddof=1)
        var_diag = np.maximum(var_diag, self.r_min)
        Ry = np.diag(var_diag)

        if self.include_offdiag and self.rho_matrix is not None:
            cov_full = np.cov(preds, rowvar=False, ddof=1)
            mask = np.abs(self.rho_matrix) >= self.rho_thresh
            offdiag = (cov_full * mask)
            offdiag = (offdiag + offdiag.T) / 2.0
            np.fill_diagonal(offdiag, 0.0)
            Ry += offdiag

        Ry += np.eye(self.n) * 1e-6
        self.Ry = Ry.astype(np.float32)

    def compute_Q(self) -> None:
        """
        Build process covariance Qy based on chosen method.
        """
        if self.Q_method == "kappa":
            q_diag = self.kappa * np.diag(self.Ry)
            Qy = np.diag(q_diag)
        elif self.Q_method == "empirical" and self.empirical_q is not None:
            Qy = np.diag(self.empirical_q)
        else:
            Qy = np.eye(self.n) * self.constant_q

        if self.include_offdiag and self.rho_matrix is not None:
            q_diag = np.diag(Qy)
            off = self.rho_matrix * np.sqrt(np.outer(q_diag, q_diag))
            mask = np.abs(self.rho_matrix) >= self.rho_thresh
            off = off * mask
            np.fill_diagonal(off, 0.0)
            Qy += off

        Qy += np.eye(self.n) * 1e-6
        self.Qy = Qy.astype(np.float32)

    def predict(self) -> None:
        """
        Kalman predict in logit-space: P_pred = P_prev + Qy; state unchanged.
        """
        self.Py = self.Py + self.Qy

    def update(
        self,
        p_obs: np.ndarray,
        obs_var: Optional[Union[np.ndarray, np.ndarray]] = None
    ) -> None:
        """
        Kalman update given observed probabilities p_obs.
        obs_var: if provided as a vector (C,), treated as diag(obs_var).
                 if provided as a matrix (C,C), used directly as R.
        """
        # Build measurement noise covariance R
        if obs_var is None:
            R = self.Ry
        else:
            R = np.diag(obs_var) if obs_var.ndim == 1 else obs_var

        # Mask missing observations
        mask = p_obs >= self.missing_tau
        y_obs = logit(p_obs)

        # Innovation covariance
        S = self.Py + R
        # Kalman gain
        K = self.Py @ np.linalg.inv(S)

        # State update
        innovation = y_obs - self.y
        self.y = self.y + K.dot(innovation * mask)
        # Covariance update
        self.Py = (np.eye(self.n) - K) @ self.Py

    @property
    def probabilities(self) -> np.ndarray:
        """
        Get current state back in probability domain.
        """
        return sigmoid(self.y)
