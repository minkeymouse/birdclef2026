"""Shared eval metrics: per-class AUC, macro AUC, per-row Spearman."""
from __future__ import annotations
import numpy as np
from sklearn.metrics import roc_auc_score


def per_class_auc(Y: np.ndarray, P: np.ndarray) -> dict[int, float]:
    """Per-class AUC for classes with both pos+neg in Y."""
    out = {}
    for c in range(Y.shape[1]):
        n_pos = Y[:, c].sum()
        if n_pos == 0 or n_pos == Y.shape[0]: continue
        try: out[c] = float(roc_auc_score(Y[:, c], P[:, c]))
        except Exception: pass
    return out


def macro_auc(Y: np.ndarray, P: np.ndarray) -> tuple[float, int]:
    aucs = per_class_auc(Y, P)
    return (float(np.mean(list(aucs.values()))) if aucs else float("nan")), len(aucs)


def per_taxon_macro(Y: np.ndarray, P: np.ndarray, sp_taxon: np.ndarray) -> dict[str, float]:
    aucs = per_class_auc(Y, P)
    out = {t: [] for t in ("Aves", "Amphibia", "Insecta", "Mammalia", "Reptilia")}
    for c, a in aucs.items():
        if sp_taxon[c] in out: out[sp_taxon[c]].append(a)
    return {t: (float(np.mean(v)) if v else float("nan")) for t, v in out.items()}


def per_row_spearman(P_ref: np.ndarray, P_new: np.ndarray) -> float:
    """Mean per-row Spearman correlation between ref and new predictions (transfer-safety proxy)."""
    from scipy.stats import spearmanr
    sps = []
    for i in range(len(P_ref)):
        r, _ = spearmanr(P_ref[i], P_new[i])
        if np.isfinite(r): sps.append(r)
    return float(np.mean(sps)) if sps else float("nan")


def gpu_mlp_binary_auc(X_tr, y_tr, X_ev, y_ev,
                       hidden: int = 64, epochs: int = 300, lr: float = 1e-3, wd: float = 1e-4,
                       device: str = "cuda") -> float:
    """Tiny GPU MLP binary classifier with auto pos_weight balancing."""
    import torch, torch.nn as nn, torch.nn.functional as F
    if y_tr.sum() < 3 or y_ev.sum() < 1 or y_ev.sum() == len(y_ev):
        return float("nan")
    in_dim = X_tr.shape[1]
    m = nn.Sequential(
        nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(0.1),
        nn.Linear(hidden, 1)).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
    Xt = torch.from_numpy(X_tr.astype("float32")).to(device)
    yt = torch.from_numpy(y_tr.astype("float32")).to(device)
    Xe = torch.from_numpy(X_ev.astype("float32")).to(device)
    pw = torch.tensor([(y_tr == 0).sum() / max(y_tr.sum(), 1)], dtype=torch.float32, device=device)
    for _ in range(epochs):
        m.train(); opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(m(Xt).squeeze(-1), yt, pos_weight=pw)
        loss.backward(); opt.step()
    m.eval()
    with torch.no_grad():
        p = torch.sigmoid(m(Xe).squeeze(-1)).cpu().numpy()
    try: return float(roc_auc_score(y_ev, p))
    except: return float("nan")
