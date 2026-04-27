"""iVAE model + load helpers (encoder-only inference + full train).

Auxiliary-conditioned VAE: encoder is a function of x only (mel-flat),
prior p(z|aux) is conditioned on (site one-hot + hour), KL pushes the
posterior toward the prior so z is supposed to be aux-invariant.

Empirically (exp80a), z still encodes site fingerprint via the encoder
itself, because the encoder has no aux signal at inference and just
compresses raw mel — and raw mel is heavily site-dependent.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class IVAE(nn.Module):
    def __init__(self, in_dim: int, z_dim: int = 32, n_aux: int = 10, hidden: int = 512):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden, 256), nn.GELU(),
            nn.Linear(256, 2 * z_dim))
        self.aux_mlp = nn.Sequential(
            nn.Linear(n_aux, 64), nn.GELU(),
            nn.Linear(64, 2 * z_dim))
        self.dec = nn.Sequential(
            nn.Linear(z_dim, 256), nn.GELU(),
            nn.Linear(256, hidden), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden, in_dim))
        self.z_dim = z_dim

    def forward(self, x, aux):
        h = self.enc(x); mu_q, lv_q = h.chunk(2, dim=-1)
        h_a = self.aux_mlp(aux); mu_p, lv_p = h_a.chunk(2, dim=-1)
        z = mu_q + (0.5 * lv_q).exp() * torch.randn_like(mu_q)
        return self.dec(z), mu_q, lv_q, mu_p, lv_p

    def encode(self, x):
        h = self.enc(x); mu, _ = h.chunk(2, dim=-1)
        return mu


def kl_div(mu_q, lv_q, mu_p, lv_p):
    return 0.5 * (lv_p - lv_q - 1 + (lv_q - lv_p).exp()
                  + (mu_q - mu_p).pow(2) * (-lv_p).exp()).sum(-1).mean()


def train_full(X: 'np.ndarray', aux: 'np.ndarray', train_mask: 'np.ndarray',
               z_dim: int = 32, hidden: int = 512,
               epochs: int = 200, lr: float = 1e-3, wd: float = 1e-4, beta: float = 0.05,
               batch: int | None = None, verbose_every: int = 50, device: str = DEVICE,
               ) -> tuple[IVAE, 'np.ndarray', 'np.ndarray']:
    """Train iVAE on `train_mask` rows; encode all rows.
    Returns (trained model, train_mean, train_std)."""
    import numpy as np
    n_aux = aux.shape[1]
    train_mean = X[train_mask].mean(0)
    train_std = X[train_mask].std(0) + 1e-6
    Xs = (X - train_mean) / train_std

    m = IVAE(in_dim=X.shape[1], z_dim=z_dim, n_aux=n_aux, hidden=hidden).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)

    Xt = torch.from_numpy(Xs[train_mask].astype('float32')).to(device)
    At = torch.from_numpy(aux[train_mask].astype('float32')).to(device)
    n_fit = Xt.shape[0]

    if batch is None or batch >= n_fit:
        # Full-batch
        for ep in range(epochs):
            m.train(); opt.zero_grad()
            rx, muq, lvq, mup, lvp = m(Xt, At)
            rec = F.mse_loss(rx, Xt) * Xt.shape[1]
            kld = kl_div(muq, lvq, mup, lvp)
            (rec + beta * kld).backward(); opt.step()
            if verbose_every and (ep % verbose_every == 0 or ep == epochs - 1):
                print(f"  ep {ep:03d}  recon {rec.item():.2f}  kl {kld.item():.2f}")
    else:
        # Mini-batch
        for ep in range(epochs):
            perm = torch.randperm(n_fit, device=device)
            er, ek, nb = 0., 0., 0
            for s in range(0, n_fit, batch):
                idx = perm[s:s+batch]
                xb, ab = Xt[idx], At[idx]
                opt.zero_grad()
                rx, muq, lvq, mup, lvp = m(xb, ab)
                rec = F.mse_loss(rx, xb) * xb.shape[1]
                kld = kl_div(muq, lvq, mup, lvp)
                (rec + beta * kld).backward(); opt.step()
                er += rec.item(); ek += kld.item(); nb += 1
            if verbose_every and (ep % verbose_every == 0 or ep == epochs - 1):
                print(f"  ep {ep:03d}  recon {er/nb:.2f}  kl {ek/nb:.2f}")

    return m, train_mean.astype('float32'), train_std.astype('float32')


def encode_all(model: IVAE, X: 'np.ndarray', train_mean: 'np.ndarray', train_std: 'np.ndarray',
               batch: int = 4096, device: str = DEVICE) -> 'np.ndarray':
    import numpy as np
    Xs = ((X - train_mean) / train_std).astype('float32')
    model.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(Xs), batch):
            xb = torch.from_numpy(Xs[s:s+batch]).to(device)
            out.append(model.encode(xb).cpu().numpy())
    return np.concatenate(out)


def load_exp78_encoder(ckpt_path: 'Path | None' = None, device: str = DEVICE):
    """Load exp78's saved iVAE encoder (used in v34 LB submission). Returns (model, mean, std, valid, centroids)."""
    from pathlib import Path
    import numpy as np
    if ckpt_path is None:
        ckpt_path = Path("/data/birdclef2026/model-weights/ivae_encoder.pt")
    ck = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    in_dim, z_dim, n_aux = int(ck["in_dim"]), int(ck["z_dim"]), int(ck["n_aux"])
    model = IVAE(in_dim, z_dim, n_aux).to(device).eval()
    model.load_state_dict(ck["encoder_state_dict"], strict=False)

    stats = np.load(ckpt_path.parent / "ivae_mel_stats.npz")
    cent = np.load(ckpt_path.parent / "ivae_z_centroids.npz")
    return (model,
            stats["mean"].astype("float32"),
            stats["std"].astype("float32"),
            cent["valid"].astype(bool),
            cent["centroids"].astype("float32"))
