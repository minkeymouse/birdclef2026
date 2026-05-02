"""exp152 — Save M5 MLP probe (h=256, SS+ext×0.1) state_dict for Kaggle inference.

M5 = AudioMAE features → standardize (mu, sd) → MLP (768→256→256→234) →
sigmoid. Per-site spread 0.435 (vs PCA32+LR 0.614). val_SS 0.820.

Saves audiomae_mlp_m5.npz with all weights as numpy. Kaggle inference:
load → standardize → MLP forward → sigmoid (no torch dependency).
"""
import sys
from pathlib import Path
ROOT = Path("/data/birdclef2026")
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings; warnings.filterwarnings("ignore")
from sklearn.metrics import roc_auc_score

from experiments._data_pipelines._shared.data import build_primaries, build_ss_splits
from experiments._data_pipelines._shared.constants import N_CLS

PRIMARY_LABELS, l2i = build_primaries()
ss_train_g, ss_eval_g = build_ss_splits(l2i)
ss_all_lbl = pd.concat([ss_train_g.assign(split="train"), ss_eval_g.assign(split="eval")], ignore_index=True)

ss = np.load(ROOT/"experiments/_data_pipelines/exp143_outputs/audiomae_embs_labeled_ss.npz")
ss_embs = ss["embs"]; ss_split = ss["splits"]
ext = np.load(ROOT/"experiments/_data_pipelines/exp143_outputs/audiomae_external_embs.npz")
ext_embs, ext_species = ext["embs"], ext["species"]

mask_tr = ss_split == "train"; mask_ev = ss_split == "eval"
Y_tr = np.zeros((mask_tr.sum(), N_CLS), dtype=np.float32)
for i, idx in enumerate(np.where(mask_tr)[0]):
    for lbl in ss_all_lbl.iloc[idx].lbls:
        if lbl in l2i: Y_tr[i, l2i[lbl]] = 1.0
Y_ev = np.zeros((mask_ev.sum(), N_CLS), dtype=np.float32)
for i, idx in enumerate(np.where(mask_ev)[0]):
    for lbl in ss_all_lbl.iloc[idx].lbls:
        if lbl in l2i: Y_ev[i, l2i[lbl]] = 1.0
Y_ext = np.zeros((len(ext_embs), N_CLS), dtype=np.float32)
for i, sp in enumerate(ext_species):
    Y_ext[i, l2i[str(sp)]] = 1.0

X_tr = ss_embs[mask_tr]
X_ev = ss_embs[mask_ev]
X_full = np.concatenate([X_tr, ext_embs], axis=0)
Y_full = np.concatenate([Y_tr, Y_ext], axis=0)

# Train M5: SS+ext×0.1, h=256, dropout=0.2
torch.manual_seed(0)
model = nn.Sequential(
    nn.Linear(768, 256), nn.GELU(), nn.Dropout(0.2),
    nn.Linear(256, 256), nn.GELU(),
    nn.Linear(256, N_CLS),
).cuda()
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

mu = X_full.mean(0); sd = X_full.std(0) + 1e-6
Xn = ((X_full - mu) / sd).astype(np.float32)
Xn_ev = ((X_ev - mu) / sd).astype(np.float32)
X = torch.from_numpy(Xn).cuda()
Y = torch.from_numpy(Y_full).cuda()
sw = np.concatenate([np.ones(len(X_tr)), np.full(len(ext_embs), 0.1)]).astype(np.float32)
SW = torch.from_numpy(sw).cuda()

bs = 64; n = len(X)
best_macro = 0; best_state = None; best_ep = 0
for ep in range(80):
    model.train()
    perm = torch.randperm(n, device="cuda")
    for bi in range(0, n, bs):
        idx = perm[bi:bi+bs]
        logits = model(X[idx])
        loss = (F.binary_cross_entropy_with_logits(logits, Y[idx], reduction='none').mean(dim=1) * SW[idx]).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    if (ep+1) % 5 == 0:
        model.eval()
        with torch.no_grad():
            preds = torch.sigmoid(model(torch.from_numpy(Xn_ev).cuda())).cpu().numpy()
        aucs = []
        for c in range(N_CLS):
            yc = Y_ev[:, c]
            if 0 < yc.sum() < len(yc):
                pc = preds[:, c]
                if pc.std() > 1e-9:
                    aucs.append(roc_auc_score(yc, pc))
        m = np.mean(aucs) if aucs else 0
        if m > best_macro:
            best_macro = m; best_state = {k: v.cpu().numpy() for k, v in model.state_dict().items()}; best_ep = ep+1
print(f"Best ep{best_ep} val_SS = {best_macro:.4f}")

# Save weights as plain numpy for easy Kaggle inference
out = {
    "mu": mu.astype(np.float32),
    "sd": sd.astype(np.float32),
}
for k, v in best_state.items():
    out[k.replace(".", "_")] = v.astype(np.float32)
print(f"Saved keys: {list(out.keys())}")
np.savez(ROOT/"model-weights/audiomae_mlp_m5.npz", **out)
print(f"Saved {ROOT}/model-weights/audiomae_mlp_m5.npz ({(ROOT/'model-weights/audiomae_mlp_m5.npz').stat().st_size/1e6:.2f} MB)")
