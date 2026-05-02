"""exp146 — DANN-style site-adversarial AudioMAE probe.

Hypothesis: AudioMAE local val_SS 0.81 + LB -0.022 = probe layer absorbs
labeled-SS site fingerprint. Test: train probe with adversarial site head
on 23 sites' worth of features (labeled + 2k unlabeled). GRL forces probe
features to be site-invariant.

Pipeline:
  1. Stratified sample: 2000 unlabeled SS files across 23 sites
  2. Extract AudioMAE 10s embeddings (center crop of file, 1 emb per file)
  3. Combine with labeled SS train embeddings (617 win rows from 55 files)
  4. Train MLP probe: shared trunk → species head (BCE on labeled) +
     site head with GRL (CE on labeled+unlabeled, gradient reversed)
  5. Evaluate on labeled SS eval (122 rows)
  6. Compare with C1 baseline (val_SS 0.806)
"""
import sys
from pathlib import Path
ROOT = Path("/data/birdclef2026")
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import warnings; warnings.filterwarnings("ignore")
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
import timm
from sklearn.metrics import roc_auc_score

from experiments._data_pipelines._shared.data import build_primaries, build_ss_splits
from experiments._data_pipelines._shared.audio import load_audio
from experiments._data_pipelines._shared.constants import DATA, SR, N_CLS

DEVICE = "cuda"
OUT = ROOT / "experiments" / "_data_pipelines" / "exp146_outputs"
OUT.mkdir(parents=True, exist_ok=True)

PRIMARY_LABELS, l2i = build_primaries()

# Step 1: stratified sample of unlabeled SS files
print("[1/6] Stratified sample of unlabeled SS files")
labels = pd.read_csv(DATA / "train_soundscapes_labels.csv").drop_duplicates()
labeled_files = set(labels["filename"].unique())
all_files = sorted([p.name for p in (DATA / "train_soundscapes").glob("*.ogg")])
unlabeled_files = [f for f in all_files if f not in labeled_files]
print(f"  unlabeled: {len(unlabeled_files)}")

import re
def site_of(fn):
    m = re.search(r"S(\d+)", fn); return f"S{int(m.group(1)):02d}" if m else "??"

unlabeled_df = pd.DataFrame({"filename": unlabeled_files})
unlabeled_df["site"] = unlabeled_df["filename"].apply(site_of)
print("Site distribution (unlabeled):")
print(unlabeled_df.site.value_counts().sort_index())

# Stratified: aim for ~100 files/site (cap at site availability)
TARGET_PER_SITE = 150
sampled = []
rng = np.random.RandomState(42)
for site, g in unlabeled_df.groupby("site"):
    n = min(TARGET_PER_SITE, len(g))
    sampled.append(g.sample(n=n, random_state=rng).copy())
sample_df = pd.concat(sampled, ignore_index=True)
print(f"\nStratified sample: {len(sample_df)} files")

# Step 2: extract AudioMAE embeddings (center crop)
print("\n[2/6] Loading AudioMAE")
model = timm.create_model('hf_hub:gaunernst/vit_base_patch16_1024_128.audiomae_as2m_ft_as20k', pretrained=True, num_classes=0)
model.eval().to(DEVICE)
mel_op = T.MelSpectrogram(sample_rate=16000, n_fft=1024, hop_length=160, n_mels=128, f_min=0, f_max=8000).to(DEVICE)

@torch.no_grad()
def extract_emb_from_file(path, ctx_start_sec=25):
    """Center 10s of 60s file, AudioMAE emb."""
    audio = load_audio(path, 60*SR)
    if len(audio) < 60*SR:
        audio = np.pad(audio, (0, 60*SR - len(audio)))
    chunk = audio[ctx_start_sec*SR:(ctx_start_sec+10)*SR]
    if len(chunk) < 10*SR: chunk = np.pad(chunk, (0, 10*SR - len(chunk)))
    a = torch.from_numpy(chunk).float().to(DEVICE)
    # 32k → 16k
    a16 = T.Resample(SR, 16000).to(DEVICE)(a)
    if a16.shape[0] < 160000: a16 = F.pad(a16, (0, 160000 - a16.shape[0]))
    a16 = a16[:160000]
    mel = torch.log(mel_op(a16) + 1e-6)
    if mel.shape[-1] < 1024: mel = F.pad(mel, (0, 1024 - mel.shape[-1]), value=mel.min())
    mel = mel[:, :1024]
    mel = (mel - mel.mean()) / (mel.std() + 1e-6)
    x = mel.T.unsqueeze(0).unsqueeze(0)
    return model(x).flatten().cpu().numpy()

cache_path = OUT / "audiomae_unlabeled_embs.npz"
if cache_path.exists():
    z = np.load(cache_path)
    unlab_embs = z["embs"]; unlab_sites = z["sites"]; unlab_files = z["files"]
    print(f"  loaded cached unlabeled embs: {unlab_embs.shape}")
else:
    print(f"\n[3/6] Extracting AudioMAE on {len(sample_df)} unlabeled SS files")
    embs, sites, files = [], [], []
    t0 = time.time()
    for i, row in sample_df.iterrows():
        try:
            emb = extract_emb_from_file(DATA / "train_soundscapes" / row.filename)
            embs.append(emb); sites.append(row.site); files.append(row.filename)
        except Exception as e:
            if i % 200 == 0: print(f"  err on {row.filename}: {e}")
        if (i+1) % 200 == 0:
            print(f"    [{i+1}/{len(sample_df)}] {time.time()-t0:.1f}s")
    unlab_embs = np.stack(embs).astype(np.float32)
    unlab_sites = np.array(sites)
    unlab_files = np.array(files)
    np.savez(cache_path, embs=unlab_embs, sites=unlab_sites, files=unlab_files)
    print(f"  extracted {len(unlab_embs)} unlabeled embs in {time.time()-t0:.1f}s")

# Step 4: combine with labeled SS embeddings
print("\n[4/6] Combine labeled SS train embeddings with unlabeled")
ss = np.load(ROOT / "experiments/_data_pipelines/exp143_outputs/audiomae_embs_labeled_ss.npz")
ss_embs, ss_split, ss_row_ids = ss["embs"], ss["splits"], ss["row_ids"]
ss_train_g, ss_eval_g = build_ss_splits(l2i)
ss_all_lbl = pd.concat([ss_train_g.assign(split="train"), ss_eval_g.assign(split="eval")], ignore_index=True)
ss_all_lbl["site"] = ss_all_lbl["filename"].str.extract(r"(S\d+)")[0].apply(lambda x: f"S{int(x[1:]):02d}")

mask_tr = ss_split == "train"
mask_ev = ss_split == "eval"
lab_embs_tr = ss_embs[mask_tr]
lab_sites_tr = ss_all_lbl[mask_tr.tolist() if isinstance(mask_tr, np.ndarray) else mask_tr.index.isin(np.where(mask_tr)[0])].site.values

# more reliable: use the ss_all_lbl in same order as ss_row_ids
ss_all_indexed = ss_all_lbl.set_index("row_id")
lab_sites_all = np.array([ss_all_indexed.loc[rid].site for rid in ss_row_ids])
lab_sites_tr = lab_sites_all[mask_tr]
lab_sites_ev = lab_sites_all[mask_ev]

# Y for labeled
Y_lab_tr = np.zeros((mask_tr.sum(), N_CLS), dtype=np.float32)
for i, idx in enumerate(np.where(mask_tr)[0]):
    for lbl in ss_all_lbl.iloc[idx].lbls:
        if lbl in l2i: Y_lab_tr[i, l2i[lbl]] = 1.0
Y_lab_ev = np.zeros((mask_ev.sum(), N_CLS), dtype=np.float32)
for i, idx in enumerate(np.where(mask_ev)[0]):
    for lbl in ss_all_lbl.iloc[idx].lbls:
        if lbl in l2i: Y_lab_ev[i, l2i[lbl]] = 1.0

# Site label encoder
all_sites = sorted(set(np.concatenate([lab_sites_all, unlab_sites]).tolist()))
site_to_idx = {s: i for i, s in enumerate(all_sites)}
n_sites = len(all_sites)
print(f"  total sites: {n_sites}: {all_sites}")
S_lab_tr = np.array([site_to_idx[s] for s in lab_sites_tr])
S_lab_ev = np.array([site_to_idx[s] for s in lab_sites_ev])
S_unlab = np.array([site_to_idx[s] for s in unlab_sites])

# Combine for site head training (labeled + unlabeled)
X_for_site = np.concatenate([lab_embs_tr, unlab_embs], axis=0).astype(np.float32)
S_for_site = np.concatenate([S_lab_tr, S_unlab], axis=0)
print(f"  combined for site head: {X_for_site.shape}, sites {len(set(S_for_site))}")

# Standardize on combined features
mu = X_for_site.mean(0); sd = X_for_site.std(0) + 1e-6
def norm(x): return (x - mu) / sd
Xn_lab_tr = norm(lab_embs_tr).astype(np.float32)
Xn_lab_ev = norm(ss_embs[mask_ev]).astype(np.float32)
Xn_for_site = norm(X_for_site).astype(np.float32)


# Step 5: DANN probe model
print("\n[5/6] Training DANN probe")

class GradReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    @staticmethod
    def backward(ctx, g):
        return -ctx.alpha * g, None

class DANNProbe(nn.Module):
    def __init__(self, in_dim=768, trunk_dim=128, n_cls=234, n_sites=23):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, trunk_dim), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(trunk_dim, trunk_dim), nn.GELU(),
        )
        self.cls_head = nn.Linear(trunk_dim, n_cls)
        self.site_head = nn.Sequential(nn.Linear(trunk_dim, 64), nn.GELU(), nn.Linear(64, n_sites))
    def forward(self, x, alpha):
        f = self.trunk(x)
        cls_logits = self.cls_head(f)
        site_logits = self.site_head(GradReversal.apply(f, alpha))
        return cls_logits, site_logits

def fit_dann(alpha_max=0.3, epochs=100, lr=3e-4, name="dann"):
    torch.manual_seed(0)
    model = DANNProbe(768, 128, N_CLS, n_sites).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    X_lab = torch.from_numpy(Xn_lab_tr).to(DEVICE)
    Y_lab = torch.from_numpy(Y_lab_tr).to(DEVICE)
    S_lab = torch.from_numpy(S_lab_tr).long().to(DEVICE)
    X_site_all = torch.from_numpy(Xn_for_site).to(DEVICE)
    S_site_all = torch.from_numpy(S_for_site).long().to(DEVICE)

    n_lab = len(X_lab); n_all = len(X_site_all)
    bs = 64
    best_val_macro = 0
    best_preds = None
    for ep in range(epochs):
        # Linear ramp-up of alpha (DANN paper)
        p = ep / max(1, epochs - 1)
        alpha = alpha_max * (2.0 / (1.0 + np.exp(-10*p)) - 1.0)
        model.train()
        perm = torch.randperm(n_lab, device=DEVICE)
        site_perm = torch.randperm(n_all, device=DEVICE)[:n_lab]  # match labeled batch size
        ep_loss_cls, ep_loss_site = 0, 0
        for bi in range(0, n_lab, bs):
            idx = perm[bi:bi+bs]
            x = X_lab[idx]; y = Y_lab[idx]
            s_idx = site_perm[bi:bi+bs]
            xs = X_site_all[s_idx]; ys = S_site_all[s_idx]
            x_combined = torch.cat([x, xs], 0)
            cls_logits, site_logits = model(x_combined, alpha)
            cls_lab = cls_logits[:len(x)]
            site_unlab = site_logits  # all
            loss_cls = F.binary_cross_entropy_with_logits(cls_lab, y)
            loss_site = F.cross_entropy(site_logits, torch.cat([S_lab[idx], ys], 0))
            loss = loss_cls + loss_site
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss_cls += loss_cls.item(); ep_loss_site += loss_site.item()
        # Eval
        if (ep+1) % 10 == 0 or ep == epochs - 1:
            model.eval()
            with torch.no_grad():
                xev = torch.from_numpy(Xn_lab_ev).to(DEVICE)
                cls_logits, _ = model(xev, 0.0)
                preds = torch.sigmoid(cls_logits).cpu().numpy()
            aucs = []
            for c in range(N_CLS):
                yc = Y_lab_ev[:, c]
                if 0 < yc.sum() < len(yc):
                    pc = preds[:, c]
                    if pc.std() > 1e-9:
                        aucs.append(roc_auc_score(yc, pc))
            macro = np.mean(aucs) if aucs else 0
            print(f"  ep{ep+1:3d} α={alpha:.3f} loss_cls={ep_loss_cls/(n_lab/bs):.4f} loss_site={ep_loss_site/(n_lab/bs):.4f}  val_SS={macro:.4f} ({len(aucs)} cls)")
            if macro > best_val_macro:
                best_val_macro = macro
                best_preds = preds.copy()
    return best_val_macro, best_preds

# Run with several alphas
results = {}
for alpha in [0.0, 0.05, 0.1, 0.3, 0.5]:
    print(f"\n--- α_max = {alpha}")
    m, p = fit_dann(alpha_max=alpha, epochs=60, name=f"a{alpha}")
    results[alpha] = (m, p)

print("\n=== Summary ===")
for a, (m, _) in sorted(results.items()):
    print(f"  α={a:.2f}: val_SS = {m:.4f}")

print(f"\nBaseline reference: AudioMAE C1 (PCA32+LR no DANN) val_SS=0.8062")

# Save best preds
best_a = max(results, key=lambda k: results[k][0])
np.savez(OUT / "dann_best_preds.npz", preds=results[best_a][1], alpha=best_a, val_SS=results[best_a][0])
print(f"\nSaved best DANN preds (α={best_a}, val_SS={results[best_a][0]:.4f})")
