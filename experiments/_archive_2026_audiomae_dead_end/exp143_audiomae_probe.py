"""exp143 — AudioMAE foundation extraction + 234-class linear probe.

First foundation swap experiment. AudioMAE-Base-FT (timm
hf_hub:gaunernst/vit_base_patch16_1024_128.audiomae_as2m_ft_as20k) is
self-supervised pretrained on AudioSet 2M then fine-tuned on AudioSet
20k — completely independent of our 5-site labeled SS, like Perch but
with non-bird-biased AudioSet corpus.

Pipeline:
  1. Extract AudioMAE 768-d CLS embedding on labeled SS (66 files × 12
     windows = 792 windows, 10s context centered on each 5s window).
  2. Use _shared.build_ss_splits — 55 train files / 11 eval files.
  3. Train logistic regression probe (PCA32 + LR C=0.25, matches our
     R5 recipe for Perch).
  4. Compute val_SS on eval split, per-taxon AUC, compare with Perch
     baseline 0.838 / exp50 0.838.
  5. If AudioMAE val_SS > 0.5 (meaningful signal), worth Kaggle blend.
"""
import sys
from pathlib import Path
ROOT = Path("/data/birdclef2026")
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
import timm
import torchaudio.transforms as T
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import warnings; warnings.filterwarnings("ignore")

from experiments._data_pipelines._shared.data import build_primaries, build_ss_splits
from experiments._data_pipelines._shared.audio import load_audio
from experiments._data_pipelines._shared.constants import DATA, SR, N_WINDOWS, WINDOW_SAMPLES

DEVICE = "cuda"
OUT = ROOT / "experiments" / "_data_pipelines" / "exp143_outputs"
OUT.mkdir(parents=True, exist_ok=True)

print("[1/5] Loading AudioMAE-Base-FT")
model_id = 'hf_hub:gaunernst/vit_base_patch16_1024_128.audiomae_as2m_ft_as20k'
model = timm.create_model(model_id, pretrained=True, num_classes=0)
model.eval().to(DEVICE)
print(f"  model: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params, dim=768")

# AudioMAE expects log-mel: 10s @ 16kHz, n_fft=1024, hop=160, n_mels=128
mel_op = T.MelSpectrogram(
    sample_rate=16000, n_fft=1024, hop_length=160, n_mels=128,
    f_min=0, f_max=8000,
).to(DEVICE)
resample_op = T.Resample(orig_freq=SR, new_freq=16000).to(DEVICE)

@torch.no_grad()
def extract_window_emb(audio32k_window, context_window32k):
    """audio32k_window: (WINDOW_SAMPLES,) — the 5-sec window we score
       context_window32k: (10s @ 32kHz,) = surrounding 10-sec context"""
    a = torch.from_numpy(context_window32k).float().to(DEVICE)
    a16 = resample_op(a)  # (160000,) at 16kHz
    if a16.shape[0] < 160000:
        a16 = torch.nn.functional.pad(a16, (0, 160000 - a16.shape[0]))
    a16 = a16[:160000]
    mel = torch.log(mel_op(a16) + 1e-6)  # (128, T)
    if mel.shape[-1] < 1024:
        mel = torch.nn.functional.pad(mel, (0, 1024 - mel.shape[-1]), value=mel.min())
    mel = mel[:, :1024]
    mel = (mel - mel.mean()) / (mel.std() + 1e-6)
    x = mel.T.unsqueeze(0).unsqueeze(0)  # (1, 1, 1024, 128)
    f = model(x)
    return f.flatten().cpu().numpy()


print("\n[2/5] Building SS splits")
PRIMARY_LABELS, l2i = build_primaries()
ss_train_g, ss_eval_g = build_ss_splits(l2i)
print(f"  ss_train rows: {len(ss_train_g)}, files: {ss_train_g.filename.nunique()}")
print(f"  ss_eval rows: {len(ss_eval_g)}, files: {ss_eval_g.filename.nunique()}")

# Combine for unified extraction
ss_all = pd.concat([ss_train_g.assign(split="train"), ss_eval_g.assign(split="eval")], ignore_index=True)
print(f"  total: {len(ss_all)} (filename, start, end) windows from {ss_all.filename.nunique()} files")


print("\n[3/5] Extracting AudioMAE embeddings on labeled SS (window-centered 10s context)")
emb_path = OUT / "audiomae_embs_labeled_ss.npz"
if emb_path.exists():
    z = np.load(emb_path)
    embs, splits, row_ids = z["embs"], z["splits"], z["row_ids"]
    print(f"  loaded cached: embs {embs.shape}")
else:
    file_audio_cache = {}
    embs = []
    splits_arr = []
    row_ids = []
    import time; t0 = time.time()
    for i, row in ss_all.iterrows():
        if row.filename not in file_audio_cache:
            audio = load_audio(DATA / "train_soundscapes" / row.filename, 60*SR)
            # ensure 60 sec
            if len(audio) < 60 * SR:
                audio = np.pad(audio, (0, 60*SR - len(audio)))
            file_audio_cache[row.filename] = audio[:60*SR]
        audio = file_audio_cache[row.filename]
        # window: 5s starting at start
        start_sec = pd.to_timedelta(row.start).total_seconds()
        end_sec = pd.to_timedelta(row.end).total_seconds()
        # 10s context centered on the 5s window
        center = (start_sec + end_sec) / 2
        ctx_start_sec = max(0, center - 5)
        ctx_start = int(ctx_start_sec * SR)
        ctx_end = ctx_start + 10*SR
        if ctx_end > len(audio):
            ctx_start = max(0, len(audio) - 10*SR)
            ctx_end = len(audio)
        ctx = audio[ctx_start:ctx_end]
        if len(ctx) < 10*SR:
            ctx = np.pad(ctx, (0, 10*SR - len(ctx)))
        # Score window itself
        w_start = int(start_sec * SR)
        w_end = w_start + WINDOW_SAMPLES
        w = audio[w_start:min(w_end, len(audio))]
        emb = extract_window_emb(w, ctx)
        embs.append(emb)
        splits_arr.append(row.split)
        row_ids.append(row.row_id)
        if (i+1) % 100 == 0:
            print(f"    [{i+1}/{len(ss_all)}] elapsed {time.time()-t0:.1f}s")
    embs = np.stack(embs).astype(np.float32)
    splits = np.array(splits_arr)
    row_ids = np.array(row_ids)
    np.savez(emb_path, embs=embs, splits=splits, row_ids=row_ids)
    print(f"  saved: embs {embs.shape}, time {time.time()-t0:.1f}s")


print("\n[4/5] Building label matrix Y (multi-label binary, shape N x 234)")
# per-row label list from ss_all matched by row_id
row_labels = ss_all.set_index("row_id").lbls.to_dict()
N = len(row_ids)
Y = np.zeros((N, 234), dtype=np.float32)
for i, rid in enumerate(row_ids):
    for lbl in row_labels[rid]:
        if lbl in l2i:
            Y[i, l2i[lbl]] = 1.0
n_pos_per_class = Y.sum(0)
n_evaluable_train = int(((Y[splits=="train"].sum(0) > 0) & (Y[splits=="train"].sum(0) < (splits=="train").sum())).sum())
n_evaluable_eval = int(((Y[splits=="eval"].sum(0) > 0) & (Y[splits=="eval"].sum(0) < (splits=="eval").sum())).sum())
print(f"  Y total positives: {Y.sum():.0f}, evaluable classes: train {n_evaluable_train}, eval {n_evaluable_eval}")


print("\n[5/5] Probe: PCA32 + per-class LR (R5 recipe)")
mask_tr = splits == "train"
mask_ev = splits == "eval"
X_tr, X_ev = embs[mask_tr], embs[mask_ev]
Y_tr, Y_ev = Y[mask_tr], Y[mask_ev]
print(f"  X_tr {X_tr.shape}  X_ev {X_ev.shape}")

# Standardize then PCA
mu = X_tr.mean(0); sd = X_tr.std(0) + 1e-6
X_tr_n = (X_tr - mu) / sd
X_ev_n = (X_ev - mu) / sd
pca = PCA(n_components=32, random_state=0).fit(X_tr_n)
Z_tr = pca.transform(X_tr_n)
Z_ev = pca.transform(X_ev_n)
print(f"  PCA32 explained variance: {pca.explained_variance_ratio_.sum():.3f}")

# Per-class LR (only fit classes with positives in train AND a positive variation in eval)
preds = np.zeros_like(Y_ev)
fitted_classes = []
for c in range(234):
    if Y_tr[:, c].sum() < 1:
        # No positives in train: predict zero
        preds[:, c] = 0
        continue
    if Y_tr[:, c].sum() == len(Y_tr):
        preds[:, c] = 1
        continue
    try:
        clf = LogisticRegression(C=0.25, max_iter=200, solver="liblinear").fit(Z_tr, Y_tr[:, c])
        preds[:, c] = clf.predict_proba(Z_ev)[:, 1]
        fitted_classes.append(c)
    except Exception as e:
        preds[:, c] = 0
print(f"  fit {len(fitted_classes)} classes")

# Per-class AUC on eval (only classes with both pos+neg in eval)
aucs, taxa_aucs = [], {t: [] for t in ["Aves", "Amphibia", "Insecta", "Mammalia", "Reptilia"]}
tax = pd.read_csv(DATA / "taxonomy.csv").set_index("primary_label")["class_name"]
for c in range(234):
    y_c = Y_ev[:, c]
    if y_c.sum() == 0 or y_c.sum() == len(y_c):
        continue
    p_c = preds[:, c]
    if p_c.std() < 1e-9:
        continue
    a = roc_auc_score(y_c, p_c)
    aucs.append(a)
    taxon = tax.get(PRIMARY_LABELS[c], "Aves")
    if taxon in taxa_aucs:
        taxa_aucs[taxon].append(a)

print(f"\n=== AudioMAE PCA32+LR val_SS (eval, {len(aucs)} classes): macro AUC = {np.mean(aucs):.4f} ===")
for t, lst in taxa_aucs.items():
    if lst:
        print(f"  {t} ({len(lst)} cls): AUC {np.mean(lst):.4f}")

# Save preds for blend exploration
np.savez(OUT / "audiomae_probe_preds.npz",
         row_ids=row_ids[mask_ev], preds=preds, Y_ev=Y_ev,
         splits=splits, embs=embs, mu=mu, sd=sd,
         pca_components=pca.components_)
print(f"\nSaved preds + probe to {OUT}")
