"""exp143d — Extend AudioMAE probe training set with external (xeno-canto/iNat) clips.

Currently exp143 probe was fit on 55 labeled SS train files (617 windows),
covering only 66/234 classes. The 2,067 external clips at data/external/
cover 77 species (mostly rare taxa); adding them as additional positives
should:
  1. Increase class coverage (probably 100+ classes fit, esp Aves Amph)
  2. Reduce 5-site fingerprint (external = many sites)
  3. Improve probe quality on rare taxa where SS-train alone is sparse

Pipeline:
  1. Iterate over data/external/<species>/*.wav|ogg|mp3|m4a
  2. For each clip, extract AudioMAE 10s-context embedding (center crop)
  3. Combine with existing labeled SS train embeddings
  4. Retrain probe on combined set
  5. Evaluate on labeled SS eval (same as exp143)
  6. Save extended probe as audiomae_probe_v2.npz
"""
import sys
from pathlib import Path
ROOT = Path("/data/birdclef2026")
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import time
import warnings; warnings.filterwarnings("ignore")
import soundfile as sf
import torch
import torchaudio.transforms as T
import timm
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from experiments._data_pipelines._shared.data import build_primaries, build_ss_splits
from experiments._data_pipelines._shared.constants import DATA, N_CLS

DEVICE = "cuda"
EXT_DIR = ROOT / "data" / "external"
OUT = ROOT / "experiments" / "_data_pipelines" / "exp143_outputs"
OUT.mkdir(parents=True, exist_ok=True)

print("[1/6] Loading AudioMAE")
model = timm.create_model('hf_hub:gaunernst/vit_base_patch16_1024_128.audiomae_as2m_ft_as20k', pretrained=True, num_classes=0)
model.eval().to(DEVICE)

mel_op = T.MelSpectrogram(sample_rate=16000, n_fft=1024, hop_length=160,
                           n_mels=128, f_min=0, f_max=8000).to(DEVICE)

@torch.no_grad()
def extract_emb_from_audio_16k(audio_16k):
    """audio_16k: numpy (N,) at 16kHz. Pad/crop to 10s. Return (768,)."""
    if len(audio_16k) < 160000:
        audio_16k = np.pad(audio_16k, (0, 160000 - len(audio_16k)))
    elif len(audio_16k) > 160000:
        # center crop
        s = (len(audio_16k) - 160000) // 2
        audio_16k = audio_16k[s:s+160000]
    a = torch.from_numpy(audio_16k).float().to(DEVICE)
    mel = torch.log(mel_op(a) + 1e-6)
    if mel.shape[-1] < 1024:
        mel = torch.nn.functional.pad(mel, (0, 1024-mel.shape[-1]), value=mel.min())
    mel = mel[:, :1024]
    mel = (mel - mel.mean()) / (mel.std() + 1e-6)
    x = mel.T.unsqueeze(0).unsqueeze(0)  # (1, 1, 1024, 128)
    return model(x).flatten().cpu().numpy()


print("\n[2/6] Indexing external clips")
PRIMARY_LABELS, l2i = build_primaries()
exts = []
for d in sorted(EXT_DIR.iterdir()):
    if not d.is_dir(): continue
    sp = d.name
    if sp not in l2i:
        continue
    for fp in d.glob("*"):
        if fp.suffix.lower() in [".wav", ".ogg", ".mp3", ".m4a", ".flac"]:
            exts.append((sp, fp))
print(f"  {len(exts)} clips across {len(set(s for s,_ in exts))} species")

print("\n[3/6] Extracting AudioMAE embeddings on external clips")
ext_embs = []
ext_species = []
import librosa
t0 = time.time()
fail = 0
for i, (sp, fp) in enumerate(exts):
    try:
        # Resample to 16k mono
        y, sr_orig = librosa.load(str(fp), sr=16000, mono=True)
        if len(y) < 16000:  # < 1s clip — skip
            fail += 1
            continue
        emb = extract_emb_from_audio_16k(y)
        ext_embs.append(emb)
        ext_species.append(sp)
    except Exception as e:
        fail += 1
        if fail <= 5:
            print(f"  err on {fp.name}: {e}")
    if (i+1) % 200 == 0:
        print(f"    [{i+1}/{len(exts)}] elapsed {time.time()-t0:.1f}s, fail {fail}")
ext_embs = np.stack(ext_embs).astype(np.float32)
print(f"  extracted {len(ext_embs)} clips ({fail} failed) in {time.time()-t0:.1f}s")
np.savez(OUT / "audiomae_external_embs.npz",
         embs=ext_embs, species=np.array(ext_species))


print("\n[4/6] Building combined train set: SS-train + external")
# Load existing SS embeddings
ss = np.load(OUT / "audiomae_embs_labeled_ss.npz")
ss_embs = ss["embs"]
ss_split = ss["splits"]
ss_row_ids = ss["row_ids"]

# SS train mask
ss_train_g, ss_eval_g = build_ss_splits(l2i)
ss_all = pd.concat([ss_train_g.assign(split="train"), ss_eval_g.assign(split="eval")], ignore_index=True)
mask_tr = ss_split == "train"
mask_ev = ss_split == "eval"

# Y on SS train: multi-label
Y_ss_tr = np.zeros((mask_tr.sum(), N_CLS), dtype=np.float32)
for i, lbls in enumerate(ss_all[mask_tr].lbls):
    for lbl in lbls:
        if lbl in l2i:
            Y_ss_tr[i, l2i[lbl]] = 1.0

# Y on external: one-hot per clip
Y_ext = np.zeros((len(ext_embs), N_CLS), dtype=np.float32)
for i, sp in enumerate(ext_species):
    Y_ext[i, l2i[sp]] = 1.0

# Combine
X_tr_combined = np.concatenate([ss_embs[mask_tr], ext_embs], axis=0)
Y_tr_combined = np.concatenate([Y_ss_tr, Y_ext], axis=0)
print(f"  combined train: X {X_tr_combined.shape}, Y total positives {Y_tr_combined.sum():.0f}")
print(f"  classes with >=1 positive: {(Y_tr_combined.sum(0) > 0).sum()}/234")


print("\n[5/6] Probe: PCA32 + per-class LR (R5 recipe)")
X_ev = ss_embs[mask_ev]
Y_ev = np.zeros((mask_ev.sum(), N_CLS), dtype=np.float32)
for i, lbls in enumerate(ss_all[mask_ev].lbls):
    for lbl in lbls:
        if lbl in l2i:
            Y_ev[i, l2i[lbl]] = 1.0

mu = X_tr_combined.mean(0); sd = X_tr_combined.std(0) + 1e-6
Xn_tr = (X_tr_combined - mu) / sd
Xn_ev = (X_ev - mu) / sd
pca = PCA(n_components=32, random_state=0).fit(Xn_tr)
Z_tr = pca.transform(Xn_tr)
Z_ev = pca.transform(Xn_ev)
print(f"  PCA var explained: {pca.explained_variance_ratio_.sum():.3f}")

# Per-class LR — also with class weights to handle external imbalance
preds_ev = np.zeros_like(Y_ev)
lr_w = np.zeros((N_CLS, 32), dtype=np.float32)
lr_b = np.zeros(N_CLS, dtype=np.float32)
lr_valid = np.zeros(N_CLS, dtype=bool)
n_fit = 0
for c in range(N_CLS):
    yc = Y_tr_combined[:, c]
    if yc.sum() < 2 or yc.sum() == len(yc):
        preds_ev[:, c] = float(yc.mean())
        continue
    try:
        clf = LogisticRegression(C=0.25, max_iter=200, solver="liblinear",
                                  class_weight="balanced").fit(Z_tr, yc)
        preds_ev[:, c] = clf.predict_proba(Z_ev)[:, 1]
        lr_w[c] = clf.coef_[0].astype(np.float32)
        lr_b[c] = float(clf.intercept_[0])
        lr_valid[c] = True
        n_fit += 1
    except Exception:
        preds_ev[:, c] = 0.0
print(f"  fit {n_fit} classes (vs original 66)")


print("\n[6/6] val_SS evaluation on labeled SS eval (122 rows, same as exp143)")
aucs = []
taxa_aucs = {t: [] for t in ["Aves", "Amphibia", "Insecta", "Mammalia", "Reptilia"]}
tax = pd.read_csv(DATA / "taxonomy.csv").set_index("primary_label")["class_name"]
for c in range(N_CLS):
    yc = Y_ev[:, c]
    if yc.sum() == 0 or yc.sum() == len(yc): continue
    p = preds_ev[:, c]
    if p.std() < 1e-9: continue
    a = roc_auc_score(yc, p)
    aucs.append(a)
    t = tax.get(PRIMARY_LABELS[c], "Aves")
    if t in taxa_aucs: taxa_aucs[t].append(a)

print(f"\n=== val_SS macro AUC ({len(aucs)} cls) = {np.mean(aucs):.4f}")
print(f"  Original exp143 (SS-train only): val_SS 0.8062 on 31 cls")
print()
for t, lst in taxa_aucs.items():
    if lst:
        print(f"  {t:10s} ({len(lst):2d} cls): AUC {np.mean(lst):.4f}")

# Save extended probe
np.savez(OUT / "../../../model-weights/audiomae_probe_v2.npz",
         mu=mu.astype(np.float32), sd=sd.astype(np.float32),
         pca_components=pca.components_.astype(np.float32),
         lr_w=lr_w, lr_b=lr_b, lr_valid=lr_valid)
print(f"\nSaved extended probe (v2) to model-weights/")
