"""exp143f — D: AudioMAE+v33 blend combinations + Path 1 freeze + ensemble probes.

Compare:
  1. v33 + α*C1 (original probe = currently submitted v49)
  2. v33 + α*C7 (no PCA, balanced)
  3. v33 + α*C5 (SS+ext w_ext=0.1, balanced)
  4. v33 + α*0.5(C5+C7) (probe ensemble)
  5. (1-4) × non-Aves freeze (preserve v33 Aves cols at v33)
  6. v33 + α*AM + β*exp84b (multi-teacher: foundation + ext-finetune)
"""
import sys
from pathlib import Path
ROOT = Path("/data/birdclef2026")
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import warnings; warnings.filterwarnings("ignore")
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

from experiments._data_pipelines._shared.data import build_primaries, build_ss_splits
from experiments._data_pipelines._shared.constants import DATA, N_CLS

PRIMARY_LABELS, l2i = build_primaries()
ss_train_g, ss_eval_g = build_ss_splits(l2i)
ss_all = pd.concat([ss_train_g.assign(split="train"), ss_eval_g.assign(split="eval")], ignore_index=True)

OUT = ROOT / "experiments" / "_data_pipelines" / "exp143_outputs"
ss = np.load(OUT / "audiomae_embs_labeled_ss.npz")
ss_embs, ss_split = ss["embs"], ss["splits"]
ext = np.load(OUT / "audiomae_external_embs.npz")
ext_embs, ext_species = ext["embs"], ext["species"]
mask_tr = ss_split == "train"; mask_ev = ss_split == "eval"

# Y
def build_Y(rows, lbls_col):
    Y = np.zeros((len(rows), N_CLS), dtype=np.float32)
    for i, lbls in enumerate(rows[lbls_col]):
        for lbl in lbls:
            if lbl in l2i: Y[i, l2i[lbl]] = 1.0
    return Y
Y_ss_tr = build_Y(ss_all[mask_tr], "lbls")
Y_ev = build_Y(ss_all[mask_ev], "lbls")
Y_ext = np.zeros((len(ext_embs), N_CLS), dtype=np.float32)
for i, sp in enumerate(ext_species):
    Y_ext[i, l2i[str(sp)]] = 1.0

X_ss = ss_embs[mask_tr]; X_ev = ss_embs[mask_ev]
X_full = np.concatenate([X_ss, ext_embs], axis=0)
Y_full = np.concatenate([Y_ss_tr, Y_ext], axis=0)

# Load v33 base preds
perch = np.load("/data/birdclef2026/experiments/_audits_post_v26/exp80_outputs/perch_prob_labeled.npz")["prob"]
exp50 = np.load("/data/birdclef2026/experiments/_audits_post_v26/exp80_outputs/exp50_scores_labeled.npz")["scores"]
exp84b = np.load("/data/birdclef2026/experiments/_audits_post_v26/exp80_outputs/exp84b_scores_labeled.npz")["scores"]
v33_simple_ev = (0.7 * perch + 0.3 * exp50)[mask_ev]
exp84b_ev = exp84b[mask_ev]
print(f"v33_simple_ev shape {v33_simple_ev.shape}")

tax = pd.read_csv(DATA / "taxonomy.csv").set_index("primary_label")["class_name"]
is_aves = np.array([tax.get(p, 'Aves') == 'Aves' for p in PRIMARY_LABELS])

def fit_probe(X_tr, Y_tr, n_pca=32, C=0.25, sample_weights=None, balanced=True):
    mu = X_tr.mean(0); sd = X_tr.std(0) + 1e-6
    Xn = (X_tr - mu) / sd
    Xn_ev = (X_ev - mu) / sd
    if n_pca and n_pca < min(Xn.shape):
        pca = PCA(n_components=n_pca, random_state=0).fit(Xn)
        Z = pca.transform(Xn); Z_ev = pca.transform(Xn_ev)
    else:
        Z, Z_ev = Xn, Xn_ev
    preds_ev = np.zeros((len(X_ev), N_CLS), dtype=np.float32)
    cw = "balanced" if balanced else None
    for c in range(N_CLS):
        yc = Y_tr[:, c]
        if yc.sum() < 2 or yc.sum() == len(yc):
            preds_ev[:, c] = float(yc.mean())
            continue
        try:
            clf = LogisticRegression(C=C, max_iter=300, solver="liblinear", class_weight=cw)
            if sample_weights is not None:
                clf.fit(Z, yc, sample_weight=sample_weights)
            else:
                clf.fit(Z, yc)
            preds_ev[:, c] = clf.predict_proba(Z_ev)[:, 1]
        except Exception:
            preds_ev[:, c] = 0.0
    return preds_ev

print("Fitting probes...")
P_C1 = fit_probe(X_ss, Y_ss_tr, n_pca=32, balanced=False)  # exp143 original
P_C5 = fit_probe(X_full, Y_full, n_pca=32, balanced=True,
                 sample_weights=np.concatenate([np.ones(len(X_ss)), np.full(len(ext_embs), 0.1)]))
P_C7 = fit_probe(X_ss, Y_ss_tr, n_pca=0, balanced=True)  # no PCA
print(f"  C1, C5, C7 probes ready")

def eval_macro(preds, name=""):
    aucs = []
    taxa = {t: [] for t in ["Aves","Amphibia","Insecta","Mammalia","Reptilia"]}
    for c in range(N_CLS):
        yc = Y_ev[:, c]
        if yc.sum() == 0 or yc.sum() == len(yc): continue
        sc = preds[:, c]
        if sc.std() < 1e-9: continue
        a = roc_auc_score(yc, sc)
        aucs.append(a)
        t = tax.get(PRIMARY_LABELS[c], "Aves")
        if t in taxa: taxa[t].append(a)
    m = np.mean(aucs) if aucs else 0
    print(f"  [{name:50s}] {len(aucs)} cls macro={m:.4f}", end="")
    for tn in ["Aves","Amphibia","Insecta","Mammalia","Reptilia"]:
        if taxa[tn]: print(f"  {tn[:3]}={np.mean(taxa[tn]):.3f}", end="")
    print()
    return m

m_v33 = eval_macro(v33_simple_ev, "v33 simple base")
print()

# v33 + α·probe sweep for each probe
for tag, P in [("C1", P_C1), ("C5", P_C5), ("C7", P_C7), ("0.5(C5+C7)", 0.5*(P_C5+P_C7))]:
    print(f"\n--- v33 + α × {tag}")
    for a in [0.05, 0.10, 0.15, 0.20]:
        b = (1-a) * v33_simple_ev + a * P
        eval_macro(b, f"v33 + {a}·{tag} uniform")

# Non-Aves freeze
print("\n--- v33 + α × C7 NON-Aves only (Path 1 + AM)")
for a in [0.10, 0.20, 0.30]:
    w_per = np.where(is_aves, 0.0, a).astype(np.float32)
    b = (1 - w_per[None]) * v33_simple_ev + w_per[None] * P_C7
    eval_macro(b, f"v33 + {a}·C7 non-Aves")

# Multi-teacher: v33 + AM + exp84b
print("\n--- Multi-teacher: v33 + α·C7 + β·exp84b (different mechanism teachers)")
for a in [0.10, 0.15]:
    for b_w in [0.05, 0.10]:
        m = (1 - a - b_w) * v33_simple_ev + a * P_C7 + b_w * exp84b_ev
        eval_macro(m, f"v33 + {a}·C7 + {b_w}·exp84b")

# Aves-W + non-Aves split (W_PERCH varies by taxon)
print("\n--- Per-taxon AudioMAE dose (Aves: 0.05, non-Aves: 0.20)")
w_per = np.where(is_aves, 0.05, 0.20).astype(np.float32)
b = (1 - w_per[None]) * v33_simple_ev + w_per[None] * P_C7
eval_macro(b, "v33 + per-taxon dose (C7)")
