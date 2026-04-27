#!/usr/bin/env python3
"""exp80d — Integration test on 122 held-out eval rows.

Builds the EXACT same final-test pipeline as the Kaggle perch-distill
notebook (Perch + exp50 SED50 blend + V9-like taxon gate + file-max coherence)
locally on labeled SS to test whether iVAE-augmented taxon gate improves
macro-AUC.

Conditions:
  base       = 0.7 * sigmoid(Perch_score) + 0.3 * sigmoid(exp50)        (v26 backbone)
  v33        = base + file_max coherence α=0.10
  v33+gate_P = v33 with Perch-only taxon gate (offset 0.1) — current production
  v33+gate_PiV_small = v33 with (Perch + small-iVAE-z) taxon gate
  v33+gate_PiV_big   = v33 with (Perch + big-iVAE-z) taxon gate

Reports per-class AUC averaged over 40 evaluable classes + per-taxon Δ.

If gate_PiV beats gate_P measurably on Insecta/Mammalia/Reptilia AND keeps
Aves stable, we have a viable lever (subject to LB anti-correlation).
"""
from __future__ import annotations
import re
from pathlib import Path
import numpy as np
import pandas as pd
import torch, torch.nn as nn
import soundfile as sf, torchaudio
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

ROOT = Path("/data/birdclef2026")
DATA = ROOT / "data/birdclef-2026"
EXP76 = ROOT / "experiments/_audits_post_v26/exp76_outputs"
EXP43A = ROOT / "experiments/_data_pipelines/exp43a_outputs"
EXP80 = ROOT / "experiments/_audits_post_v26/exp80_outputs"
MW = ROOT / "model-weights"
DEVICE = "cuda"; SEED = 42; N_CLS = 234
SR = 32000
TAXA = ["Aves", "Amphibia", "Insecta", "Mammalia", "Reptilia"]

FN_RE = re.compile(r"BC2026_(?:Train|Test)_\d+_(S\d+)_(\d{8})_(\d{6})")


def build_ss():
    sc = pd.read_csv(DATA / "train_soundscapes_labels.csv").drop_duplicates().reset_index(drop=True)
    primary = pd.read_csv(DATA / "sample_submission.csv").columns[1:].tolist()
    def parse(x):
        if pd.isna(x): return []
        return [t.strip() for t in str(x).split(";") if t.strip()]
    sc_g = (sc.groupby(["filename","start","end"])["primary_label"]
            .apply(lambda s: sorted({l for x in s for l in parse(x)})).reset_index(name="lbls"))
    sc_g["end_sec"] = pd.to_timedelta(sc_g["end"]).dt.total_seconds().astype(int)
    sc_g["row_id"] = sc_g["filename"].str.replace(".ogg","",regex=False) + "_" + sc_g["end_sec"].astype(str)
    sc_g[["site","hour"]] = sc_g.filename.apply(lambda f: pd.Series([FN_RE.match(f).group(1), int(FN_RE.match(f).group(2)[:2])]))
    rng = np.random.RandomState(SEED); files = sorted(sc_g.filename.unique()); rng.shuffle(files)
    eval_files = set(files[:11])
    sc_g["split"] = ["eval" if f in eval_files else "train" for f in sc_g.filename]
    l2i = {p: i for i, p in enumerate(primary)}
    Y = np.zeros((len(sc_g), len(primary)), dtype=np.uint8)
    for i, labs in enumerate(sc_g.lbls):
        for l in labs:
            if l in l2i: Y[i, l2i[l]] = 1
    return sc_g, Y, primary, l2i


def get_perch(sc_g):
    perch_emb = np.load(EXP43A / "perch_ss_all.npz")
    perch_meta = pd.read_parquet(EXP43A / "perch_ss_all_meta.parquet")
    rid2i = {r: i for i, r in enumerate(perch_meta["row_id"].values)}
    sc = np.zeros((len(sc_g), N_CLS), dtype=np.float32)
    em = np.zeros((len(sc_g), 1536), dtype=np.float32)
    for i, rid in enumerate(sc_g.row_id.values):
        j = rid2i.get(rid, -1)
        if j >= 0:
            sc[i] = perch_emb["scores"][j]
            em[i] = perch_emb["emb"][j]
    return sc, em


def get_smallpool_z():
    ck = torch.load(MW / "ivae_encoder.pt", map_location=DEVICE, weights_only=False)
    stats = np.load(MW / "ivae_mel_stats.npz")
    train_mean = stats["mean"].astype(np.float32)
    train_std = stats["std"].astype(np.float32)
    in_dim = int(ck["in_dim"]); z_dim = int(ck["z_dim"]); n_aux = int(ck["n_aux"])
    class Enc(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = nn.Sequential(
                nn.Linear(in_dim, 512), nn.GELU(), nn.Dropout(0.1),
                nn.Linear(512, 256), nn.GELU(), nn.Linear(256, 2 * z_dim))
            self.aux_mlp = nn.Sequential(
                nn.Linear(n_aux, 64), nn.GELU(), nn.Linear(64, 2 * z_dim))
        def encode(self, x):
            h = self.enc(x); mu, _ = h.chunk(2, -1); return mu
    enc = Enc().to(DEVICE).eval()
    enc.load_state_dict(ck["encoder_state_dict"], strict=False)
    mel = np.load(EXP76 / "mel_cache.npz")["mel"]
    X = mel.reshape(len(mel), -1).astype(np.float32)
    X = (X - train_mean) / train_std
    with torch.no_grad():
        Z = enc.encode(torch.from_numpy(X).to(DEVICE)).cpu().numpy()
    return Z


def get_bigpool_z():
    p = EXP80 / "bigpool_z.npz"
    if not p.exists(): return None
    return np.load(p)["Z_lab"]


def run_exp50_inference(sc_g):
    """Produce exp50 SED scores per row (returns sigmoid probabilities, (n, 234))."""
    cache_path = EXP80 / "exp50_scores_labeled.npz"
    if cache_path.exists():
        return np.load(cache_path)["scores"]
    print("  no cached exp50 scores — running inference (~3 min)")
    import timm

    SED_N_MELS, SED_N_FFT, SED_HOP = 128, 2048, 512
    SED_FMIN, SED_FMAX = 50, 14000
    SED_CHUNK_SEC = 20
    SED_CHUNK_SAMPLES = SR * SED_CHUNK_SEC

    class _MelExtractor(nn.Module):
        def __init__(self):
            super().__init__()
            self.mel = torchaudio.transforms.MelSpectrogram(
                sample_rate=SR, n_fft=SED_N_FFT, hop_length=SED_HOP, n_mels=SED_N_MELS,
                f_min=SED_FMIN, f_max=SED_FMAX, power=2.0, center=True)
            self.adb = torchaudio.transforms.AmplitudeToDB(stype='power', top_db=80)
        def forward(self, x):
            m = self.mel(x); m = self.adb(m); return m.unsqueeze(1)

    class _SEDHead(nn.Module):
        def __init__(self, feat_dim, n_classes):
            super().__init__()
            self.att = nn.Conv1d(feat_dim, n_classes, 1)
            self.cla = nn.Conv1d(feat_dim, n_classes, 1)
        def forward(self, x):
            a = self.att(x); c = self.cla(x)
            w = torch.softmax(a, dim=-1)
            return (w * c).sum(-1), c.max(-1).values

    class _SEDModel(nn.Module):
        def __init__(self, backbone='hgnetv2_b0.ssld_stage2_ft_in1k'):
            super().__init__()
            self.mel = _MelExtractor()
            self.bn0 = nn.BatchNorm2d(SED_N_MELS)
            self.backbone = timm.create_model(backbone, pretrained=False, in_chans=1, num_classes=0, global_pool='')
            with torch.no_grad():
                feat = self.backbone(torch.zeros(1, 1, SED_N_MELS, 100))
            self.head = _SEDHead(feat.shape[1], N_CLS)
        def forward(self, x):
            m = self.mel(x); m = m.transpose(1, 2); m = self.bn0(m); m = m.transpose(1, 2)
            feat = self.backbone(m); feat = feat.mean(dim=2) if feat.dim() == 4 else feat
            clip, _ = self.head(feat)
            return clip

    st = torch.load(MW / "exp50_hgnet_sed.pt", map_location=DEVICE, weights_only=False)
    bb = st.get('backbone', 'hgnetv2_b0.ssld_stage2_ft_in1k')
    model = _SEDModel(bb).to(DEVICE).eval()
    model.load_state_dict(st['state_dict'])

    files = sorted(sc_g.filename.unique())
    SS_DIR = DATA / "train_soundscapes"
    fname_idx = {f: [] for f in files}
    for i, r in sc_g.iterrows():
        fname_idx[r.filename].append((i, int(r.end_sec)))

    scores = np.zeros((len(sc_g), N_CLS), dtype=np.float32)
    BATCH_F = 8
    with torch.inference_mode():
        for start in range(0, len(files), BATCH_F):
            batch = files[start:start + BATCH_F]
            chunks = []
            chunk_meta = []  # (file_idx, chunk_idx)
            for bi, fn in enumerate(batch):
                y, _ = sf.read(SS_DIR / fn, dtype="float32", always_2d=False)
                if y.ndim == 2: y = y.mean(axis=1)
                if len(y) < 60 * SR: y = np.pad(y, (0, 60 * SR - len(y)))
                y = y[:60 * SR]
                # 3 chunks of 20s each
                for ci in range(3):
                    s = ci * SED_CHUNK_SAMPLES
                    chunks.append(y[s:s + SED_CHUNK_SAMPLES])
                    chunk_meta.append((bi, ci))
            x = torch.from_numpy(np.stack(chunks)).to(DEVICE)
            clip = model(x)
            p = torch.sigmoid(clip).cpu().numpy()
            for k, (bi, ci) in enumerate(chunk_meta):
                fn = batch[bi]
                for row_idx, end_sec in fname_idx[fn]:
                    # row's window (5s) belongs to chunk_idx = (end_sec - 5) // 20
                    win_idx = (end_sec - 5) // 5
                    chunk_for_row = win_idx // 4
                    if chunk_for_row == ci:
                        scores[row_idx] = p[k]
    np.savez_compressed(cache_path, scores=scores)
    print(f"  cached → {cache_path}")
    return scores


def main():
    print("=== exp80d: integration test ===\n")
    sc_g, Y, primary, l2i = build_ss()
    tax = pd.read_csv(DATA / "taxonomy.csv")
    l2t = dict(zip(tax.primary_label.astype(str), tax.class_name))
    species_taxon = np.array([l2t.get(p, "?") for p in primary])
    species_to_taxon_idx = np.array([TAXA.index(t) if t in TAXA else 0 for t in species_taxon])

    # ================== Features ==================
    P_score, P_emb = get_perch(sc_g)
    P_prob = 1.0 / (1.0 + np.exp(-P_score))
    print(f"Perch: scores {P_score.shape}, emb {P_emb.shape}")

    print("Loading exp50 scores...")
    S50 = run_exp50_inference(sc_g)
    print(f"exp50: {S50.shape}")

    Z_small = get_smallpool_z()
    Z_big = get_bigpool_z()
    print(f"iVAE z: small {Z_small.shape}, big {Z_big.shape if Z_big is not None else None}")

    tr_mask = sc_g.split.values == "train"
    ev_mask = sc_g.split.values == "eval"

    # ================== Build base predictions ==================
    # v26-style: 0.7 * sigmoid(Perch) + 0.3 * exp50 (in z-space, but here use simple sigmoid-blend)
    base = 0.7 * P_prob + 0.3 * S50

    # v33: file-max coherence
    def file_max_blend(probs, sc_g, alpha=0.10):
        """For each file group, blend each window with the file's per-class max."""
        out = probs.copy()
        for fname, idx in sc_g.groupby("filename").indices.items():
            sub = probs[idx]
            fmax = sub.max(axis=0, keepdims=True)
            out[idx] = (1 - alpha) * sub + alpha * fmax
        return out

    v33 = file_max_blend(base, sc_g, alpha=0.10)

    # ================== Train taxon classifiers ==================
    def fit_taxon_clf(X_feat):
        """Fit per-taxon LogReg on train rows. Returns (n_eval_rows, 5) prob array."""
        out_prob = np.zeros((len(sc_g), 5), dtype=np.float32)
        for t_idx, t in enumerate(TAXA):
            cols = species_taxon == t
            y = (Y[:, cols].sum(axis=1) > 0).astype(np.uint8)
            if y[tr_mask].sum() < 3:
                out_prob[:, t_idx] = 0.5
                continue
            clf = LogisticRegression(max_iter=500, C=1.0, class_weight="balanced", random_state=SEED)
            clf.fit(X_feat[tr_mask], y[tr_mask])
            out_prob[:, t_idx] = clf.predict_proba(X_feat)[:, 1]
        return out_prob

    print("\nTraining taxon classifiers (LogReg, balanced)...")
    feature_sets = {
        "P":     P_emb,
        "P+iVS": np.concatenate([P_emb, Z_small], axis=1),
    }
    if Z_big is not None:
        feature_sets["P+iVB"] = np.concatenate([P_emb, Z_big], axis=1)

    taxon_probs = {name: fit_taxon_clf(X) for name, X in feature_sets.items()}

    # ================== V9-style gate apply ==================
    def apply_gate(probs, taxon_prob, offset=0.1):
        gate = np.clip(taxon_prob[:, species_to_taxon_idx] + offset, 0.0, 1.0)
        return probs * gate

    candidates = {"v33 (no gate)": v33}
    for name, tp in taxon_probs.items():
        candidates[f"v33 + gate({name})"] = apply_gate(v33, tp, offset=0.1)
        candidates[f"v33 + gate({name}, off=0.2)"] = apply_gate(v33, tp, offset=0.2)

    # ================== Eval per-class AUC ==================
    print(f"\n=== Per-condition macro AUC + per-taxon Δ on 122 eval rows ===")
    Y_ev = Y[ev_mask]
    ref = candidates["v33 (no gate)"][ev_mask]

    def macro_auc(P, label):
        common = []
        per_taxon = {t: [] for t in TAXA}
        for c in range(N_CLS):
            n_pos = Y_ev[:, c].sum()
            if n_pos == 0 or n_pos == Y_ev.shape[0]: continue
            try:
                a = roc_auc_score(Y_ev[:, c], P[:, c])
                common.append(a)
                if species_taxon[c] in per_taxon: per_taxon[species_taxon[c]].append(a)
            except: pass
        return np.mean(common), {t: np.mean(v) if v else np.nan for t, v in per_taxon.items()}, len(common)

    base_auc, base_tax, n_eval = macro_auc(ref, "v33")
    print(f"\n  Reference: v33 (no gate)  macro={base_auc:.4f} ({n_eval} eval cols)")
    print(f"    {' / '.join(f'{t}={base_tax[t]:.3f}' for t in TAXA if not np.isnan(base_tax[t]))}")

    print(f"\n  {'condition':<32} {'macro':>8} {'Δ':>8} {'Aves':>7} {'Amph':>7} {'Insc':>7} {'Mamm':>7}")
    for name, P_full in candidates.items():
        if name == "v33 (no gate)": continue
        P_ev = P_full[ev_mask]
        m, t, _ = macro_auc(P_ev, name)
        # Per-taxon delta
        d_aves = (t["Aves"] - base_tax["Aves"]) if not np.isnan(t["Aves"]) and not np.isnan(base_tax["Aves"]) else np.nan
        d_amph = (t["Amphibia"] - base_tax["Amphibia"]) if not np.isnan(t["Amphibia"]) and not np.isnan(base_tax["Amphibia"]) else np.nan
        d_insc = (t["Insecta"] - base_tax["Insecta"]) if not np.isnan(t["Insecta"]) and not np.isnan(base_tax["Insecta"]) else np.nan
        d_mamm = (t["Mammalia"] - base_tax["Mammalia"]) if not np.isnan(t["Mammalia"]) and not np.isnan(base_tax["Mammalia"]) else np.nan
        print(f"  {name:<32} {m:>8.4f} {m - base_auc:>+8.4f} {d_aves:>+7.4f} {d_amph:>+7.4f} {d_insc:>+7.4f} {d_mamm:>+7.4f}")

    # Spearman per-row vs v33
    from scipy.stats import spearmanr
    print("\n  per-row Spearman vs v33 (transfer safety):")
    for name, P_full in candidates.items():
        if name == "v33 (no gate)": continue
        sps = []
        for i in range(ev_mask.sum()):
            r, _ = spearmanr(ref[i], P_full[ev_mask][i])
            if np.isfinite(r): sps.append(r)
        print(f"    {name:<32}: {np.mean(sps):.4f}")


if __name__ == "__main__":
    main()
