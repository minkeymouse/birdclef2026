#!/usr/bin/env python3
"""exp81 — Comprehensive local lever audit on 122 held-out eval rows.

Tests 5 candidate post-hoc levers we haven't sufficiently measured locally,
ranks them by predicted LB transferability using anti-correlation heuristics
(prefer Aves Δ ≥ 0, sp_row ≥ 0.99, modest macro Δ, and universal mechanism).

Pipeline:
  base v33 = v26 (0.7 sigmoid(Perch) + 0.3 exp50 sigmoid) + V9 taxon gate +
             file-max coherence α=0.10
  candidates:
    L1: logit-level 1D Kalman smoothing across 12 windows/file (universal physics)
    L2: logit-level HMM 2-state (present/absent, universal transition prior)
    L3: cross-model confidence masking (Perch + exp50 agreement → confidence)
    L4: file-level feature isotonic per-class (max/mean/std/p95, universal aggregation)
    L5: per-class Platt scaling fit on train OOF, applied on eval (mild calibration)

Reports per candidate: (macro Δ vs v33, sp_row, Aves Δ, Insecta Δ, predicted_LB_class).

Predicted LB class:
  A (likely positive)  : sp_row ≥ 0.99 AND Aves Δ ≥ 0 AND universal mechanism
  B (likely neutral)   : sp_row ≥ 0.99 AND |Δ| ≤ 0.005
  C (likely negative)  : sp_row < 0.99 OR Aves Δ < 0
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch, torch.nn as nn
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))
from _lib.data import (build_ss, species_taxon_array, load_perch_emb_labeled,
                        load_perch_scores_labeled, DATA, MW, EXP80, N_CLS, SR, TAXA)
from _lib.eval_metrics import per_class_auc, macro_auc, per_taxon_macro, per_row_spearman

# ------------------------------------------------------------------ exp50 inference cache
def get_exp50_scores(sc_g) -> np.ndarray:
    cache = EXP80 / "exp50_scores_labeled.npz"
    if cache.exists():
        return np.load(cache)["scores"]
    print("  building exp50 scores cache (one-time, ~3 min)...", flush=True)
    import torchaudio, timm
    SED_N_MELS, SED_N_FFT, SED_HOP = 128, 2048, 512
    SED_FMIN, SED_FMAX = 50, 14000
    SED_CHUNK_SEC = 20
    SED_CHUNK_SAMPLES = SR * SED_CHUNK_SEC

    class _MelExt(nn.Module):
        def __init__(self):
            super().__init__()
            self.mel = torchaudio.transforms.MelSpectrogram(
                sample_rate=SR, n_fft=SED_N_FFT, hop_length=SED_HOP, n_mels=SED_N_MELS,
                f_min=SED_FMIN, f_max=SED_FMAX, power=2.0, center=True)
            self.adb = torchaudio.transforms.AmplitudeToDB(stype='power', top_db=80)
        def forward(self, x): return self.adb(self.mel(x)).unsqueeze(1)

    class _SEDHead(nn.Module):
        def __init__(self, fd, nc):
            super().__init__()
            self.att = nn.Conv1d(fd, nc, 1); self.cla = nn.Conv1d(fd, nc, 1)
        def forward(self, x):
            a, c = self.att(x), self.cla(x)
            return (torch.softmax(a, dim=-1) * c).sum(-1), c.max(-1).values

    class _SED(nn.Module):
        def __init__(self, bb='hgnetv2_b0.ssld_stage2_ft_in1k'):
            super().__init__()
            self.mel = _MelExt(); self.bn0 = nn.BatchNorm2d(SED_N_MELS)
            self.backbone = timm.create_model(bb, pretrained=False, in_chans=1, num_classes=0, global_pool='')
            with torch.no_grad():
                f = self.backbone(torch.zeros(1, 1, SED_N_MELS, 100))
            self.head = _SEDHead(f.shape[1], N_CLS)
        def forward(self, x):
            m = self.mel(x); m = m.transpose(1, 2); m = self.bn0(m); m = m.transpose(1, 2)
            f = self.backbone(m); f = f.mean(dim=2) if f.dim() == 4 else f
            clip, _ = self.head(f); return clip

    st = torch.load(MW / "exp50_hgnet_sed.pt", map_location="cuda", weights_only=False)
    bb = st.get('backbone', 'hgnetv2_b0.ssld_stage2_ft_in1k')
    m = _SED(bb).to("cuda").eval()
    m.load_state_dict(st['state_dict'])

    SS_DIR = DATA / "train_soundscapes"
    files = sorted(sc_g.filename.unique())
    fname_idx = {f: [] for f in files}
    for i, r in sc_g.iterrows():
        fname_idx[r.filename].append((i, int(r.end_sec)))

    out = np.zeros((len(sc_g), N_CLS), dtype=np.float32)
    t0 = time.time()
    BATCH_F = 8
    with torch.inference_mode():
        for s in range(0, len(files), BATCH_F):
            batch = files[s:s+BATCH_F]
            chunks = []; meta = []
            for bi, fn in enumerate(batch):
                y, _ = sf.read(SS_DIR / fn, dtype="float32", always_2d=False)
                if y.ndim == 2: y = y.mean(axis=1)
                if len(y) < 60*SR: y = np.pad(y, (0, 60*SR - len(y)))
                y = y[:60*SR]
                for ci in range(3):
                    st_idx = ci * SED_CHUNK_SAMPLES
                    chunks.append(y[st_idx:st_idx + SED_CHUNK_SAMPLES])
                    meta.append((bi, ci))
            x = torch.from_numpy(np.stack(chunks)).to("cuda")
            clip = m(x)
            p = torch.sigmoid(clip).cpu().numpy()
            for k, (bi, ci) in enumerate(meta):
                fn = batch[bi]
                for row_idx, end_sec in fname_idx[fn]:
                    win_idx = (end_sec - 5) // 5
                    if win_idx // 4 == ci:
                        out[row_idx] = p[k]
            if (s // BATCH_F) % 10 == 0:
                print(f"    {s}/{len(files)} files, {time.time()-t0:.1f}s", flush=True)
    cache.parent.mkdir(exist_ok=True, parents=True)
    np.savez_compressed(cache, scores=out)
    print(f"  cached → {cache}", flush=True)
    return out


# ------------------------------------------------------------------ V9 taxon gate
def apply_v9_gate(probs, perch_emb, sp_taxon, offset=0.1):
    """Replicate exp45c V9 gate: probs *= clip(taxon_prob + offset, 0, 1)."""
    ck = torch.load(MW / "exp45a_taxon_head.pt", map_location="cuda", weights_only=False)
    class _Tx(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(1536, 256), nn.GELU(), nn.Dropout(0.2), nn.Linear(256, 5))
        def forward(self, x): return self.net(x)
    tx = _Tx().to("cuda").eval()
    tx.load_state_dict(ck["state_dict"])
    sp2t = np.asarray(ck["species_to_taxon"], dtype=np.int64)
    with torch.no_grad():
        E = torch.from_numpy(perch_emb.astype(np.float32)).to("cuda")
        tp = torch.sigmoid(tx(E)).cpu().numpy()
    gate = np.clip(tp[:, sp2t] + offset, 0, 1)
    return probs * gate


# ------------------------------------------------------------------ v33 baseline
def build_v33(sc_g, perch_prob, exp50_prob, perch_emb, sp_taxon, alpha=0.10):
    """v26 (0.7P + 0.3 exp50) + V9 gate + file-max coherence α=0.10."""
    base = 0.7 * perch_prob + 0.3 * exp50_prob
    v26_gated = apply_v9_gate(base, perch_emb, sp_taxon, offset=0.1)
    out = v26_gated.copy()
    for fname, idx in sc_g.groupby("filename").indices.items():
        sub = v26_gated[idx]
        fmax = sub.max(axis=0, keepdims=True)
        out[idx] = (1 - alpha) * sub + alpha * fmax
    return out.astype(np.float32)


# ------------------------------------------------------------------ Lever implementations
def L1_kalman_logit(probs, sc_g, q=0.1, r=0.5):
    """1-D Kalman smoother across 12 windows/file at LOGIT level, per class."""
    eps = 1e-6
    logit = np.log(np.clip(probs, eps, 1 - eps) / (1 - np.clip(probs, eps, 1 - eps)))
    out = logit.copy()
    for fname, idx in sc_g.groupby("filename").indices.items():
        sub = logit[idx]   # (T, C) where T <=12
        T = len(sub)
        # forward
        x_hat = sub.copy()
        P = np.full(sub.shape, r)
        for t in range(1, T):
            P_pred = P[t-1] + q
            K = P_pred / (P_pred + r)
            x_hat[t] = x_hat[t-1] + K * (sub[t] - x_hat[t-1])
            P[t] = (1 - K) * P_pred
        # backward (RTS)
        for t in range(T - 2, -1, -1):
            x_hat[t] = x_hat[t] + (q / (q + r)) * (x_hat[t+1] - x_hat[t])
        out[idx] = x_hat
    return (1 / (1 + np.exp(-out))).astype(np.float32)


def L2_hmm_present_absent(probs, sc_g, p_stay=0.7, p_switch=0.3):
    """Per-class 2-state HMM (absent=0, present=1) viterbi-style smoothing on logits."""
    eps = 1e-6
    out = probs.copy()
    log_trans = np.log(np.array([[p_stay, p_switch], [p_switch, p_stay]]) + eps)
    for fname, idx in sc_g.groupby("filename").indices.items():
        sub = probs[idx]  # (T, C)
        T = len(sub)
        for c in range(N_CLS):
            obs = sub[:, c]
            # log-emission: present = log(p), absent = log(1-p)
            log_em = np.stack([np.log(1 - obs + eps), np.log(obs + eps)], axis=1)
            # forward-backward (sum-product)
            fwd = np.zeros((T, 2))
            fwd[0] = log_em[0]
            for t in range(1, T):
                fwd[t] = log_em[t] + np.array([
                    np.logaddexp(fwd[t-1, 0] + log_trans[0, 0], fwd[t-1, 1] + log_trans[1, 0]),
                    np.logaddexp(fwd[t-1, 0] + log_trans[0, 1], fwd[t-1, 1] + log_trans[1, 1]),
                ])
            bwd = np.zeros((T, 2))
            for t in range(T - 2, -1, -1):
                bwd[t] = np.array([
                    np.logaddexp(log_trans[0, 0] + log_em[t+1, 0] + bwd[t+1, 0],
                                 log_trans[0, 1] + log_em[t+1, 1] + bwd[t+1, 1]),
                    np.logaddexp(log_trans[1, 0] + log_em[t+1, 0] + bwd[t+1, 0],
                                 log_trans[1, 1] + log_em[t+1, 1] + bwd[t+1, 1]),
                ])
            posterior = fwd + bwd
            posterior -= posterior.max(axis=1, keepdims=True)
            posterior = np.exp(posterior)
            posterior /= posterior.sum(axis=1, keepdims=True)
            out[idx, c] = posterior[:, 1]
    return out.astype(np.float32)


def L3_cross_model_confidence(probs_v33, perch_prob, exp50_prob, w_agree=0.05):
    """Boost where Perch & exp50 agree (high), suppress where disagree."""
    # agreement = 1 - |Perch - exp50| in prob space
    agree = 1.0 - np.abs(perch_prob - exp50_prob)
    # multiplicative correction: (1-w) + w * agree
    correction = (1 - w_agree) + w_agree * agree
    return (probs_v33 * correction).astype(np.float32)


def L4_file_feature_blend(probs, sc_g, alpha=0.05):
    """Add small contribution from file-level mean (smoother than file-max)."""
    out = probs.copy()
    for fname, idx in sc_g.groupby("filename").indices.items():
        sub = probs[idx]
        fmean = sub.mean(axis=0, keepdims=True)
        out[idx] = (1 - alpha) * sub + alpha * fmean
    return out.astype(np.float32)


def L5_per_class_platt(v33, Y, train_mask, eval_mask):
    """Fit per-class Platt scaling on train rows, apply to eval. Universal calibration."""
    from sklearn.linear_model import LogisticRegression
    out = v33.copy()
    for c in range(N_CLS):
        if Y[train_mask, c].sum() < 5: continue
        try:
            X_tr = v33[train_mask, c].reshape(-1, 1)
            y_tr = Y[train_mask, c]
            X_ev = v33[eval_mask, c].reshape(-1, 1)
            clf = LogisticRegression(max_iter=200, C=1.0, random_state=42).fit(X_tr, y_tr)
            out[eval_mask, c] = clf.predict_proba(X_ev)[:, 1]
        except Exception: pass
    return out.astype(np.float32)


# ------------------------------------------------------------------ Eval helper
def evaluate(P_full, ref_full, ev_mask, Y, sp_taxon, label):
    P_ev = P_full[ev_mask]
    ref_ev = ref_full[ev_mask]
    Y_ev = Y[ev_mask]
    macro, n = macro_auc(Y_ev, P_ev)
    macro_ref, _ = macro_auc(Y_ev, ref_ev)
    per_tax = per_taxon_macro(Y_ev, P_ev, sp_taxon)
    per_tax_ref = per_taxon_macro(Y_ev, ref_ev, sp_taxon)
    sp = per_row_spearman(ref_ev, P_ev)
    deltas = {t: (per_tax[t] - per_tax_ref[t]) if not (np.isnan(per_tax[t]) or np.isnan(per_tax_ref[t])) else np.nan
              for t in TAXA}
    # Predicted LB class
    aves_d = deltas.get("Aves", np.nan)
    macro_d = macro - macro_ref
    if not np.isfinite(sp):
        cls = "?"
    elif sp >= 0.99 and (np.isnan(aves_d) or aves_d >= 0):
        cls = "A (likely positive)"
    elif sp >= 0.99 and abs(macro_d) <= 0.005:
        cls = "B (likely neutral)"
    else:
        cls = "C (likely negative)"
    return {
        "label": label, "macro": macro, "macro_d": macro_d, "sp_row": sp,
        "Aves": deltas.get("Aves", np.nan), "Amphib": deltas.get("Amphibia", np.nan),
        "Insecta": deltas.get("Insecta", np.nan), "Mammal": deltas.get("Mammalia", np.nan),
        "Reptil": deltas.get("Reptilia", np.nan), "n_eval": n, "predicted": cls,
    }


def main():
    print("=== exp81: comprehensive local lever audit ===\n", flush=True)
    sc_g, Y, primary, _ = build_ss()
    sp_taxon = species_taxon_array()
    perch_prob = load_perch_scores_labeled()
    perch_emb = load_perch_emb_labeled()
    print(f"data: {len(sc_g)} rows, Perch prob {perch_prob.shape}, emb {perch_emb.shape}", flush=True)

    exp50_prob = get_exp50_scores(sc_g)
    print(f"exp50: {exp50_prob.shape}", flush=True)

    print("\nBuilding v33 baseline...", flush=True)
    v33 = build_v33(sc_g, perch_prob, exp50_prob, perch_emb, sp_taxon, alpha=0.10)
    print(f"v33: {v33.shape}, range [{v33.min():.3f}, {v33.max():.3f}]", flush=True)

    ev_mask = sc_g.split.values == "eval"
    tr_mask = sc_g.split.values == "train"

    # Reference
    rows = []
    rows.append(evaluate(v33, v33, ev_mask, Y, sp_taxon, "v33 (ref)"))

    # === Candidate sweep ===
    print("\n=== L1 Kalman logit-level (q×r grid) ===", flush=True)
    for q in [0.05, 0.1, 0.3]:
        for r in [0.5, 1.0]:
            P = L1_kalman_logit(v33, sc_g, q=q, r=r)
            rows.append(evaluate(P, v33, ev_mask, Y, sp_taxon, f"L1 Kalman q={q} r={r}"))

    print("=== L2 HMM 2-state (p_stay grid) ===", flush=True)
    for p_stay in [0.6, 0.7, 0.8]:
        P = L2_hmm_present_absent(v33, sc_g, p_stay=p_stay, p_switch=1 - p_stay)
        rows.append(evaluate(P, v33, ev_mask, Y, sp_taxon, f"L2 HMM p_stay={p_stay}"))

    print("=== L3 cross-model agreement (w grid) ===", flush=True)
    for w in [0.03, 0.05, 0.10, 0.15]:
        P = L3_cross_model_confidence(v33, perch_prob, exp50_prob, w_agree=w)
        rows.append(evaluate(P, v33, ev_mask, Y, sp_taxon, f"L3 agree w={w}"))

    print("=== L4 file-mean blend (alpha grid) ===", flush=True)
    for a in [0.03, 0.05, 0.10]:
        P = L4_file_feature_blend(v33, sc_g, alpha=a)
        rows.append(evaluate(P, v33, ev_mask, Y, sp_taxon, f"L4 fmean α={a}"))

    print("=== L5 per-class Platt (fit on train OOF) ===", flush=True)
    P = L5_per_class_platt(v33, Y, tr_mask, ev_mask)
    rows.append(evaluate(P, v33, ev_mask, Y, sp_taxon, "L5 Platt"))

    df = pd.DataFrame(rows)
    print("\n=== Results (sorted by macro_d) ===")
    cols = ["label", "macro_d", "sp_row", "Aves", "Insecta", "Mammal", "Amphib", "Reptil", "predicted"]
    print(df[cols].sort_values("macro_d", ascending=False).to_string(index=False))
    df.to_csv(EXP80 / "exp81_lever_audit.csv", index=False)
    print(f"\nSaved → {EXP80}/exp81_lever_audit.csv")

    # Top recommendation
    print("\n=== Top recommendation ===")
    safe = df[df.predicted.str.startswith("A")]
    if len(safe) > 0:
        best = safe.sort_values("macro_d", ascending=False).iloc[0]
        print(f"  {best.label}")
        print(f"    macro_d = {best.macro_d:+.4f}, sp_row = {best.sp_row:.4f}")
        print(f"    Aves = {best.Aves:+.4f}, Insecta = {best.Insecta:+.4f}")
    else:
        print("  No 'likely positive' candidate. Try class B candidates with caution.")


if __name__ == "__main__":
    main()
