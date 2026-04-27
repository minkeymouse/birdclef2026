#!/usr/bin/env python3
"""exp82 Q1 — Re-run exp63 v30e (4-way blend with ConvNeXt-tiny exp59)
on top of the v33 base (V9 gate + file-max coherence) instead of v26.

Specific question: does adding exp59 to the v33 pipeline preserve the
v30e local benefit (+0.019 macro, +0.009 Aves) AND keep sp_row safe enough
for LB transfer? v33 = LB 0.932 reference. We need predicted-class A
(Aves Δ ≥ 0, sp_row ≥ 0.99) to recommend LB submission.

Tests several blend configurations on top of the v33 base recipe so we
get a full sweep, not a single point.
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
                        load_perch_scores_labeled, DATA, MW, EXP80, ROOT, N_CLS, SR, TAXA)
from _lib.eval_metrics import macro_auc, per_taxon_macro, per_row_spearman

EXP59_CKPT = ROOT / "experiments/_archive_2026_audits/outputs/exp59_outputs/best_ckpt.pt"


def get_exp50_scores(sc_g):
    cache = EXP80 / "exp50_scores_labeled.npz"
    if cache.exists():
        return np.load(cache)["scores"]
    raise FileNotFoundError("expected exp50 cache from exp81. run exp81 first.")


def get_sed_scores(sc_g, ckpt_path: Path, backbone: str, cache_name: str):
    cache = EXP80 / cache_name
    if cache.exists():
        return np.load(cache)["scores"]
    print(f"  building {cache_name} (~3 min)...", flush=True)
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

    class _Head(nn.Module):
        def __init__(self, fd, nc):
            super().__init__()
            self.att = nn.Conv1d(fd, nc, 1); self.cla = nn.Conv1d(fd, nc, 1)
        def forward(self, x):
            a, c = self.att(x), self.cla(x)
            return (torch.softmax(a, dim=-1) * c).sum(-1), c.max(-1).values

    class _SED(nn.Module):
        def __init__(self, bb):
            super().__init__()
            self.mel = _MelExt(); self.bn0 = nn.BatchNorm2d(SED_N_MELS)
            self.backbone = timm.create_model(bb, pretrained=False, in_chans=1, num_classes=0, global_pool='')
            with torch.no_grad():
                f = self.backbone(torch.zeros(1, 1, SED_N_MELS, 100))
            self.head = _Head(f.shape[1], N_CLS)
        def forward(self, x):
            m = self.mel(x); m = m.transpose(1, 2); m = self.bn0(m); m = m.transpose(1, 2)
            f = self.backbone(m); f = f.mean(dim=2) if f.dim() == 4 else f
            clip, _ = self.head(f); return clip

    st = torch.load(ckpt_path, map_location="cuda", weights_only=False)
    m = _SED(backbone).to("cuda").eval()
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


def apply_v9_gate(probs, perch_emb, species_to_taxon_idx, offset=0.1):
    ck = torch.load(MW / "exp45a_taxon_head.pt", map_location="cuda", weights_only=False)
    class _Tx(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(1536, 256), nn.GELU(), nn.Dropout(0.2), nn.Linear(256, 5))
        def forward(self, x): return self.net(x)
    tx = _Tx().to("cuda").eval()
    tx.load_state_dict(ck["state_dict"])
    sp2t = np.asarray(ck["species_to_taxon"], dtype=np.int64)
    with torch.no_grad():
        E = torch.from_numpy(perch_emb.astype(np.float32)).to("cuda")
        tp = torch.sigmoid(tx(E)).cpu().numpy()
    gate = np.clip(tp[:, sp2t] + offset, 0, 1)
    return probs * gate


def file_max_blend(probs, sc_g, alpha=0.10):
    out = probs.copy()
    for fname, idx in sc_g.groupby("filename").indices.items():
        sub = probs[idx]
        fmax = sub.max(axis=0, keepdims=True)
        out[idx] = (1 - alpha) * sub + alpha * fmax
    return out.astype(np.float32)


def build_full_pipeline(perch_prob, exp50_prob, perch_emb, sc_g, sp_taxon,
                         exp59_prob=None, wP=0.7, w50=0.3, w59=0.0):
    """Mimics v33 production pipeline: blend → V9 gate → file-max coherence."""
    if exp59_prob is None or w59 == 0:
        base = wP * perch_prob + w50 * exp50_prob
    else:
        base = wP * perch_prob + w50 * exp50_prob + w59 * exp59_prob
    gated = apply_v9_gate(base, perch_emb, sp_taxon, offset=0.1)
    return file_max_blend(gated, sc_g, alpha=0.10)


def evaluate(P_full, ref_full, ev_mask, Y, sp_taxon, label, sp_floor=0.99):
    P_ev = P_full[ev_mask]; ref_ev = ref_full[ev_mask]; Y_ev = Y[ev_mask]
    macro, n = macro_auc(Y_ev, P_ev)
    macro_ref, _ = macro_auc(Y_ev, ref_ev)
    pt = per_taxon_macro(Y_ev, P_ev, sp_taxon)
    pt_ref = per_taxon_macro(Y_ev, ref_ev, sp_taxon)
    sp = per_row_spearman(ref_ev, P_ev)
    deltas = {t: (pt[t] - pt_ref[t]) if not (np.isnan(pt[t]) or np.isnan(pt_ref[t])) else np.nan for t in TAXA}
    macro_d = macro - macro_ref
    aves_d = deltas.get("Aves", np.nan)
    cls = "?"
    if np.isfinite(sp):
        if sp >= sp_floor and (np.isnan(aves_d) or aves_d >= 0):
            cls = "A (likely positive)"
        elif sp >= sp_floor and abs(macro_d) <= 0.005:
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
    print("=== exp82 Q1: 4-way blend (P + exp50 + exp59) on v33 base ===\n", flush=True)
    sc_g, Y, primary, _ = build_ss()
    sp_taxon = species_taxon_array()

    perch_prob = load_perch_scores_labeled()
    perch_emb = load_perch_emb_labeled()
    exp50 = get_exp50_scores(sc_g)
    print(f"loaded: perch_prob {perch_prob.shape}, exp50 {exp50.shape}", flush=True)

    print("Loading exp59 ConvNeXt-tiny scores (build cache if missing)...", flush=True)
    exp59 = get_sed_scores(sc_g, EXP59_CKPT, "convnext_tiny.fb_in22k_ft_in1k", "exp59_scores_labeled.npz")
    print(f"exp59: {exp59.shape}, range [{exp59.min():.3f}, {exp59.max():.3f}]", flush=True)

    # Independence diagnostic
    from scipy.stats import pearsonr
    print(f"\nPearson correlation across 739 × 234 cells:")
    print(f"  Perch ↔ exp50:  {pearsonr(perch_prob.flatten(), exp50.flatten())[0]:.3f}")
    print(f"  Perch ↔ exp59:  {pearsonr(perch_prob.flatten(), exp59.flatten())[0]:.3f}")
    print(f"  exp50 ↔ exp59:  {pearsonr(exp50.flatten(), exp59.flatten())[0]:.3f}")

    # Build v33 reference + 4-way variants
    ev_mask = sc_g.split.values == "eval"

    print("\n=== Building v33 baseline (Perch + exp50 + V9 gate + file-max) ===", flush=True)
    v33_ref = build_full_pipeline(perch_prob, exp50, perch_emb, sc_g, sp_taxon,
                                    exp59_prob=None, wP=0.7, w50=0.3, w59=0.0)
    print(f"v33 ref: range [{v33_ref.min():.3f}, {v33_ref.max():.3f}]", flush=True)

    rows = [evaluate(v33_ref, v33_ref, ev_mask, Y, sp_taxon, "v33 ref (0.7P + 0.3 exp50)")]

    # Sweep
    configs = [
        # exp59 added at small weight, exp50 reduced
        ("v33+e59 0.7P + 0.25 + 0.05",   0.70, 0.25, 0.05),
        ("v33+e59 0.7P + 0.20 + 0.10",   0.70, 0.20, 0.10),
        ("v33+e59 0.7P + 0.15 + 0.15",   0.70, 0.15, 0.15),
        # exp30e equivalent
        ("v33+e59 0.6P + 0.20 + 0.20 (=v30e)", 0.60, 0.20, 0.20),
        ("v33+e59 0.6P + 0.15 + 0.25",   0.60, 0.15, 0.25),
        ("v33+e59 0.6P + 0.30 + 0.10",   0.60, 0.30, 0.10),
        # exp59 swap
        ("v33-e50 0.7P + 0.0 + 0.30 (replace)", 0.70, 0.00, 0.30),
        ("v33-e50 0.6P + 0.0 + 0.40 (replace)", 0.60, 0.00, 0.40),
        # exp59 light addition keeping exp50 strong
        ("v33+e59 0.65P + 0.30 + 0.05",  0.65, 0.30, 0.05),
        ("v33+e59 0.65P + 0.25 + 0.10",  0.65, 0.25, 0.10),
    ]
    print("\n=== Variant sweep (all run through V9 gate + file-max α=0.10) ===", flush=True)
    for label, wP, w50, w59 in configs:
        P = build_full_pipeline(perch_prob, exp50, perch_emb, sc_g, sp_taxon,
                                 exp59_prob=exp59, wP=wP, w50=w50, w59=w59)
        rows.append(evaluate(P, v33_ref, ev_mask, Y, sp_taxon, label))

    df = pd.DataFrame(rows)
    cols = ["label", "macro_d", "sp_row", "Aves", "Insecta", "Mammal", "Amphib", "Reptil", "predicted"]
    print("\n=== Results (sorted by macro_d desc) ===")
    print(df[cols].sort_values("macro_d", ascending=False).to_string(index=False))
    df.to_csv(EXP80 / "exp82_q1_results.csv", index=False)
    print(f"\nSaved → {EXP80}/exp82_q1_results.csv")

    # Top recommendation
    safe = df[df.predicted.str.startswith("A") & (df.label != "v33 ref (0.7P + 0.3 exp50)")]
    print("\n=== Top class-A configurations ===")
    if len(safe) > 0:
        top = safe.sort_values("macro_d", ascending=False).head(3)
        print(top[cols].to_string(index=False))
    else:
        print("  (none — class-A criterion is sp_row ≥ 0.99 AND Aves Δ ≥ 0)")

    # Best overall (relaxed)
    print("\n=== Best by Aves Δ (LB-transfer most predictive variable) ===")
    nonref = df[df.label != "v33 ref (0.7P + 0.3 exp50)"]
    print(nonref.sort_values("Aves", ascending=False).head(5)[cols].to_string(index=False))


if __name__ == "__main__":
    main()
