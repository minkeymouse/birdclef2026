#!/usr/bin/env python3
"""exp121b — Score exp121 ckpt on labeled SS rows + v33 blend audit.

Run after exp121 training completes. Generates exp121_scores_labeled.npz
(739, 234) for blend testing.
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
import torchaudio
import timm
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))
from _lib.data import build_ss, species_taxon_array, EXP80, ROOT, N_CLS, DATA, SR
from exp82_q1_4way_on_v33 import apply_v9_gate, file_max_blend, evaluate
from _lib.eval_metrics import macro_auc, per_taxon_macro

DEVICE = "cuda"
EXP121_CKPT = ROOT / "experiments/_data_pipelines/exp121_outputs/best_ckpt.pt"
SED_N_FFT = 2048; SED_HOP = 512; SED_N_MELS = 128; SED_FMIN = 50; SED_FMAX = 14000
SED_CHUNK_SAMPLES = SR * 20
N_WINDOWS = 12


class _MelExt(nn.Module):
    def __init__(self):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=SR, n_fft=SED_N_FFT, hop_length=SED_HOP, n_mels=SED_N_MELS,
            f_min=SED_FMIN, f_max=SED_FMAX, power=2.0, center=True)
        self.adb = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)
    def forward(self, x): return self.adb(self.mel(x)).unsqueeze(1)


class _SEDHead(nn.Module):
    def __init__(self, fd, nc):
        super().__init__()
        self.att = nn.Conv1d(fd, nc, 1); self.cla = nn.Conv1d(fd, nc, 1)
    def forward(self, x):
        a = self.att(x); c = self.cla(x)
        return (torch.softmax(a, dim=-1) * c).sum(-1), c.max(-1).values


class _SED(nn.Module):
    def __init__(self, bb="hgnetv2_b0.ssld_stage2_ft_in1k"):
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


def get_exp121_scores(sc_g, ckpt_path):
    """Run exp121 SED on labeled SS, return (n_rows, 234) cache."""
    cache = EXP80 / "exp121_scores_labeled.npz"
    if cache.exists():
        print(f"  Loading cached exp121 scores...")
        return np.load(cache)["scores"]

    print("Loading exp121 ckpt + running on labeled SS...")
    st = torch.load(str(ckpt_path), map_location=DEVICE, weights_only=False)
    print(f"  ckpt val_SS: {st.get('val_SS', '?')}")
    bb = st.get("config", {}).get("backbone", "hgnetv2_b0.ssld_stage2_ft_in1k")
    m = _SED(bb).to(DEVICE).eval()
    m.load_state_dict(st["state_dict"])

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
                # 3 chunks of 20s = full 60s
                for ci in range(3):
                    st_idx = ci * SED_CHUNK_SAMPLES
                    chunks.append(y[st_idx:st_idx + SED_CHUNK_SAMPLES])
            x = torch.from_numpy(np.stack(chunks).astype(np.float32)).to(DEVICE)
            logits = m(x)
            probs = torch.sigmoid(logits).cpu().numpy()  # (3*n_files, 234)
            # Each chunk has 4 windows (5 sec each in 20 sec chunk)
            for bi, fn in enumerate(batch):
                for ci in range(3):
                    chunk_prob = probs[bi*3 + ci]  # one prob vector per 20s chunk
                    # Assign to 4 windows in this chunk
                    for ri, (row_idx, end_sec) in enumerate(fname_idx[fn]):
                        # Map row's end_sec (5s window) to 20s chunk index
                        chunk_for_row = (end_sec - 1) // 20
                        if chunk_for_row == ci:
                            out[row_idx] = chunk_prob

    np.savez_compressed(cache, scores=out)
    print(f"  Saved {cache} in {(time.time()-t0)/60:.1f} min")
    return out


def main():
    print("=== exp121b: Score exp121 + v33 blend audit ===\n", flush=True)
    sc_g, Y, primary, _ = build_ss()
    sp_taxon = species_taxon_array()

    from _lib.data import load_perch_scores_labeled, load_perch_emb_labeled
    perch_prob = load_perch_scores_labeled()
    perch_emb = load_perch_emb_labeled()

    exp50 = np.load(EXP80 / "exp50_scores_labeled.npz")["scores"]

    # Build v33
    base = 0.7 * perch_prob + 0.3 * exp50
    gated = apply_v9_gate(base, perch_emb, sp_taxon, offset=0.1)
    v33 = file_max_blend(gated, sc_g, alpha=0.10)

    # Get exp121 scores
    exp121 = get_exp121_scores(sc_g, EXP121_CKPT)
    print(f"  exp121 scores: {exp121.shape}, range [{exp121.min():.4f}, {exp121.max():.4f}]")

    # Pearson correlations
    from scipy.stats import pearsonr
    print(f"\n  Pearson correlations:")
    print(f"    Perch ↔ exp50:   {pearsonr(perch_prob.flatten(), exp50.flatten())[0]:.3f}")
    print(f"    Perch ↔ exp121:  {pearsonr(perch_prob.flatten(), exp121.flatten())[0]:.3f}")
    print(f"    exp50 ↔ exp121:  {pearsonr(exp50.flatten(), exp121.flatten())[0]:.3f}")

    # Blend audits
    ev_mask = sc_g.split.values == "eval"
    rows = [evaluate(v33, v33, ev_mask, Y, sp_taxon, "v33 ref")]

    # Replace exp50 with exp121 entirely
    base_new = 0.7 * perch_prob + 0.3 * exp121
    gated_new = apply_v9_gate(base_new, perch_emb, sp_taxon, offset=0.1)
    v33_new = file_max_blend(gated_new, sc_g, alpha=0.10)
    rows.append(evaluate(v33_new, v33, ev_mask, Y, sp_taxon, "v33-style: 0.7P + 0.3*exp121 (full swap)"))

    # Mix exp50 + exp121
    for w_121 in [0.05, 0.10, 0.15, 0.20]:
        base_mix = 0.7 * perch_prob + (0.3 - w_121) * exp50 + w_121 * exp121
        gated_mix = apply_v9_gate(base_mix, perch_emb, sp_taxon, offset=0.1)
        v33_mix = file_max_blend(gated_mix, sc_g, alpha=0.10)
        rows.append(evaluate(v33_mix, v33, ev_mask, Y, sp_taxon, f"v33-style 0.7P+{0.3-w_121:.2f}exp50+{w_121}exp121"))

    # Also test additive on top of v33
    for w_121 in [0.05, 0.10, 0.15]:
        P = (1 - w_121) * v33 + w_121 * exp121
        rows.append(evaluate(P, v33, ev_mask, Y, sp_taxon, f"v33 + {w_121}*exp121 additive"))

    res = pd.DataFrame(rows)
    cols = ["label", "macro_d", "sp_row", "Aves", "Insecta", "Mammal", "Amphib", "Reptil", "predicted"]
    print("\n=== Blend audit (sorted by macro_d desc) ===")
    print(res.sort_values("macro_d", ascending=False)[cols].to_string(index=False))


if __name__ == "__main__":
    main()
