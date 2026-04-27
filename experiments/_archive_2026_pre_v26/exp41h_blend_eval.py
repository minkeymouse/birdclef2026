#!/usr/bin/env python3
"""exp41e blend eval: student pseudo SED vs v12 baseline."""
import json
import numpy as np
import pandas as pd
import soundfile as sf
import timm
import torch
import torch.nn as nn
import torchaudio
from pathlib import Path
from sklearn.metrics import roc_auc_score
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import gaussian_filter1d

ROOT = Path("/data/birdclef2026")
DATA = ROOT / "data/birdclef-2026"
EXP21 = ROOT / "experiments/exp21_outputs/perch_cache"
EXP28 = ROOT / "experiments/exp28_outputs"
EXP29 = ROOT / "experiments/exp29_outputs"
EXP41 = ROOT / "experiments/exp41h_outputs"

SR = 32000
CLIP_SAMPLES = 20 * SR
WINDOW_SEC = 5
N_CLASSES = 234
N_FFT = 2048; HOP = 512; N_MELS = 128
FMIN = 50; FMAX = 14000
SEED = 42
EVAL_N_FILES = 11


def macro(Y, S):
    k = Y.sum(0) > 0
    return float(roc_auc_score(Y[:, k], S[:, k], average="macro"))


def zs(X): return (X - X.mean(0, keepdims=True)) / (X.std(0, keepdims=True) + 1e-8)


class Mel(nn.Module):
    def __init__(self):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS,
            f_min=FMIN, f_max=FMAX, power=2.0, center=True)
        self.adb = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)
    def forward(self, x): return self.adb(self.mel(x)).unsqueeze(1)

class Head(nn.Module):
    def __init__(self, d, nc):
        super().__init__()
        self.att = nn.Conv1d(d, nc, 1); self.cla = nn.Conv1d(d, nc, 1)
    def forward(self, x):
        a = self.att(x); c = self.cla(x)
        w = torch.softmax(a, dim=-1)
        return (w * c).sum(-1), c.max(-1).values

class SEDM(nn.Module):
    def __init__(self, bb="hgnetv2_b0.ssld_stage2_ft_in1k"):
        super().__init__()
        self.mel = Mel(); self.bn0 = nn.BatchNorm2d(N_MELS)
        self.backbone = timm.create_model(bb, pretrained=False, in_chans=1,
                                          num_classes=0, global_pool="")
        with torch.no_grad():
            f = self.backbone(torch.zeros(1, 1, N_MELS, 100))
        self.head = Head(f.shape[1], N_CLASSES)
    def forward(self, x):
        m = self.mel(x); m = m.transpose(1, 2); m = self.bn0(m); m = m.transpose(1, 2)
        feat = self.backbone(m)
        if feat.dim() == 4: feat = feat.mean(2)
        return self.head(feat)


class ValDS(Dataset):
    def __init__(self, meta):
        self.meta = meta.reset_index(drop=True)
    def __len__(self): return len(self.meta)
    def __getitem__(self, idx):
        row = self.meta.iloc[idx]
        end = int(row["row_id"].rsplit("_", 1)[1])
        start = end - WINDOW_SEC
        y, sr = sf.read(DATA / "train_soundscapes" / row["filename"], dtype="float32", always_2d=False)
        if y.ndim == 2: y = y.mean(1)
        c = ((start + end) // 2) * SR
        half = CLIP_SAMPLES // 2
        s = max(0, c - half); e = s + CLIP_SAMPLES
        if e > len(y): e = len(y); s = max(0, e - CLIP_SAMPLES)
        clip = y[s:e]
        if len(clip) < CLIP_SAMPLES: clip = np.pad(clip, (0, CLIP_SAMPLES - len(clip)))
        return torch.from_numpy(clip.astype(np.float32)), idx


def build_truth():
    sc_raw = pd.read_csv(DATA / "train_soundscapes_labels.csv").drop_duplicates().reset_index(drop=True)
    sub = pd.read_csv(DATA / "sample_submission.csv")
    primary = sub.columns[1:].tolist()
    l2i = {c: i for i, c in enumerate(primary)}
    def parse(x):
        if pd.isna(x): return []
        return [t.strip() for t in str(x).split(";") if t.strip()]
    sc = (sc_raw.groupby(["filename", "start", "end"])["primary_label"]
          .apply(lambda s: sorted({l for x in s for l in parse(x)})).reset_index(name="lbls"))
    sc["end_sec"] = pd.to_timedelta(sc["end"]).dt.total_seconds().astype(int)
    sc["row_id"] = sc["filename"].str.replace(".ogg", "", regex=False) + "_" + sc["end_sec"].astype(str)
    meta = pd.read_parquet(EXP21 / "full_perch_meta.parquet")
    Y_sc = np.zeros((len(sc), len(primary)), dtype=np.uint8)
    for i, labs in enumerate(sc["lbls"]):
        for l in labs:
            if l in l2i: Y_sc[i, l2i[l]] = 1
    idx = sc.set_index("row_id")
    Y = np.stack([Y_sc[idx.index.get_loc(rid)] for rid in meta["row_id"]])
    return meta, Y


def inference(model, meta):
    ds = ValDS(meta)
    dl = DataLoader(ds, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)
    preds = np.zeros((len(meta), N_CLASSES), dtype=np.float32)
    with torch.no_grad():
        for x, idxs in dl:
            x = x.to("cuda", non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                clip, _ = model(x)
            p = torch.sigmoid(clip).float().cpu().numpy()
            for i, j in zip(idxs.tolist(), range(len(p))):
                preds[i] = p[j]
    return preds


def main():
    meta, Y = build_truth()

    # Reproduce exp41e's file split (same SEED=42 as exp38)
    sc_raw = pd.read_csv(DATA / "train_soundscapes_labels.csv").drop_duplicates()
    rng = np.random.RandomState(SEED)
    files_all = sorted(sc_raw.filename.unique())
    rng.shuffle(files_all)
    eval_files = set(files_all[:EVAL_N_FILES])
    mask_eval = meta["filename"].isin(eval_files).values
    print(f"Eval-only rows: {mask_eval.sum()}  (train-leaked: {(~mask_eval).sum()})")

    # Load exp41e model
    m = SEDM().to("cuda").eval()
    ckpt = torch.load(EXP41 / "best_ckpt.pt", map_location="cuda", weights_only=False)
    m.load_state_dict(ckpt["state_dict"])
    print(f"exp41e ckpt epoch={ckpt.get('epoch')}  val_auc={ckpt.get('val_auc'):.4f}")

    # Inference on full 59-file Val-A
    sed41h = inference(m, meta)
    np.savez_compressed(EXP41 / "val_scores_full.npz", preds=sed41h)

    # Load references
    perch = np.load(EXP28 / "best_oof.npz")["val_a_smoothed"]
    sed29 = np.load(EXP29 / "val_scores.npz")["preds"]

    # --- Clean eval (11 held-out files) ---
    print("\n=== Clean eval (11 held-out files) ===")
    Ye = Y[mask_eval]
    Pe = perch[mask_eval]
    S29e = sed29[mask_eval]
    S41h = sed41h[mask_eval]
    print(f"  Perch: {macro(Ye, Pe):.4f}")
    print(f"  SED29: {macro(Ye, S29e):.4f}")
    print(f"  SED41h: {macro(Ye, S41h):.4f}")

    zP, z29, z41 = zs(Pe), zs(S29e), zs(S41h)
    files_e = meta.loc[mask_eval, "filename"].values

    def gauss_blend(S):
        out = np.zeros_like(S)
        for f in np.unique(files_e):
            m = files_e == f
            out[m] = gaussian_filter1d(S[m], sigma=0.5, axis=0)
        return out

    # v12 reference (Perch + SED29, α=0.80)
    v12 = gauss_blend(0.80 * zP + 0.20 * z29)
    print(f"  v12 ref (P·0.80 + SED29·0.20 + Gauss): {macro(Ye, v12):.4f}")

    # Perch + SED41h 2-way
    print("\n  Perch + SED41h 2-way:")
    best_2 = (-1, None)
    for a in np.arange(0.50, 0.96, 0.05):
        s = a * zP + (1 - a) * z41
        s_g = gauss_blend(s)
        auc = macro(Ye, s_g)
        if auc > best_2[0]: best_2 = (auc, a)
        print(f"    α={a:.2f}: {auc:.4f}")
    print(f"    best: α={best_2[1]:.2f} → {best_2[0]:.4f}")

    # 3-way Perch + SED29 + SED41h
    print("\n  3-way: wP + w29 + w41:")
    best_3 = (-1, None)
    for wP in np.arange(0.65, 0.91, 0.05):
        for w29 in np.arange(0.0, 1.01 - wP, 0.025):
            w41 = 1.0 - wP - w29
            if w41 < -1e-9: continue
            s = wP * zP + w29 * z29 + w41 * z41
            s_g = gauss_blend(s)
            auc = macro(Ye, s_g)
            if auc > best_3[0]: best_3 = (auc, (wP, w29, w41))
    print(f"    best: {best_3[0]:.4f}  (wP={best_3[1][0]:.2f}, w29={best_3[1][1]:.3f}, w41={best_3[1][2]:.3f})")

    # Pearson SED29 vs SED41h
    from scipy.stats import pearsonr
    keep = Ye.sum(0) > 0
    r, _ = pearsonr(S29e[:, keep].flatten(), S41h[:, keep].flatten())
    print(f"\n  SED29 vs SED41h Pearson: {r:.3f}")

    # --- Full 59-file (leaked) for reference ---
    print("\n=== Full 59-file (leaked, upper bound) ===")
    print(f"  SED41h alone: {macro(Y, sed41h):.4f}")

    out = {
        "clean_11files": {
            "perch": macro(Ye, Pe),
            "sed29": macro(Ye, S29e),
            "sed41h": macro(Ye, S41h),
            "v12_ref": float(macro(Ye, v12)),
            "best_P_SED41h": {"alpha": float(best_2[1]), "val_a": float(best_2[0])},
            "best_3way": {"wP": float(best_3[1][0]), "w29": float(best_3[1][1]),
                          "w41": float(best_3[1][2]), "val_a": float(best_3[0])},
            "pearson_29_41": float(r),
        },
        "leaked_full": float(macro(Y, sed41h)),
    }
    (EXP41 / "blend_results.json").write_text(json.dumps(out, indent=2))
    print(f"\nSaved: {EXP41}/blend_results.json")


if __name__ == "__main__":
    main()
