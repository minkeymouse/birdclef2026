#!/usr/bin/env python3
"""exp44c — Dedicated 27-species head on labeled SS ONLY.

Rationale (from exp43r audit):
  27 species (25 Insecta sonXX + 2 Amphibia) have NO train_audio clips AND
  are NOT in Perch's 14,795-class training.  They are invisible to the
  current exp41f pipeline — teacher scores are structurally 0 for them,
  so pseudo generation fails.  Their only supervision is labeled SS.

Setup:
  - Training: 55 labeled SS files (exp38 split, seed 42)
  - Restrict label set to 27 double-blind species
  - Windows with ≥1 of 27 species labels are "positive"; others are negatives
  - HGNet B0, 20-sec clip → 12 × 5-sec windows, mel spec
  - BCE(clipwise) + BCE(framewise_max), raw-waveform mixup
  - WeightedSampler: oversample positive windows 4-8x
  - 25 epochs max, early stop patience 5

Eval:
  - 11 held-out labeled SS files (same split as exp38)
  - 27-class per-class AUC (on classes with eval positives)

Output:
  experiments/exp44c_outputs/
    best_ckpt.pt
    results.json (history + final Val-A for 27 species)
    val_scores.npz (132 eval rows × 27 species probs)
"""
from __future__ import annotations
import json, random, re, time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from sklearn.metrics import roc_auc_score
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "birdclef-2026"
OUT = ROOT / "experiments" / "exp44c_outputs"
OUT.mkdir(parents=True, exist_ok=True)

SR = 32000
CLIP_SEC = 20
CLIP_SAMPLES = SR * CLIP_SEC
WINDOW_SEC = 5
WINDOW_SAMPLES = SR * WINDOW_SEC
FILE_SAMPLES = SR * 60
N_WINDOWS_FILE = 12

N_FFT = 2048; HOP = 512; N_MELS = 128
FMIN = 50; FMAX = 14000

BATCH_SIZE = 32; EPOCHS = 40
LR = 1e-4; WD = 1e-2                 # lowered from 5e-4 to prevent NaN
NUM_WORKERS = 4
MIXUP_ALPHA = 0.4; MIXUP_P = 0.3      # lower mixup prob to avoid extreme combos
POS_OVERSAMPLE = 6.0
EARLY_STOP_PATIENCE = 8
EVAL_N_FILES = 11
SEED = 42

DEVICE = "cuda"
BACKBONE = "hgnetv2_b0.ssld_stage2_ft_in1k"


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def get_27_species():
    """Return (27_species_primary_labels, label_to_local_idx) for double-blind species."""
    tax = pd.read_csv(DATA / "taxonomy.csv")
    perch_sci = set(open(ROOT / "perch_v2/assets/labels.csv").read().strip().split("\n"))
    tax["in_perch"] = tax["scientific_name"].isin(perch_sci)
    ta = pd.read_csv(DATA / "train.csv")
    ta_counts = ta.groupby("primary_label").size()
    tax["n_train_audio"] = tax["primary_label"].astype(str).map(ta_counts).fillna(0).astype(int)
    # Double-blind: NOT in Perch AND 0 train_audio clips
    double_blind = tax[(~tax.in_perch) & (tax.n_train_audio == 0)]
    labels = double_blind["primary_label"].astype(str).tolist()
    l2i = {l: i for i, l in enumerate(labels)}
    print(f"Double-blind species: {len(labels)}")
    print(f"  {double_blind.groupby('class_name').size().to_dict()}")
    return labels, l2i, double_blind


def build_segments(labels_27, l2i_27):
    sc = pd.read_csv(DATA / "train_soundscapes_labels.csv").drop_duplicates().reset_index(drop=True)
    def parse(x):
        if pd.isna(x): return []
        return [t.strip() for t in str(x).split(";") if t.strip()]
    sc_g = (sc.groupby(["filename","start","end"])["primary_label"]
            .apply(lambda s: sorted({l for x in s for l in parse(x)})).reset_index(name="lbls"))
    sc_g["end_sec"] = pd.to_timedelta(sc_g["end"]).dt.total_seconds().astype(int)

    # Split files
    rng = np.random.RandomState(SEED)
    files = sorted(sc_g["filename"].unique())
    rng.shuffle(files)
    eval_files = set(files[:EVAL_N_FILES])
    train_files = set(files[EVAL_N_FILES:])

    sc_train = sc_g[sc_g.filename.isin(train_files)].reset_index(drop=True)
    sc_eval  = sc_g[sc_g.filename.isin(eval_files)].reset_index(drop=True)

    # Per-window 27-class targets
    def make_y(df):
        Y = np.zeros((len(df), len(labels_27)), dtype=np.float32)
        for i, lbls in enumerate(df["lbls"]):
            for l in lbls:
                if l in l2i_27:
                    Y[i, l2i_27[l]] = 1.0
        return Y
    Y_train = make_y(sc_train)
    Y_eval = make_y(sc_eval)
    print(f"Train windows {len(sc_train)}  positives with any 27sp: {(Y_train.sum(1)>0).sum()}")
    print(f"Eval windows  {len(sc_eval)}  positives with any 27sp: {(Y_eval.sum(1)>0).sum()}")
    print(f"Per-27species train positive counts: min={Y_train.sum(0).min():.0f}  max={Y_train.sum(0).max():.0f}  mean={Y_train.sum(0).mean():.1f}")
    print(f"Per-27species eval  positive counts: min={Y_eval.sum(0).min():.0f}  max={Y_eval.sum(0).max():.0f}  mean={Y_eval.sum(0).mean():.1f}")
    return sc_train, sc_eval, Y_train, Y_eval


class SSClipDataset(Dataset):
    """Returns 20-sec clip randomly centered on a target window + 12-label vector."""
    def __init__(self, sc_df, Y, train=True):
        self.sc = sc_df.reset_index(drop=True)
        self.Y = Y
        self.train = train
        # group by filename for quick lookup
        self.by_file = {f: self.sc.index[self.sc.filename == f].tolist() for f in self.sc.filename.unique()}

    def __len__(self): return len(self.sc)

    def __getitem__(self, idx):
        row = self.sc.iloc[idx]
        p = DATA / "train_soundscapes" / row.filename
        try:
            wav, _ = sf.read(str(p), dtype="float32")
            if wav.ndim > 1: wav = wav.mean(1)
            if len(wav) < FILE_SAMPLES:
                wav = np.pad(wav, (0, FILE_SAMPLES - len(wav)))
            wav = wav[:FILE_SAMPLES]
        except Exception:
            wav = np.zeros(FILE_SAMPLES, dtype=np.float32)
        end_sec = int(row.end_sec)
        # 20s clip centered around the target window
        target_center = (end_sec - WINDOW_SEC / 2) * SR
        clip_start = int(max(0, target_center - CLIP_SAMPLES / 2))
        clip_start = min(clip_start, FILE_SAMPLES - CLIP_SAMPLES)
        clip = wav[clip_start:clip_start + CLIP_SAMPLES]
        if len(clip) < CLIP_SAMPLES:
            clip = np.pad(clip, (0, CLIP_SAMPLES - len(clip)))
        # Build per-window labels for 12 windows within the clip
        # For each of 12 windows in the clip, use the multi-label set that aligns to that window's time
        win_y = np.zeros((N_WINDOWS_FILE, self.Y.shape[1]), dtype=np.float32)
        # Map each clip window to its corresponding sc row (same filename, matching end_sec)
        for w in range(N_WINDOWS_FILE):
            w_end_sec = (clip_start // SR) + (w + 1) * WINDOW_SEC
            rows = self.by_file.get(row.filename, [])
            for r_idx in rows:
                if int(self.sc.at[r_idx, "end_sec"]) == w_end_sec:
                    win_y[w] = self.Y[r_idx]; break
        return torch.from_numpy(clip), torch.from_numpy(win_y)


class Mel(nn.Module):
    def __init__(self):
        super().__init__()
        self.m = torchaudio.transforms.MelSpectrogram(
            sample_rate=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS,
            f_min=FMIN, f_max=FMAX, power=2.0, center=True)
        self.adb = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)
    def forward(self, x): return self.adb(self.m(x)).unsqueeze(1)


class SED27(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.mel = Mel()
        self.bn0 = nn.BatchNorm2d(N_MELS)                               # exp29-style input BN
        self.backbone = timm.create_model(BACKBONE, pretrained=True, in_chans=1,
                                           drop_rate=0.1, drop_path_rate=0.1,
                                           num_classes=0, global_pool="")
        with torch.no_grad():
            feat = self.backbone(torch.zeros(1, 1, N_MELS, 200))
        C = feat.shape[1]
        self.att = nn.Conv1d(C, n_classes, 1)
        self.cls = nn.Conv1d(C, n_classes, 1)

    def forward(self, x):
        m = self.mel(x)                                                # (B, 1, n_mels, T)
        m = m.transpose(1, 2)                                          # (B, n_mels, 1, T)
        m = self.bn0(m)
        m = m.transpose(1, 2)                                          # back (B, 1, n_mels, T)
        feat = self.backbone(m)                                         # (B, C, H', W')
        f = feat.mean(dim=2) if feat.dim() == 4 else feat              # (B, C, T)
        att = torch.softmax(self.att(f), dim=-1)                       # (B, C', T)
        cls = self.cls(f)                                              # (B, C', T)
        clip = (att * cls).sum(-1)                                      # (B, C')
        frame_max = cls.max(-1).values                                  # (B, C')
        return clip, frame_max, cls


def mixup_data(x, y, alpha=MIXUP_ALPHA):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], torch.maximum(lam * y, (1 - lam) * y[idx])


def train_one(model, loader, opt, dev):
    """Pure fp32 — AMP off. 4M params, small dataset."""
    model.train()
    tot, n, nan_skipped = 0.0, 0, 0
    for x, y in loader:
        x = x.to(dev, non_blocking=True); y = y.to(dev, non_blocking=True)
        if not torch.isfinite(x).all():
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        y_clip = y.amax(dim=1)
        if random.random() < MIXUP_P:
            x, y_clip = mixup_data(x, y_clip)
        clip, fmax, _ = model(x)
        if not (torch.isfinite(clip).all() and torch.isfinite(fmax).all()):
            nan_skipped += 1; opt.zero_grad(set_to_none=True); continue
        loss = F.binary_cross_entropy_with_logits(clip, y_clip) + \
               F.binary_cross_entropy_with_logits(fmax, y_clip)
        if not torch.isfinite(loss):
            nan_skipped += 1; opt.zero_grad(set_to_none=True); continue
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(gnorm):
            nan_skipped += 1; continue
        opt.step()
        tot += loss.item() * x.size(0); n += x.size(0)
    return (tot / max(n, 1)), nan_skipped


@torch.no_grad()
def evaluate(model, sc_eval, Y_eval, dev):
    model.eval()
    preds = np.zeros(Y_eval.shape, dtype=np.float32)   # FIX: float32, was uint8 truncating sigmoid
    for i in range(len(sc_eval)):
        row = sc_eval.iloc[i]
        p = DATA / "train_soundscapes" / row.filename
        try:
            wav, _ = sf.read(str(p), dtype="float32")
            if wav.ndim > 1: wav = wav.mean(1)
            if len(wav) < FILE_SAMPLES:
                wav = np.pad(wav, (0, FILE_SAMPLES - len(wav)))
            wav = wav[:FILE_SAMPLES]
        except Exception:
            wav = np.zeros(FILE_SAMPLES, dtype=np.float32)
        end_sec = int(row.end_sec)
        target_center = (end_sec - WINDOW_SEC/2) * SR
        start = int(max(0, target_center - CLIP_SAMPLES/2))
        start = min(start, FILE_SAMPLES - CLIP_SAMPLES)
        clip = wav[start:start + CLIP_SAMPLES]
        x = torch.from_numpy(clip).unsqueeze(0).to(dev)
        clip_logits, fmax_logits, _ = model(x)        # fp32
        probs = torch.sigmoid(clip_logits)[0].cpu().numpy()
        if not np.isfinite(probs).all():
            probs = np.nan_to_num(probs, nan=0.5)
        preds[i] = probs
    # macro AUC over classes with ≥1 positive AND ≥1 negative
    aucs = []
    n_skipped_allpos = 0; n_skipped_allneg = 0; n_skipped_err = 0
    preds_nan = int(np.isnan(preds).any())
    for c in range(Y_eval.shape[1]):
        y_c = Y_eval[:, c].astype(int)
        if y_c.sum() == 0: n_skipped_allneg += 1; continue
        if y_c.sum() == len(y_c): n_skipped_allpos += 1; continue
        try:
            aucs.append(roc_auc_score(y_c, preds[:, c]))
        except Exception as e:
            n_skipped_err += 1
            if n_skipped_err <= 2:
                print(f"    [eval debug] class {c} roc_auc error: {e}  "
                      f"y_sum={y_c.sum()}  pred_min={preds[:,c].min():.3f}  pred_max={preds[:,c].max():.3f}")
    if aucs and n_skipped_err == 0 and n_skipped_allpos == 0:
        pass  # silent normal case
    else:
        print(f"    [eval] n_aucs={len(aucs)}  allneg={n_skipped_allneg}  allpos={n_skipped_allpos}  err={n_skipped_err}  preds_has_nan={preds_nan}  preds_range=[{preds.min():.3f},{preds.max():.3f}]")
    return (np.mean(aucs) if aucs else 0.0), aucs, preds


def main():
    set_seed(SEED)
    labels_27, l2i_27, taxinfo = get_27_species()
    sc_train, sc_eval, Y_train, Y_eval = build_segments(labels_27, l2i_27)

    ds_train = SSClipDataset(sc_train, Y_train, train=True)
    # Oversample positive windows
    has_pos = (Y_train.sum(1) > 0).astype(np.float32)
    weights = 1.0 + has_pos * (POS_OVERSAMPLE - 1.0)
    sampler = WeightedRandomSampler(weights.tolist(), num_samples=len(ds_train), replacement=True)
    loader = DataLoader(ds_train, batch_size=BATCH_SIZE, sampler=sampler,
                         num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)

    model = SED27(n_classes=len(labels_27)).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"SED27 params: {n_params/1e6:.2f} M  (output classes: {len(labels_27)})")

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    best_auc = -1.0; best_epoch = 0; patience_ctr = 0
    history = []
    for ep in range(1, EPOCHS + 1):
        t0 = time.time()
        tr_loss, nan_skipped = train_one(model, loader, opt, DEVICE)
        sched.step()
        auc, per_class_aucs, _ = evaluate(model, sc_eval, Y_eval, DEVICE)
        dt = time.time() - t0
        history.append({"epoch": ep, "train_loss": tr_loss, "val_auc": auc,
                         "time_s": dt, "n_classes_eval": len(per_class_aucs),
                         "nan_skipped": nan_skipped})
        print(f"  ep {ep:02d}  loss {tr_loss:.4f}  val_auc {auc:.4f}  n_cls={len(per_class_aucs)}  nan_skip={nan_skipped}  ({dt:.0f}s)")
        if auc > best_auc:
            best_auc = auc; best_epoch = ep; patience_ctr = 0
            torch.save({"state_dict": model.state_dict(), "epoch": ep, "val_auc": auc,
                        "labels_27": labels_27, "l2i_27": l2i_27},
                       OUT / "best_ckpt.pt")
        else:
            patience_ctr += 1
            if patience_ctr >= EARLY_STOP_PATIENCE:
                print(f"Early stop at ep {ep}")
                break

    # Final: reload best, extract predictions
    ckpt = torch.load(OUT / "best_ckpt.pt", map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    final_auc, final_per_class, final_preds = evaluate(model, sc_eval, Y_eval, DEVICE)
    print(f"\nFinal best_auc={best_auc:.4f} at epoch {best_epoch}")
    print(f"Per-class AUC (top-10): {sorted(final_per_class, reverse=True)[:10]}")
    print(f"Per-class AUC (bot-10): {sorted(final_per_class)[:10]}")
    np.savez_compressed(OUT / "val_scores.npz", preds=final_preds, Y_eval=Y_eval)

    with open(OUT / "results.json", "w") as fp:
        json.dump({"history": history, "best_auc": best_auc, "best_epoch": best_epoch,
                   "final_per_class_auc": final_per_class,
                   "labels_27": labels_27}, fp, indent=2, default=float)
    print(f"Saved → {OUT}/")


if __name__ == "__main__":
    main()
