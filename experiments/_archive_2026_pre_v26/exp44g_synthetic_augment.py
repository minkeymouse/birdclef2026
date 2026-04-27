#!/usr/bin/env python3
"""exp44g — Synthetic augmentation for 27 double-blind species to break site shortcut.

Problem (exp43r + exp44c + site audit):
  16/27 species appear in a SINGLE site only. Training on these labeled SS
  risks learning "site signature = sonotype" rather than true acoustic content.
  exp44c hits Val-A_v2 0.848 but held-out eval is SAME-site distribution
  as training → overestimates test-LB generalization.

Synthetic recipe:
  For each 27-species labeled window (source call):
    1. Extract audio (5s or 20s around the labeled window)
    2. Pick K different-site background clips from unlabeled SS (abundant:
       10,592 files × 12 windows, almost all different-site)
    3. Generate K synthetic clips: α * source + (1-α) * bg, α ~ U(0.4, 0.8)
    4. Preserve species labels from source window only (bg assumed species-free)

  Training data:
    real_positives + synthetic_from_real (3x) + real_negatives
    Site-diverse exposure forces model to learn content, not site.

Architecture: same SED27 as exp44c (HGNet B0 + BN2d + attention SED head,
fp32, LR 1e-4). Only the dataloader differs.

Eval: unchanged 11 held-out files → SAME metric as exp44c.  Gate: Val-A_v2
≥ 0.80 (close to exp44c 0.848). If synth hurts real-domain eval much, the
recipe is wrong.  If synth ≈ real performance AND site-invariant metric
improves (cross-site per-species AUC), synth wins even with equal scalar AUC.

Output: experiments/exp44g_outputs/best_ckpt.pt + results.json.
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
OUT = ROOT / "experiments" / "exp44g_outputs"
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
LR = 1e-4; WD = 1e-2
NUM_WORKERS = 4
MIXUP_ALPHA = 0.4; MIXUP_P = 0.3
POS_OVERSAMPLE = 4.0
SYNTH_MULTIPLIER = 3        # 3x synthetic per real positive
SYNTH_ALPHA_MIN = 0.4       # source call gain mix range
SYNTH_ALPHA_MAX = 0.8
EARLY_STOP_PATIENCE = 8
EVAL_N_FILES = 11
SEED = 42
BACKBONE = "hgnetv2_b0.ssld_stage2_ft_in1k"
DEVICE = "cuda"

FNAME_RE = re.compile(r"BC2026_(?:Train|Test)_(\d+)_(S\d+)_(\d{8})_(\d{6})\.ogg")


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def get_27_species():
    tax = pd.read_csv(DATA / "taxonomy.csv")
    perch_sci = set(open(ROOT / "perch_v2/assets/labels.csv").read().strip().split("\n"))
    tax["in_perch"] = tax["scientific_name"].isin(perch_sci)
    ta = pd.read_csv(DATA / "train.csv")
    tax["n_ta"] = tax["primary_label"].astype(str).map(ta.groupby("primary_label").size()).fillna(0).astype(int)
    db = tax[(~tax.in_perch) & (tax.n_ta == 0)]
    labels = db["primary_label"].astype(str).tolist()
    l2i = {l: i for i, l in enumerate(labels)}
    return labels, l2i, db


def parse_site(fname):
    m = FNAME_RE.match(fname)
    return m.group(2) if m else None


def read_60s(path):
    wav, _ = sf.read(str(path), dtype="float32")
    if wav.ndim > 1: wav = wav.mean(1)
    if len(wav) < FILE_SAMPLES:
        wav = np.pad(wav, (0, FILE_SAMPLES - len(wav)))
    return wav[:FILE_SAMPLES]


def build_segments(labels_27, l2i_27):
    sc = pd.read_csv(DATA / "train_soundscapes_labels.csv").drop_duplicates().reset_index(drop=True)
    def parse(x):
        if pd.isna(x): return []
        return [t.strip() for t in str(x).split(";") if t.strip()]
    sc_g = (sc.groupby(["filename","start","end"])["primary_label"]
            .apply(lambda s: sorted({l for x in s for l in parse(x)})).reset_index(name="lbls"))
    sc_g["end_sec"] = pd.to_timedelta(sc_g["end"]).dt.total_seconds().astype(int)
    sc_g["site"] = sc_g["filename"].str.extract(r"_(S\d+)_")

    rng = np.random.RandomState(SEED)
    files = sorted(sc_g["filename"].unique())
    rng.shuffle(files)
    eval_files = set(files[:EVAL_N_FILES])
    train_files = set(files[EVAL_N_FILES:])

    sc_train = sc_g[sc_g.filename.isin(train_files)].reset_index(drop=True)
    sc_eval = sc_g[sc_g.filename.isin(eval_files)].reset_index(drop=True)

    def make_y(df):
        Y = np.zeros((len(df), len(labels_27)), dtype=np.float32)
        for i, lbls in enumerate(df["lbls"]):
            for l in lbls:
                if l in l2i_27: Y[i, l2i_27[l]] = 1.0
        return Y
    return sc_train, sc_eval, make_y(sc_train), make_y(sc_eval)


def build_background_pool(exclude_sites_per_source, min_pool=200):
    """For each site, build a pool of (file, win_idx) from unlabeled SS of OTHER sites.
       Background = unlabeled SS windows (we assume these contain minimal 27-species content)."""
    ss_dir = DATA / "train_soundscapes"
    labeled_files = set(
        pd.read_csv(DATA / "train_soundscapes_labels.csv").drop_duplicates()["filename"]
    )
    all_files = sorted(p.name for p in ss_dir.glob("*.ogg"))
    unlabeled = [f for f in all_files if f not in labeled_files]
    # group by site
    by_site = {}
    for f in unlabeled:
        s = parse_site(f)
        by_site.setdefault(s, []).append(f)
    # drop underrepresented sites
    for s in list(by_site):
        if len(by_site[s]) < 5:
            del by_site[s]
    print(f"Background pool: {sum(len(v) for v in by_site.values())} files across {len(by_site)} sites")
    for s in sorted(by_site): print(f"  {s}: {len(by_site[s])} files")
    return by_site


class SSSynthClipDataset(Dataset):
    """Each item: 20s clip containing a source window.
       With prob synth_p, mix with different-site background.
       Y is 12×27 window-level target computed from source segment labels."""
    def __init__(self, sc_df, Y, bg_pool_by_site, labels_27, synth_p=0.75, train=True):
        self.sc = sc_df.reset_index(drop=True)
        self.Y = Y
        self.bg_pool = bg_pool_by_site          # site -> list of filenames (unlabeled)
        self.labels_27 = labels_27
        self.synth_p = synth_p if train else 0.0
        self.train = train
        self.by_file = {f: self.sc.index[self.sc.filename == f].tolist() for f in self.sc.filename.unique()}

    def __len__(self): return len(self.sc)

    def _load_clip(self, filename, end_sec):
        p = DATA / "train_soundscapes" / filename
        try:
            wav = read_60s(p)
        except Exception:
            wav = np.zeros(FILE_SAMPLES, dtype=np.float32)
        center = (end_sec - WINDOW_SEC / 2) * SR
        start = int(max(0, center - CLIP_SAMPLES / 2))
        start = min(start, FILE_SAMPLES - CLIP_SAMPLES)
        clip = wav[start:start + CLIP_SAMPLES]
        if len(clip) < CLIP_SAMPLES: clip = np.pad(clip, (0, CLIP_SAMPLES - len(clip)))
        return clip, start

    def _sample_bg(self, exclude_site):
        other_sites = [s for s in self.bg_pool if s != exclude_site]
        if not other_sites: return np.zeros(CLIP_SAMPLES, dtype=np.float32)
        s = random.choice(other_sites)
        f = random.choice(self.bg_pool[s])
        p = DATA / "train_soundscapes" / f
        try:
            wav = read_60s(p)
        except Exception:
            return np.zeros(CLIP_SAMPLES, dtype=np.float32)
        bg_start = random.randint(0, FILE_SAMPLES - CLIP_SAMPLES)
        return wav[bg_start:bg_start + CLIP_SAMPLES]

    def __getitem__(self, idx):
        row = self.sc.iloc[idx]
        source, clip_start = self._load_clip(row.filename, int(row.end_sec))
        source_site = row.site

        # Decide synth
        has_pos = self.Y[idx].sum() > 0
        do_synth = self.train and has_pos and (random.random() < self.synth_p)
        if do_synth:
            bg = self._sample_bg(source_site)
            alpha = random.uniform(SYNTH_ALPHA_MIN, SYNTH_ALPHA_MAX)
            clip = alpha * source + (1 - alpha) * bg
        else:
            clip = source

        # Build window labels (12 × 27) based on sc_train rows matching this filename
        win_y = np.zeros((N_WINDOWS_FILE, self.Y.shape[1]), dtype=np.float32)
        rows = self.by_file.get(row.filename, [])
        for w in range(N_WINDOWS_FILE):
            w_end_sec = (clip_start // SR) + (w + 1) * WINDOW_SEC
            for r_idx in rows:
                if int(self.sc.at[r_idx, "end_sec"]) == w_end_sec:
                    win_y[w] = self.Y[r_idx]; break
        return torch.from_numpy(clip.astype(np.float32)), torch.from_numpy(win_y)


class Mel(nn.Module):
    def __init__(self):
        super().__init__()
        self.m = torchaudio.transforms.MelSpectrogram(
            sample_rate=SR, n_fft=N_FFT, hop_length=HOP, win_length=N_FFT,
            n_mels=N_MELS, f_min=FMIN, f_max=FMAX, power=2.0, center=True)
        self.adb = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)
    def forward(self, x): return self.adb(self.m(x)).unsqueeze(1)


class SED27(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.mel = Mel()
        self.bn0 = nn.BatchNorm2d(N_MELS)
        self.backbone = timm.create_model(BACKBONE, pretrained=True, in_chans=1,
                                           drop_rate=0.1, drop_path_rate=0.1,
                                           num_classes=0, global_pool="")
        with torch.no_grad():
            feat = self.backbone(torch.zeros(1, 1, N_MELS, 200))
        C = feat.shape[1]
        self.att = nn.Conv1d(C, n_classes, 1)
        self.cls = nn.Conv1d(C, n_classes, 1)

    def forward(self, x):
        m = self.mel(x)
        m = m.transpose(1, 2); m = self.bn0(m); m = m.transpose(1, 2)
        feat = self.backbone(m)
        f = feat.mean(dim=2) if feat.dim() == 4 else feat
        att = torch.softmax(self.att(f), dim=-1); cls = self.cls(f)
        return (att * cls).sum(-1), cls.max(-1).values, cls


def mixup_data(x, y, alpha=MIXUP_ALPHA):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], torch.maximum(lam * y, (1 - lam) * y[idx])


def train_one(model, loader, opt, dev):
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
    preds = np.zeros(Y_eval.shape, dtype=np.float32)
    for i in range(len(sc_eval)):
        row = sc_eval.iloc[i]
        p = DATA / "train_soundscapes" / row.filename
        try:
            wav = read_60s(p)
        except Exception:
            wav = np.zeros(FILE_SAMPLES, dtype=np.float32)
        end_sec = int(row.end_sec)
        target_center = (end_sec - WINDOW_SEC/2) * SR
        start = int(max(0, target_center - CLIP_SAMPLES/2))
        start = min(start, FILE_SAMPLES - CLIP_SAMPLES)
        clip = wav[start:start + CLIP_SAMPLES]
        if len(clip) < CLIP_SAMPLES: clip = np.pad(clip, (0, CLIP_SAMPLES - len(clip)))
        x = torch.from_numpy(clip).unsqueeze(0).to(dev)
        clip_logits, _, _ = model(x)
        probs = torch.sigmoid(clip_logits)[0].cpu().numpy()
        if not np.isfinite(probs).all():
            probs = np.nan_to_num(probs, nan=0.5)
        preds[i] = probs
    aucs = []
    for c in range(Y_eval.shape[1]):
        y = Y_eval[:, c].astype(int)
        if y.sum() == 0 or y.sum() == len(y): continue
        try: aucs.append(roc_auc_score(y, preds[:, c]))
        except Exception: pass
    return (np.mean(aucs) if aucs else 0.0), aucs, preds


def main():
    set_seed(SEED)
    labels_27, l2i_27, taxinfo = get_27_species()
    print(f"27 double-blind species: {labels_27[:5]}...")

    sc_train, sc_eval, Y_train, Y_eval = build_segments(labels_27, l2i_27)
    print(f"Train windows {len(sc_train)}  positives {int((Y_train.sum(1)>0).sum())}")
    print(f"Eval  windows {len(sc_eval)}  positives {int((Y_eval.sum(1)>0).sum())}")

    bg_pool = build_background_pool(None)

    ds_train = SSSynthClipDataset(sc_train, Y_train, bg_pool, labels_27, synth_p=0.75, train=True)
    # Oversample positive windows; since synth multiplier is implicit via synth_p, not needed extra
    has_pos = (Y_train.sum(1) > 0).astype(np.float32)
    weights = 1.0 + has_pos * (POS_OVERSAMPLE - 1.0)
    sampler = WeightedRandomSampler(weights.tolist(), num_samples=len(ds_train)*2, replacement=True)
    loader = DataLoader(ds_train, batch_size=BATCH_SIZE, sampler=sampler,
                         num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)

    model = SED27(n_classes=len(labels_27)).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"SED27 params: {n_params/1e6:.2f} M")

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
                print(f"Early stop at ep {ep}"); break

    ckpt = torch.load(OUT / "best_ckpt.pt", map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    final_auc, final_per_class, final_preds = evaluate(model, sc_eval, Y_eval, DEVICE)
    print(f"\nFinal best_auc={best_auc:.4f} at epoch {best_epoch}")
    print(f"Top-10 per-class AUC: {sorted(final_per_class, reverse=True)[:10]}")
    print(f"Bot-10 per-class AUC: {sorted(final_per_class)[:10]}")
    np.savez_compressed(OUT / "val_scores.npz", preds=final_preds, Y_eval=Y_eval)
    with open(OUT / "results.json", "w") as fp:
        json.dump({"history": history, "best_auc": best_auc, "best_epoch": best_epoch,
                   "final_per_class_auc": final_per_class,
                   "labels_27": labels_27, "config": {
                       "synth_p": 0.75, "synth_alpha": [SYNTH_ALPHA_MIN, SYNTH_ALPHA_MAX],
                       "synth_multiplier_via_sampler": 2, "pos_oversample": POS_OVERSAMPLE,
                       "epochs": EPOCHS, "lr": LR,
                   }}, fp, indent=2, default=float)
    print(f"Saved → {OUT}/")


if __name__ == "__main__":
    main()
