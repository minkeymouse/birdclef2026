#!/usr/bin/env python3
"""exp73 — Fine-tune exp50 with quality-filtered iNat external clips.

Pre-filter: only clips where exp50 self-prediction > 0.3 (excludes background-noise
and mislabeled clips). Use as additional positive samples for the 5 target species.

Warm-start from exp50 best ckpt, train 8 epochs on combined data.
Compare: held-out 11-file val_SS (rare-heavy) before/after.
"""
from __future__ import annotations
import json, random, time
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
import timm
import torch, torch.nn as nn, torch.nn.functional as F
import torchaudio
from sklearn.metrics import roc_auc_score
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "birdclef-2026"
EXT = ROOT / "data" / "external"
BG_PATH = ROOT / "experiments" / "_data_pipelines" / "exp49a" if False else None
BG_NPZ = ROOT / "experiments" / "exp49_outputs" / "bg_quiet_2025.npz"  # archived path
EXP50_CKPT = ROOT / "experiments" / "_data_pipelines" / "exp50_outputs" / "best_ckpt.pt"
OUT = ROOT / "experiments" / "_data_pipelines" / "exp73_outputs"
OUT.mkdir(parents=True, exist_ok=True)

SR = 32000; CLIP_SEC = 20; CLIP_SAMPLES = SR * CLIP_SEC
WINDOW_SEC = 5; FILE_SAMPLES = SR * 60
N_FFT = 2048; HOP = 512; N_MELS = 128; FMIN = 50; FMAX = 14000
BATCH_SIZE = 32; EPOCHS = 8; LR = 5e-5  # lower LR for fine-tune
WD = 1e-2
MIXUP_ALPHA = 0.5; MIXUP_P = 0.5  # less mixup for fine-tune
BG_MIX_P = 0.3
BG_ALPHA_LO, BG_ALPHA_HI = 0.4, 0.7
SECONDARY_WEIGHT = 0.3
WARMUP_EPOCHS = 1
SPEC_FREQ_MASK = 16; SPEC_TIME_MASK = 40
EVAL_SS_N_FILES = 11; SEED = 42
EXT_FILTER_THRESH = 0.3  # only use clips where exp50 self > this
DEVICE = "cuda"


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def build_primaries():
    sub = pd.read_csv(DATA / "sample_submission.csv")
    primary = sub.columns[1:].tolist()
    return primary, {c: i for i, c in enumerate(primary)}


def build_ta_splits(l2i, val_frac=0.20, seed=SEED):
    df = pd.read_csv(DATA / "train.csv")
    df = df[df["primary_label"].astype(str).isin(l2i)].reset_index(drop=True)
    df["primary_idx"] = df["primary_label"].astype(str).map(l2i)
    rng = np.random.RandomState(seed); val_idx = []; train_idx = []
    for lbl, g in df.groupby("primary_label"):
        g_idx = g.index.tolist(); rng.shuffle(g_idx)
        n_val = max(1, int(len(g_idx) * val_frac)) if len(g_idx) >= 5 else 0
        val_idx.extend(g_idx[:n_val]); train_idx.extend(g_idx[n_val:])
    train_df = df.loc[train_idx].reset_index(drop=True)
    val_df = df.loc[val_idx].reset_index(drop=True)
    def parse_sec(x):
        if pd.isna(x) or x in ("[]", ""): return []
        try: return [s.strip("'\" ") for s in x.strip("[]").split(",") if s.strip("'\" ")]
        except Exception: return []
    train_df["secondary_list"] = train_df["secondary_labels"].apply(parse_sec)
    val_df["secondary_list"] = val_df["secondary_labels"].apply(parse_sec)
    return train_df, val_df


def build_ss_splits(l2i):
    sc_raw = pd.read_csv(DATA / "train_soundscapes_labels.csv").drop_duplicates().reset_index(drop=True)
    def parse(x):
        if pd.isna(x): return []
        return [t.strip() for t in str(x).split(";") if t.strip()]
    sc_g = (sc_raw.groupby(["filename","start","end"])["primary_label"]
            .apply(lambda s: sorted({l for x in s for l in parse(x)})).reset_index(name="lbls"))
    sc_g["end_sec"] = pd.to_timedelta(sc_g["end"]).dt.total_seconds().astype(int)
    sc_g["row_id"] = sc_g["filename"].str.replace(".ogg","",regex=False) + "_" + sc_g["end_sec"].astype(str)
    rng = np.random.RandomState(SEED)
    files = sorted(sc_g["filename"].unique()); rng.shuffle(files)
    eval_files = set(files[:EVAL_SS_N_FILES])
    train_files = set(files[EVAL_SS_N_FILES:])
    ss_train = sc_g[sc_g.filename.isin(train_files)].reset_index(drop=True)
    ss_eval = sc_g[sc_g.filename.isin(eval_files)].reset_index(drop=True)
    return ss_train, ss_eval


def collect_ext_clips(l2i):
    """Collect all external clips. Returns list of (species_idx, path)."""
    clips = []
    for sp_dir in EXT.iterdir():
        if not sp_dir.is_dir() or sp_dir.name.startswith("_"): continue
        sp = sp_dir.name
        if sp not in l2i: continue
        for f in sp_dir.iterdir():
            if f.suffix.lower() in {".ogg", ".mp3", ".wav"} and f.is_file():
                clips.append((l2i[sp], f))
    return clips


def load_audio(path, target_samples):
    try:
        wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
        if wav.ndim > 1: wav = wav.mean(1)
        if sr != SR:
            import torchaudio.functional as TF
            wav = TF.resample(torch.from_numpy(wav), sr, SR).numpy()
        if len(wav) == 0: return np.zeros(target_samples, dtype=np.float32)
        if len(wav) < target_samples:
            reps = target_samples // len(wav) + 1
            wav = np.tile(wav, reps)[:target_samples]
        return wav.astype(np.float32)
    except Exception:
        return np.zeros(target_samples, dtype=np.float32)


def random_crop(wav, target):
    if len(wav) <= target:
        if len(wav) < target: wav = np.pad(wav, (0, target - len(wav)))
        return wav[:target]
    s = random.randint(0, len(wav) - target); return wav[s:s + target]


def center_crop(wav, target):
    if len(wav) <= target:
        if len(wav) < target: wav = np.pad(wav, (0, target - len(wav)))
        return wav[:target]
    s = (len(wav) - target) // 2; return wav[s:s + target]


class TADataset(Dataset):
    def __init__(self, df, l2i, train=True):
        self.df = df.reset_index(drop=True); self.l2i = l2i; self.train = train
        self.n_cls = len(l2i)
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = DATA / "train_audio" / row.filename
        wav = load_audio(path, CLIP_SAMPLES * 2)
        wav = random_crop(wav, CLIP_SAMPLES) if self.train else center_crop(wav, CLIP_SAMPLES)
        y = np.zeros(self.n_cls, dtype=np.float32)
        y[row.primary_idx] = 1.0
        for sl in row.secondary_list:
            if sl in self.l2i: y[self.l2i[sl]] = SECONDARY_WEIGHT
        return torch.from_numpy(wav), torch.from_numpy(y), int(row.primary_idx), 1


class SSDataset(Dataset):
    def __init__(self, ss_df, l2i, train=True):
        self.ss = ss_df.reset_index(drop=True); self.l2i = l2i; self.train = train
        self.n_cls = len(l2i)
    def __len__(self): return len(self.ss)
    def __getitem__(self, idx):
        row = self.ss.iloc[idx]
        p = DATA / "train_soundscapes" / row.filename
        wav = load_audio(p, FILE_SAMPLES)
        end_sec = int(row.end_sec)
        target_c = (end_sec - WINDOW_SEC/2) * SR
        cs = int(max(0, target_c - CLIP_SAMPLES/2))
        cs = min(cs, FILE_SAMPLES - CLIP_SAMPLES)
        if self.train:
            cs = int(cs + random.randint(-SR, SR)); cs = max(0, min(cs, FILE_SAMPLES - CLIP_SAMPLES))
        clip = wav[cs:cs + CLIP_SAMPLES]
        if len(clip) < CLIP_SAMPLES: clip = np.pad(clip, (0, CLIP_SAMPLES - len(clip)))
        y = np.zeros(self.n_cls, dtype=np.float32)
        for l in row.lbls:
            if l in self.l2i: y[self.l2i[l]] = 1.0
        return torch.from_numpy(clip.astype(np.float32)), torch.from_numpy(y), -1, 0


class ExtDataset(Dataset):
    """Filtered external clips as positive samples."""
    def __init__(self, clip_list, l2i, train=True):
        self.clips = clip_list  # list of (sp_idx, path)
        self.l2i = l2i; self.train = train
        self.n_cls = len(l2i)
    def __len__(self): return len(self.clips)
    def __getitem__(self, idx):
        sp_idx, path = self.clips[idx]
        wav = load_audio(path, CLIP_SAMPLES * 2)
        wav = random_crop(wav, CLIP_SAMPLES) if self.train else center_crop(wav, CLIP_SAMPLES)
        y = np.zeros(self.n_cls, dtype=np.float32)
        y[sp_idx] = 1.0
        return torch.from_numpy(wav), torch.from_numpy(y), int(sp_idx), 1


class MelExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS,
            f_min=FMIN, f_max=FMAX, power=2.0, center=True)
        self.adb = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)
    def forward(self, x): return self.adb(self.mel(x)).unsqueeze(1)


class SpecAug(nn.Module):
    def __init__(self):
        super().__init__()
        self.fm = torchaudio.transforms.FrequencyMasking(freq_mask_param=SPEC_FREQ_MASK)
        self.tm = torchaudio.transforms.TimeMasking(time_mask_param=SPEC_TIME_MASK)
    def forward(self, x): return self.tm(self.fm(x))


class SEDHead(nn.Module):
    def __init__(self, feat_dim, n_cls):
        super().__init__()
        self.att = nn.Conv1d(feat_dim, n_cls, 1); self.cla = nn.Conv1d(feat_dim, n_cls, 1)
    def forward(self, x):
        a = self.att(x); c = self.cla(x)
        w = torch.softmax(a, dim=-1)
        return (w * c).sum(-1), c.max(-1).values


class SEDModel(nn.Module):
    def __init__(self, n_cls=234):
        super().__init__()
        self.mel = MelExtractor()
        self.bn0 = nn.BatchNorm2d(N_MELS)
        self.spec_aug = SpecAug()
        self.backbone = timm.create_model("hgnetv2_b0.ssld_stage2_ft_in1k",
                                          pretrained=True, in_chans=1,
                                          drop_rate=0.1, drop_path_rate=0.1,
                                          num_classes=0, global_pool="")
        with torch.no_grad():
            feat = self.backbone(torch.zeros(1, 1, N_MELS, 100))
        self.head = SEDHead(feat.shape[1], n_cls)
    def forward(self, x, training_aug=False):
        m = self.mel(x); m = m.transpose(1, 2); m = self.bn0(m); m = m.transpose(1, 2)
        if training_aug and self.training: m = self.spec_aug(m)
        feat = self.backbone(m)
        f = feat.mean(dim=2) if feat.dim() == 4 else feat
        clip, fmax = self.head(f)
        return clip, fmax


def bg_to_20s(bg5): return np.tile(bg5, 4).astype(np.float32)[:CLIP_SAMPLES]


def mixup_with_bg(x, y, primary_idx, is_ta, bg_pool):
    B = x.size(0)
    if bg_pool is not None and random.random() < BG_MIX_P:
        idx = np.random.randint(0, bg_pool.shape[0], size=B)
        bg_wavs = np.stack([bg_to_20s(bg_pool[i]) for i in idx])
        bg_t = torch.from_numpy(bg_wavs).to(x.device)
        lam = np.random.uniform(BG_ALPHA_LO, BG_ALPHA_HI)
        return lam * x + (1 - lam) * bg_t, y, primary_idx, is_ta, lam
    lam = np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA)
    idx = torch.randperm(B, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[idx]
    mixed_y = torch.maximum(lam * y, (1 - lam) * y[idx])
    return mixed_x, mixed_y, primary_idx.clone(), torch.minimum(is_ta, is_ta[idx]), lam


def hybrid_loss(clip, fmax, y, primary_idx, is_ta):
    ta_mask = is_ta == 1; ml_mask = ~ta_mask
    losses = []
    if ta_mask.any():
        losses.append(F.binary_cross_entropy_with_logits(clip[ta_mask], y[ta_mask]))
        losses.append(F.binary_cross_entropy_with_logits(fmax[ta_mask], y[ta_mask]))
    if ml_mask.any():
        losses.append(F.binary_cross_entropy_with_logits(clip[ml_mask], y[ml_mask]))
        losses.append(F.binary_cross_entropy_with_logits(fmax[ml_mask], y[ml_mask]))
    return sum(losses) / max(1, len(losses))


def train_epoch(model, loader, opt, dev, bg_pool):
    model.train(); tot, n, nan_s = 0.0, 0, 0
    for x, y, pr, is_ta in loader:
        x = x.to(dev, non_blocking=True); y = y.to(dev, non_blocking=True)
        pr = pr.to(dev, non_blocking=True); is_ta = is_ta.to(dev, non_blocking=True)
        if not torch.isfinite(x).all(): x = torch.nan_to_num(x, 0, 1, -1)
        if random.random() < MIXUP_P:
            x, y, pr, is_ta, _ = mixup_with_bg(x, y, pr, is_ta, bg_pool)
        clip, fmax = model(x, training_aug=True)
        if not (torch.isfinite(clip).all() and torch.isfinite(fmax).all()):
            nan_s += 1; opt.zero_grad(set_to_none=True); continue
        loss = hybrid_loss(clip, fmax, y, pr, is_ta)
        if not torch.isfinite(loss):
            nan_s += 1; opt.zero_grad(set_to_none=True); continue
        opt.zero_grad(set_to_none=True); loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(gnorm): nan_s += 1; continue
        opt.step()
        tot += loss.item() * x.size(0); n += x.size(0)
    return tot / max(n, 1), nan_s


@torch.no_grad()
def evaluate_ss(model, ss_eval, l2i, dev, batch=4):
    model.eval(); n_cls = len(l2i); n = len(ss_eval)
    preds = np.zeros((n, n_cls), dtype=np.float32); Y = np.zeros((n, n_cls), dtype=np.uint8)
    for i in range(0, n, batch):
        j = min(n, i + batch); wavs = []
        for k in range(i, j):
            row = ss_eval.iloc[k]
            wav = load_audio(DATA / "train_soundscapes" / row.filename, FILE_SAMPLES)
            target_c = (int(row.end_sec) - WINDOW_SEC/2) * SR
            cs = int(max(0, target_c - CLIP_SAMPLES/2)); cs = min(cs, FILE_SAMPLES - CLIP_SAMPLES)
            clip = wav[cs:cs + CLIP_SAMPLES]
            if len(clip) < CLIP_SAMPLES: clip = np.pad(clip, (0, CLIP_SAMPLES - len(clip)))
            wavs.append(clip.astype(np.float32))
            for l in row.lbls:
                if l in l2i: Y[k, l2i[l]] = 1
        x = torch.from_numpy(np.stack(wavs)).to(dev)
        clip_logits, _ = model(x)
        p = torch.sigmoid(clip_logits).cpu().numpy().astype(np.float32)
        if not np.isfinite(p).all(): p = np.nan_to_num(p, 0.5)
        preds[i:j] = p
    aucs = []
    for c in range(n_cls):
        if Y[:, c].sum() == 0 or Y[:, c].sum() == n: continue
        try: aucs.append(roc_auc_score(Y[:, c], preds[:, c]))
        except: pass
    return float(np.mean(aucs)) if aucs else 0.0, len(aucs), preds, Y


def main():
    set_seed(SEED)
    primary, l2i = build_primaries()
    ta_train, ta_val = build_ta_splits(l2i)
    ss_train, ss_eval = build_ss_splits(l2i)

    # Collect external clips
    ext_clips = collect_ext_clips(l2i)
    print(f"External clips total: {len(ext_clips)}")

    # Load exp50 ckpt for warm start
    print("Loading exp50 base ckpt...")
    ck = torch.load(EXP50_CKPT, map_location=DEVICE, weights_only=False)
    model = SEDModel(n_cls=len(l2i)).to(DEVICE)
    model.load_state_dict(ck["state_dict"])
    model.eval()

    # Filter external clips by exp50 self-prediction > threshold
    print(f"Filtering ext clips by exp50 self > {EXT_FILTER_THRESH}...")
    keep = []
    for sp_idx, path in ext_clips:
        wav = load_audio(path, CLIP_SAMPLES)
        with torch.no_grad():
            x = torch.from_numpy(wav[None]).to(DEVICE)
            p = torch.sigmoid(model(x)[0]).cpu().numpy()[0]
        if p[sp_idx] > EXT_FILTER_THRESH:
            keep.append((sp_idx, path))
    print(f"Kept {len(keep)} / {len(ext_clips)} clips after filter")

    # Build training data: 2026 TA + 2026 SS train + filtered ext
    ta_ds = TADataset(ta_train, l2i)
    ss_ds = SSDataset(ss_train, l2i)
    ext_ds = ExtDataset(keep, l2i)
    from torch.utils.data import ConcatDataset
    combined = ConcatDataset([ta_ds, ss_ds, ext_ds])
    # weights: TA 80%, SS 15%, ext oversample 5x to make it count
    w_ta = [1.0] * len(ta_ds)
    w_ss = [(len(ta_ds) / max(1, len(ss_ds))) * (0.15 / 0.80)] * len(ss_ds)
    w_ext = [(len(ta_ds) / max(1, len(ext_ds))) * (0.05 / 0.80) * 5] * len(ext_ds)  # 5x boost
    weights = w_ta + w_ss + w_ext
    sampler = WeightedRandomSampler(weights, num_samples=len(ta_ds) + len(ext_ds) * 5, replacement=True)
    loader = DataLoader(combined, batch_size=BATCH_SIZE, sampler=sampler,
                         num_workers=4, pin_memory=True, drop_last=True,
                         persistent_workers=True)

    # BG pool
    bg_pool = np.load(BG_NPZ)["windows"] if BG_NPZ.exists() else None
    print(f"BG: {bg_pool.shape if bg_pool is not None else 'none'}")

    # Optimizer
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS - WARMUP_EPOCHS)

    # Eval baseline (exp50 alone)
    print("\n=== Baseline exp50 val_SS (held-out 11) ===")
    ss_auc_base, n_ss, preds_base, Y_ev = evaluate_ss(model, ss_eval, l2i, DEVICE)
    print(f"  baseline: {ss_auc_base:.4f} ({n_ss} cls)")
    # Per-target species
    target_sps = ['326272', '67107', '74113', 'bafcur1', 'litnig1']
    for sp in target_sps:
        if sp in l2i:
            ci = l2i[sp]
            yc = Y_ev[:, ci]
            if yc.sum() > 0 and yc.sum() < len(yc):
                try:
                    auc = roc_auc_score(yc, preds_base[:, ci])
                    print(f"    {sp:<14}  baseline AUC = {auc:.3f}  (n_pos {yc.sum()})")
                except Exception: pass

    # Fine-tune
    print(f"\n=== Fine-tune {EPOCHS} ep (LR {LR}) ===")
    best = {"val_SS": ss_auc_base, "epoch": 0}
    for ep in range(1, EPOCHS + 1):
        if ep <= WARMUP_EPOCHS:
            for pg in opt.param_groups: pg["lr"] = LR * ep / WARMUP_EPOCHS
        t0 = time.time()
        tr_loss, nan_s = train_epoch(model, loader, opt, DEVICE, bg_pool)
        if ep > WARMUP_EPOCHS: sched.step()
        cur_lr = opt.param_groups[0]["lr"]
        ss_auc, n_ss, preds_ep, _ = evaluate_ss(model, ss_eval, l2i, DEVICE)
        dt = time.time() - t0
        per_target = {}
        for sp in target_sps:
            if sp in l2i:
                ci = l2i[sp]
                yc = Y_ev[:, ci]
                if yc.sum() > 0 and yc.sum() < len(yc):
                    try: per_target[sp] = float(roc_auc_score(yc, preds_ep[:, ci]))
                    except: pass
        print(f"  ep {ep:02d}  lr {cur_lr:.5f}  loss {tr_loss:.4f}  val_SS {ss_auc:.4f}  "
              f"target=[{','.join(f'{s}:{a:.2f}' for s,a in per_target.items())}]  ({dt:.0f}s)", flush=True)
        if ss_auc > best["val_SS"]:
            best = {"val_SS": ss_auc, "epoch": ep, "per_target": per_target}
            torch.save({"state_dict": model.state_dict(), "epoch": ep, "val_SS": ss_auc},
                       OUT / "best_ckpt.pt")
    print(f"\nBest ep {best['epoch']}  val_SS {best['val_SS']:.4f}  base {ss_auc_base:.4f}  Δ {best['val_SS']-ss_auc_base:+.4f}")
    with open(OUT / "results.json", "w") as f:
        json.dump({"baseline_val_ss": ss_auc_base, "best": best,
                   "n_ext_kept": len(keep), "n_ext_total": len(ext_clips)}, f, indent=2, default=float)


if __name__ == "__main__":
    main()
