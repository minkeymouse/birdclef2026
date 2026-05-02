#!/usr/bin/env python3
"""exp121 — exp50 fine-tune with aggressive cross-region synthesis.

Builds on exp50 (HGNet-B0 + 2025 BG mix, val_SS 0.838, LB +0.001 only positive).

Changes from exp50:
  1. BG_MIX_P 0.4 → 0.7 (more cross-region mixing)
  2. Per-class boost: for non-Aves classes, BG_MIX_P → 0.85
  3. Add 2025 train_audio (41 overlap species, ~13k clips from exp120)
  4. Continue training from exp50 best ckpt (~8 epochs)
  5. Cross-region mixup partner: 60% of the time the partner clip is from 2025 BG
     (not from TA) — forces MORE cross-region acoustic exposure

Hypothesis: exp50 lifted LB by introducing one new site-invariance source
(2025 BG at low p). Stronger cross-region exposure should amplify the
mechanism, especially for rare classes where Perch has weakest prior.

Time: ~2-3h GPU.
"""
from __future__ import annotations
import json, random, re, time
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import soundfile as sf
import timm
import torch, torch.nn as nn, torch.nn.functional as F
import torchaudio
from sklearn.metrics import roc_auc_score
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

ROOT = Path("/data/birdclef2026")
DATA = ROOT / "data" / "birdclef-2026"
DATA25 = ROOT / "data" / "birdclef-2025"
BG_PATH = ROOT / "experiments/_data_pipelines/exp49_outputs/bg_quiet_2025.npz"
TA25_PATH = ROOT / "experiments/_data_pipelines/exp120_outputs/ta25_perch.npz"  # for label index, not embeddings
EXP50_CKPT = ROOT / "experiments/_data_pipelines/exp50_outputs/best_ckpt.pt"
OUT = ROOT / "experiments/_data_pipelines/exp121_outputs"
OUT.mkdir(parents=True, exist_ok=True)

SR = 32000; CLIP_SEC = 20; CLIP_SAMPLES = SR * CLIP_SEC
WINDOW_SEC = 5; WINDOW_SAMPLES = SR * WINDOW_SEC
FILE_SAMPLES = SR * 60
N_FFT = 2048; HOP = 512; N_MELS = 128; FMIN = 50; FMAX = 14000

BATCH_SIZE = 32; EPOCHS = 10; LR = 3e-4; WD = 1e-2  # lower LR for fine-tune
NUM_WORKERS = 4
MIXUP_ALPHA = 0.5; MIXUP_P = 0.6
BG_MIX_P_AVES = 0.5     # was 0.4
BG_MIX_P_NON_AVES = 0.85   # NEW: aggressive for non-Aves
BG_AS_PARTNER_P = 0.6  # NEW: probability that mixup partner is BG (not TA)
BG_ALPHA_LO, BG_ALPHA_HI = 0.3, 0.7
SECONDARY_WEIGHT = 0.3
SPEC_FREQ_MASK = 16; SPEC_TIME_MASK = 40
EVAL_SS_N_FILES = 11; SEED = 42
BACKBONE = "hgnetv2_b0.ssld_stage2_ft_in1k"; DEVICE = "cuda"


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def build_primaries():
    sub = pd.read_csv(DATA / "sample_submission.csv")
    primary = sub.columns[1:].tolist()
    return primary, {c: i for i, c in enumerate(primary)}


def build_ta_combined(l2i, val_frac=0.20, seed=SEED):
    """Combine 2026 train_audio + 2025 train_audio for 41 overlap species."""
    df_2026 = pd.read_csv(DATA / "train.csv")
    df_2026 = df_2026[df_2026["primary_label"].astype(str).isin(l2i)].reset_index(drop=True)
    df_2026["primary_idx"] = df_2026["primary_label"].astype(str).map(l2i)
    df_2026["data_year"] = 2026
    df_2026["audio_root"] = str(DATA / "train_audio")

    # Load 2025 train_audio for overlap species (only after exp120 confirms paths)
    if (DATA25 / "train.csv").exists():
        df_2025 = pd.read_csv(DATA25 / "train.csv")
        df_2025 = df_2025[df_2025["primary_label"].astype(str).isin(l2i)].reset_index(drop=True)
        df_2025["primary_idx"] = df_2025["primary_label"].astype(str).map(l2i)
        df_2025["data_year"] = 2025
        df_2025["audio_root"] = str(DATA25 / "train_audio")
        # Filter: only 41 overlap species
        overlap = set(df_2025.primary_label) & set(df_2026.primary_label)
        df_2025 = df_2025[df_2025.primary_label.isin(overlap)].reset_index(drop=True)
        df_combined = pd.concat([df_2026, df_2025], ignore_index=True)
        print(f"  Combined TA: 2026={len(df_2026)}, 2025={len(df_2025)} (overlap species), total={len(df_combined)}")
    else:
        df_combined = df_2026
        print(f"  TA 2026 only: {len(df_combined)}")

    # Stratified split
    rng = np.random.RandomState(seed); val_idx = []; train_idx = []
    for lbl, g in df_combined.groupby("primary_label"):
        g_idx = g.index.tolist(); rng.shuffle(g_idx)
        n_val = max(1, int(len(g_idx) * val_frac)) if len(g_idx) >= 5 else 0
        val_idx.extend(g_idx[:n_val]); train_idx.extend(g_idx[n_val:])
    train_df = df_combined.loc[train_idx].reset_index(drop=True)
    val_df = df_combined.loc[val_idx].reset_index(drop=True)

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
    ss_train = sc_g[~sc_g.filename.isin(eval_files)].reset_index(drop=True)
    ss_eval = sc_g[sc_g.filename.isin(eval_files)].reset_index(drop=True)
    return ss_train, ss_eval


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


def get_taxon_array(primary, l2i):
    """Return per-class taxon array."""
    tax = pd.read_csv(DATA / "taxonomy.csv")
    sp2tax = dict(zip(tax.primary_label.astype(str), tax.class_name))
    return np.array([sp2tax.get(p, "Aves") for p in primary])


class TADataset(Dataset):
    def __init__(self, df, l2i, train=True):
        self.df = df.reset_index(drop=True); self.l2i = l2i; self.train = train
        self.n_cls = len(l2i)
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        audio_root = row.audio_root if "audio_root" in row else str(DATA / "train_audio")
        path = Path(audio_root) / row.filename
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


class MelExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS,
            f_min=FMIN, f_max=FMAX, power=2.0, center=True)
        self.adb = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)
    def forward(self, x): return self.adb(self.mel(x)).unsqueeze(1)


class SpecAug(nn.Module):
    def __init__(self, f=SPEC_FREQ_MASK, t=SPEC_TIME_MASK):
        super().__init__()
        self.fm = torchaudio.transforms.FrequencyMasking(freq_mask_param=f)
        self.tm = torchaudio.transforms.TimeMasking(time_mask_param=t)
    def forward(self, x): return self.tm(self.fm(x))


class SEDHead(nn.Module):
    def __init__(self, feat_dim, n_cls):
        super().__init__()
        self.att = nn.Conv1d(feat_dim, n_cls, 1)
        self.cla = nn.Conv1d(feat_dim, n_cls, 1)
    def forward(self, x):
        a = self.att(x); c = self.cla(x)
        w = torch.softmax(a, dim=-1)
        return (w * c).sum(-1), c.max(-1).values


class SEDModel(nn.Module):
    def __init__(self, n_cls):
        super().__init__()
        self.mel = MelExtractor()
        self.bn0 = nn.BatchNorm2d(N_MELS)
        self.specaug = SpecAug()
        self.backbone = timm.create_model(BACKBONE, pretrained=True, in_chans=1, num_classes=0, global_pool='')
        with torch.no_grad():
            f = self.backbone(torch.zeros(1, 1, N_MELS, 100))
        self.head = SEDHead(f.shape[1], n_cls)
    def forward(self, x, train=True):
        m = self.mel(x); m = m.transpose(1, 2); m = self.bn0(m); m = m.transpose(1, 2)
        if train: m = self.specaug(m)
        f = self.backbone(m)
        f = f.mean(dim=2) if f.dim() == 4 else f
        clip, fmax = self.head(f)
        return clip, fmax


def aggressive_mixup(x, y, primary_idx, is_ta, bg_pool, taxon_array, l2i,
                      bg_mix_p_aves=BG_MIX_P_AVES, bg_mix_p_non_aves=BG_MIX_P_NON_AVES,
                      bg_as_partner_p=BG_AS_PARTNER_P, alpha=MIXUP_ALPHA):
    """Per-row aggressive mixup. Non-Aves rows get higher BG mix prob."""
    B = x.size(0)
    out_x = x.clone(); out_y = y.clone()
    for i in range(B):
        # Per-class BG mix probability
        if int(primary_idx[i]) >= 0:
            cls_taxon = taxon_array[int(primary_idx[i])]
            bg_mix_p = bg_mix_p_non_aves if cls_taxon != "Aves" else bg_mix_p_aves
        else:
            bg_mix_p = bg_mix_p_aves

        # Decide partner type
        use_bg_partner = (bg_pool is not None and random.random() < bg_mix_p)
        if use_bg_partner:
            # Sample BG window, repeat to CLIP_SAMPLES
            bg_idx = random.randint(0, len(bg_pool) - 1)
            bg_5s = bg_pool[bg_idx]  # (160000,)
            reps = CLIP_SAMPLES // len(bg_5s) + 1
            bg_partner = np.tile(bg_5s, reps)[:CLIP_SAMPLES].astype(np.float32)
            bg_t = torch.from_numpy(bg_partner).to(x.device)
            lam_alpha = random.uniform(BG_ALPHA_LO, BG_ALPHA_HI)
            out_x[i] = lam_alpha * x[i] + (1 - lam_alpha) * bg_t
            # Label stays as-is (BG has no labels)
        else:
            # Standard mixup with TA partner (within batch)
            if random.random() < MIXUP_P:
                lam = np.random.beta(alpha, alpha)
                lam = max(lam, 1 - lam)
                j = random.randint(0, B - 1)
                out_x[i] = lam * x[i] + (1 - lam) * x[j]
                out_y[i] = lam * y[i] + (1 - lam) * y[j]
    return out_x, out_y


def main():
    print("=== exp121: Aggressive cross-region synthesis (continue exp50) ===\n", flush=True)
    set_seed(SEED)
    primary, l2i = build_primaries()
    n_cls = len(primary)

    print("Loading data...")
    ta_train, ta_val = build_ta_combined(l2i)
    ss_train, ss_eval = build_ss_splits(l2i)
    print(f"  ta_train {len(ta_train)}, ta_val {len(ta_val)}")
    print(f"  ss_train {len(ss_train)}, ss_eval {len(ss_eval)}")

    # 2025 BG pool
    bg_pool = None
    if BG_PATH.exists():
        bg = np.load(BG_PATH)
        bg_pool = bg["windows"]
        print(f"  BG pool: {bg_pool.shape}")
    else:
        print("  WARN: no BG pool — disabling cross-region mixing")

    taxon_array = get_taxon_array(primary, l2i)
    n_non_aves_train = (taxon_array[ta_train.primary_idx.astype(int)] != "Aves").sum()
    print(f"  Non-Aves training rows: {n_non_aves_train}")

    # Datasets
    ta_ds = TADataset(ta_train, l2i, train=True)
    ta_val_ds = TADataset(ta_val, l2i, train=False)
    ss_train_ds = SSDataset(ss_train, l2i, train=True)
    ss_eval_ds = SSDataset(ss_eval, l2i, train=False)

    # Combined train via WeightedRandomSampler
    from torch.utils.data import ConcatDataset
    combined = ConcatDataset([ta_ds, ss_train_ds])
    # Up-weight SS rows (multi-label closer to test)
    ss_weight = 5.0
    weights = np.concatenate([
        np.ones(len(ta_ds), dtype=np.float32),
        np.full(len(ss_train_ds), ss_weight, dtype=np.float32),
    ])
    sampler = WeightedRandomSampler(weights, num_samples=len(combined), replacement=True)
    train_loader = DataLoader(combined, batch_size=BATCH_SIZE, sampler=sampler,
                               num_workers=NUM_WORKERS, pin_memory=True)
    val_loader_ta = DataLoader(ta_val_ds, batch_size=BATCH_SIZE, shuffle=False,
                                 num_workers=NUM_WORKERS, pin_memory=True)
    val_loader_ss = DataLoader(ss_eval_ds, batch_size=BATCH_SIZE, shuffle=False,
                                 num_workers=NUM_WORKERS, pin_memory=True)

    # Model — load from exp50 ckpt
    print("\nLoading exp50 ckpt as init...")
    model = SEDModel(n_cls).to(DEVICE)
    if EXP50_CKPT.exists():
        ckpt = torch.load(str(EXP50_CKPT), map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        print(f"  Loaded exp50 ckpt (val_TA={ckpt.get('val_auc', '?')})")
    else:
        print("  WARN: exp50 ckpt missing — training from scratch")

    optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=EPOCHS, eta_min=LR/20)

    best_val_ss = 0.0; best_state = None; history = []

    for ep in range(EPOCHS):
        model.train()
        ep_loss = 0.0; n_batches = 0
        t0 = time.time()
        for batch in train_loader:
            x, y, primary_idx, is_ta = batch
            x = x.to(DEVICE, non_blocking=True); y = y.to(DEVICE, non_blocking=True)
            x_m, y_m = aggressive_mixup(x, y, primary_idx, is_ta, bg_pool, taxon_array, l2i)
            optim.zero_grad()
            clip, fmax = model(x_m, train=True)
            loss_clip = F.binary_cross_entropy_with_logits(clip, y_m)
            loss_fmax = F.binary_cross_entropy_with_logits(fmax, y_m)
            loss = 0.5 * loss_clip + 0.5 * loss_fmax
            loss.backward()
            optim.step()
            ep_loss += loss.item(); n_batches += 1
        scheduler.step()

        # Eval
        model.eval()
        all_y_ta, all_p_ta = [], []
        all_y_ss, all_p_ss = [], []
        with torch.no_grad():
            for x, y, _, _ in val_loader_ta:
                x = x.to(DEVICE); clip, _ = model(x, train=False)
                all_y_ta.append(y.numpy()); all_p_ta.append(torch.sigmoid(clip).cpu().numpy())
            for x, y, _, _ in val_loader_ss:
                x = x.to(DEVICE); clip, _ = model(x, train=False)
                all_y_ss.append(y.numpy()); all_p_ss.append(torch.sigmoid(clip).cpu().numpy())
        all_y_ta = np.concatenate(all_y_ta); all_p_ta = np.concatenate(all_p_ta)
        all_y_ss = np.concatenate(all_y_ss); all_p_ss = np.concatenate(all_p_ss)

        def macro_auc(y, p):
            aucs = []
            for c in range(n_cls):
                if y[:, c].sum() == 0 or y[:, c].sum() == len(y): continue
                try: aucs.append(roc_auc_score(y[:, c], p[:, c]))
                except ValueError: pass
            return float(np.mean(aucs)) if aucs else float('nan'), len(aucs)

        val_ta, n_eval_ta = macro_auc(all_y_ta, all_p_ta)
        val_ss, n_eval_ss = macro_auc(all_y_ss, all_p_ss)
        elapsed = time.time() - t0
        print(f"ep {ep:02d}  loss {ep_loss/n_batches:.4f}  val_TA {val_ta:.4f} ({n_eval_ta})  val_SS {val_ss:.4f} ({n_eval_ss})  ({elapsed/60:.1f} min)", flush=True)

        if val_ss > best_val_ss:
            best_val_ss = val_ss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            torch.save({"state_dict": best_state, "val_TA": val_ta, "val_SS": val_ss,
                          "epoch": ep, "config": {"backbone": BACKBONE, "bg_mix_p_aves": BG_MIX_P_AVES,
                                                    "bg_mix_p_non_aves": BG_MIX_P_NON_AVES, "seed": SEED}},
                         OUT / "best_ckpt.pt")
            print(f"  -> saved best ckpt @ ep{ep:02d}")
        history.append({"epoch": ep, "loss": ep_loss/n_batches, "val_TA": val_ta, "val_SS": val_ss})

    with open(OUT / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nDone. Best val_SS: {best_val_ss:.4f}")
    print(f"  exp50 baseline val_SS: 0.838")


if __name__ == "__main__":
    main()
