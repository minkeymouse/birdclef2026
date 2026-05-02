"""Data builders + Dataset classes."""
import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .constants import (DATA, DATA25, SR, WINDOW_SEC, CLIP_SAMPLES, FILE_SAMPLES,
                          SECONDARY_WEIGHT, SEED, EVAL_SS_N_FILES, N_CLS)
from .audio import load_audio, random_crop, center_crop


def build_primaries():
    sub = pd.read_csv(DATA / "sample_submission.csv")
    primary = sub.columns[1:].tolist()
    return primary, {c: i for i, c in enumerate(primary)}


def build_ta_combined(l2i, val_frac=0.20, seed=SEED, include_2025=True):
    """Combine 2026 + 2025 train_audio (overlap species only)."""
    df_2026 = pd.read_csv(DATA / "train.csv")
    df_2026 = df_2026[df_2026["primary_label"].astype(str).isin(l2i)].reset_index(drop=True)
    df_2026["primary_idx"] = df_2026["primary_label"].astype(str).map(l2i)
    df_2026["audio_root"] = str(DATA / "train_audio")

    if include_2025 and (DATA25 / "train.csv").exists():
        df_2025 = pd.read_csv(DATA25 / "train.csv")
        df_2025 = df_2025[df_2025["primary_label"].astype(str).isin(l2i)].reset_index(drop=True)
        df_2025["primary_idx"] = df_2025["primary_label"].astype(str).map(l2i)
        df_2025["audio_root"] = str(DATA25 / "train_audio")
        overlap = set(df_2025.primary_label) & set(df_2026.primary_label)
        df_2025 = df_2025[df_2025.primary_label.isin(overlap)].reset_index(drop=True)
        df_combined = pd.concat([df_2026, df_2025], ignore_index=True)
    else:
        df_combined = df_2026

    rng = np.random.RandomState(seed)
    val_idx, train_idx = [], []
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
    """Labeled SS — 11 eval files holdout, rest train."""
    sc_raw = pd.read_csv(DATA / "train_soundscapes_labels.csv").drop_duplicates().reset_index(drop=True)
    def parse(x):
        if pd.isna(x): return []
        return [t.strip() for t in str(x).split(";") if t.strip()]
    sc_g = (sc_raw.groupby(["filename", "start", "end"])["primary_label"]
            .apply(lambda s: sorted({l for x in s for l in parse(x)})).reset_index(name="lbls"))
    sc_g["end_sec"] = pd.to_timedelta(sc_g["end"]).dt.total_seconds().astype(int)
    sc_g["row_id"] = sc_g["filename"].str.replace(".ogg", "", regex=False) + "_" + sc_g["end_sec"].astype(str)
    rng = np.random.RandomState(SEED)
    files = sorted(sc_g["filename"].unique()); rng.shuffle(files)
    eval_files = set(files[:EVAL_SS_N_FILES])
    ss_train = sc_g[~sc_g.filename.isin(eval_files)].reset_index(drop=True)
    ss_eval = sc_g[sc_g.filename.isin(eval_files)].reset_index(drop=True)
    return ss_train, ss_eval


def build_pseudo_ss(csv_path, l2i, eval_files=None):
    """Aggregate pseudo-label CSV into per-(filename, end_sec) multi-label rows.

    csv_path: path to pseudo-label CSV (must have filename, start, end, primary_label).
    eval_files: set of filenames to exclude (avoid leakage to eval).
    """
    df = pd.read_csv(csv_path)
    df["start"] = df["start"].astype(int)
    df["end"] = df["end"].astype(int)
    grouped = (df.groupby(["filename", "start", "end"])
                  .agg(lbls=("primary_label", lambda s: sorted(set(s)))).reset_index())
    grouped["end_sec"] = grouped["end"]
    grouped["row_id"] = grouped["filename"].str.replace(".ogg", "", regex=False) + "_" + grouped["end_sec"].astype(str)
    if eval_files:
        grouped = grouped[~grouped.filename.isin(eval_files)].reset_index(drop=True)
    # Optional: also load v33 score per (row, class) for masking later
    if "v33_score" in df.columns:
        # Build lookup: (filename, end_sec, class) -> v33 score
        score_lookup = {}
        for _, r in df.iterrows():
            key = (r.filename, int(r.end), r.primary_label)
            score_lookup[key] = float(r.v33_score)
        grouped.attrs["score_lookup"] = score_lookup
    return grouped


class TADataset(Dataset):
    def __init__(self, df, l2i, train=True):
        self.df = df.reset_index(drop=True); self.l2i = l2i; self.train = train
        self.n_cls = len(l2i)

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = Path(row.audio_root) / row.filename
        wav = load_audio(path, CLIP_SAMPLES * 2)
        wav = random_crop(wav, CLIP_SAMPLES) if self.train else center_crop(wav, CLIP_SAMPLES)
        y = np.zeros(self.n_cls, dtype=np.float32)
        y[row.primary_idx] = 1.0
        for sl in row.secondary_list:
            if sl in self.l2i: y[self.l2i[sl]] = SECONDARY_WEIGHT
        # Full mask = all 234 are valid (no masking for hard labels)
        mask = np.ones(self.n_cls, dtype=np.float32)
        return torch.from_numpy(wav), torch.from_numpy(y), torch.from_numpy(mask), int(row.primary_idx), 1


class SSDataset(Dataset):
    """Hard-labeled SS rows."""
    def __init__(self, ss_df, l2i, train=True):
        self.ss = ss_df.reset_index(drop=True); self.l2i = l2i; self.train = train
        self.n_cls = len(l2i)

    def __len__(self): return len(self.ss)

    def __getitem__(self, idx):
        row = self.ss.iloc[idx]
        p = DATA / "train_soundscapes" / row.filename
        wav = load_audio(p, FILE_SAMPLES)
        end_sec = int(row.end_sec)
        target_c = (end_sec - WINDOW_SEC / 2) * SR
        cs = int(max(0, target_c - CLIP_SAMPLES / 2))
        cs = min(cs, FILE_SAMPLES - CLIP_SAMPLES)
        if self.train:
            cs = int(cs + random.randint(-SR, SR)); cs = max(0, min(cs, FILE_SAMPLES - CLIP_SAMPLES))
        clip = wav[cs:cs + CLIP_SAMPLES]
        if len(clip) < CLIP_SAMPLES:
            clip = np.pad(clip, (0, CLIP_SAMPLES - len(clip)))
        y = np.zeros(self.n_cls, dtype=np.float32)
        for l in row.lbls:
            if l in self.l2i: y[self.l2i[l]] = 1.0
        # Full mask = all 234 valid
        mask = np.ones(self.n_cls, dtype=np.float32)
        return torch.from_numpy(clip.astype(np.float32)), torch.from_numpy(y), torch.from_numpy(mask), -1, 0


class SSPseudoDataset(Dataset):
    """Pseudo-labeled SS rows with optional per-class loss masking.

    If `score_lookup` provided, uses it to compute mask:
        confident positive (v33 > tau_pos): label=1, mask=1
        confident negative (v33 < tau_neg): label=0, mask=1
        uncertain (in between):              mask=0 (skip in BCE loss)
    Otherwise: all classes valid (mask=ones), label = present in lbls or 0.
    """
    def __init__(self, pseudo_df, l2i, train=True, score_lookup=None,
                  tau_pos=0.5, tau_neg=0.05):
        self.ss = pseudo_df.reset_index(drop=True); self.l2i = l2i; self.train = train
        self.n_cls = len(l2i)
        self.score_lookup = score_lookup
        self.tau_pos = tau_pos
        self.tau_neg = tau_neg
        self.idx_to_label = [None] * self.n_cls
        for lbl, idx in l2i.items():
            self.idx_to_label[idx] = lbl

    def __len__(self): return len(self.ss)

    def __getitem__(self, idx):
        row = self.ss.iloc[idx]
        p = DATA / "train_soundscapes" / row.filename
        wav = load_audio(p, FILE_SAMPLES)
        end_sec = int(row.end_sec)
        target_c = (end_sec - WINDOW_SEC / 2) * SR
        cs = int(max(0, target_c - CLIP_SAMPLES / 2))
        cs = min(cs, FILE_SAMPLES - CLIP_SAMPLES)
        if self.train:
            cs = int(cs + random.randint(-SR, SR)); cs = max(0, min(cs, FILE_SAMPLES - CLIP_SAMPLES))
        clip = wav[cs:cs + CLIP_SAMPLES]
        if len(clip) < CLIP_SAMPLES:
            clip = np.pad(clip, (0, CLIP_SAMPLES - len(clip)))

        y = np.zeros(self.n_cls, dtype=np.float32)
        for l in row.lbls:
            if l in self.l2i: y[self.l2i[l]] = 1.0

        if self.score_lookup is None:
            mask = np.ones(self.n_cls, dtype=np.float32)
        else:
            # For each class, look up v33 score
            mask = np.zeros(self.n_cls, dtype=np.float32)
            for c, label in enumerate(self.idx_to_label):
                key = (row.filename, end_sec, label)
                v33_c = self.score_lookup.get(key, 0.0)
                if v33_c > self.tau_pos or v33_c < self.tau_neg:
                    mask[c] = 1.0

        return (torch.from_numpy(clip.astype(np.float32)),
                torch.from_numpy(y), torch.from_numpy(mask), -1, 0)
