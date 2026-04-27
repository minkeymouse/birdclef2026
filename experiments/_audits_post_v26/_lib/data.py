"""Shared data loaders for SS audits — single source of truth for the
labeled-SS split, taxonomy mapping, Perch cache alignment, and aux features.
"""
from __future__ import annotations
import re
from pathlib import Path
from functools import lru_cache
import numpy as np
import pandas as pd

ROOT = Path("/data/birdclef2026")
DATA = ROOT / "data/birdclef-2026"
EXP43A = ROOT / "experiments/_data_pipelines/exp43a_outputs"
EXP76 = ROOT / "experiments/_data_pipelines/exp76_outputs"
EXP80 = ROOT / "experiments/_audits_post_v26/exp80_outputs"
MW = ROOT / "model-weights"

SEED = 42
N_CLS = 234
TAXA = ["Aves", "Amphibia", "Insecta", "Mammalia", "Reptilia"]
SR = 32000
N_WIN = 12  # 12 × 5-sec windows per 60-sec file

FNAME_RE = re.compile(r"BC2026_(?:Train|Test)_(\d+)_(S\d+)_(\d{8})_(\d{6})")


def parse_meta(fn: str) -> tuple[str | None, int]:
    m = FNAME_RE.match(fn)
    return (m.group(2), int(m.group(4)[:2])) if m else (None, -1)


def primary_labels() -> list[str]:
    return pd.read_csv(DATA / "sample_submission.csv").columns[1:].tolist()


def species_taxon_array() -> np.ndarray:
    primary = primary_labels()
    tax = pd.read_csv(DATA / "taxonomy.csv")
    l2t = dict(zip(tax.primary_label.astype(str), tax.class_name))
    return np.array([l2t.get(p, "?") for p in primary])


@lru_cache(maxsize=1)
def build_ss(seed: int = SEED, n_eval_files: int = 11):
    """Returns sc_g (DataFrame), Y (n × 234 uint8), primary (list[str]), l2i (dict).

    sc_g columns: filename, start, end, lbls, end_sec, row_id, site, hour, split.
    Split is file-stratified; first 11 of shuffled-files become 'eval', rest 'train'.
    Y[i, c] = 1 iff species `primary[c]` is positive in row i.
    """
    sc = pd.read_csv(DATA / "train_soundscapes_labels.csv").drop_duplicates().reset_index(drop=True)
    primary = primary_labels()

    def _parse(x):
        if pd.isna(x): return []
        return [t.strip() for t in str(x).split(";") if t.strip()]

    sc_g = (sc.groupby(["filename","start","end"])["primary_label"]
            .apply(lambda s: sorted({l for x in s for l in _parse(x)})).reset_index(name="lbls"))
    sc_g["end_sec"] = pd.to_timedelta(sc_g["end"]).dt.total_seconds().astype(int)
    sc_g["row_id"] = sc_g["filename"].str.replace(".ogg","",regex=False) + "_" + sc_g["end_sec"].astype(str)
    sc_g[["site","hour"]] = sc_g.filename.apply(lambda f: pd.Series(parse_meta(f)))

    rng = np.random.RandomState(seed)
    files = sorted(sc_g.filename.unique())
    rng.shuffle(files)
    eval_files = set(files[:n_eval_files])
    sc_g["split"] = ["eval" if f in eval_files else "train" for f in sc_g.filename]

    l2i = {p: i for i, p in enumerate(primary)}
    Y = np.zeros((len(sc_g), len(primary)), dtype=np.uint8)
    for i, labs in enumerate(sc_g.lbls):
        for l in labs:
            if l in l2i: Y[i, l2i[l]] = 1
    return sc_g, Y, primary, l2i


def get_taxon_y(Y: np.ndarray, sp_taxon: np.ndarray, taxon: str) -> np.ndarray:
    """Per-row binary GT: 1 iff the row contains any species of the given taxon."""
    return (Y[:, sp_taxon == taxon].sum(axis=1) > 0).astype(np.uint8)


def aux_matrix(sc_g: pd.DataFrame, sites: list[str] | None = None) -> tuple[np.ndarray, list[str]]:
    """Auxiliary feature for iVAE prior: (n × (n_sites + 1)) — site one-hot + hour/24."""
    if sites is None:
        sites = sorted(sc_g.site.unique())
    s2i = {s: i for i, s in enumerate(sites)}
    aux = np.zeros((len(sc_g), len(sites) + 1), dtype=np.float32)
    for i, r in sc_g.iterrows():
        if r.site in s2i:
            aux[i, s2i[r.site]] = 1.0
        aux[i, -1] = r.hour / 24.0
    return aux, sites


def load_labeled_mel() -> np.ndarray:
    """Returns (739, 16, 128) pooled mel from exp76 cache."""
    return np.load(EXP76 / "mel_cache.npz")["mel"]


def load_perch_emb_labeled() -> np.ndarray:
    """Perch 1536-d emb aligned with labeled SS rows. Cached to disk after first build."""
    cache = EXP80 / "perch_emb_labeled.npz"
    if cache.exists():
        return np.load(cache)["P"]
    print(f"  building Perch labeled cache (one-time, ~5-30s decompression)...")
    sc_g, _, _, _ = build_ss()
    perch = np.load(EXP43A / "perch_ss_all.npz")
    emb_full = perch["emb"]
    perch_meta = pd.read_parquet(EXP43A / "perch_ss_all_meta.parquet")
    rid2i = {r: i for i, r in enumerate(perch_meta["row_id"].values)}
    P = np.zeros((len(sc_g), 1536), dtype=np.float32)
    for i, rid in enumerate(sc_g.row_id.values):
        j = rid2i.get(rid, -1)
        if j >= 0: P[i] = emb_full[j]
    cache.parent.mkdir(exist_ok=True, parents=True)
    np.savez_compressed(cache, P=P)
    return P


def load_perch_scores_labeled() -> np.ndarray:
    """Perch 234-d sigmoid prob aligned with labeled SS rows."""
    cache = EXP80 / "perch_prob_labeled.npz"
    if cache.exists():
        return np.load(cache)["prob"]
    print("  building Perch score labeled cache (one-time)...")
    sc_g, _, _, _ = build_ss()
    perch = np.load(EXP43A / "perch_ss_all.npz")
    sc_full = perch["scores"]
    perch_meta = pd.read_parquet(EXP43A / "perch_ss_all_meta.parquet")
    rid2i = {r: i for i, r in enumerate(perch_meta["row_id"].values)}
    S = np.zeros((len(sc_g), N_CLS), dtype=np.float32)
    for i, rid in enumerate(sc_g.row_id.values):
        j = rid2i.get(rid, -1)
        if j >= 0: S[i] = sc_full[j]
    P = (1.0 / (1.0 + np.exp(-S))).astype(np.float32)
    cache.parent.mkdir(exist_ok=True, parents=True)
    np.savez_compressed(cache, prob=P)
    return P
