"""
BirdCLEF+ 2026 — Perch v2 + Bayesian Priors + Embedding Probes
Based on public 0.910 notebook by chiranjithdharma.
Pipeline: Perch v2 embeddings → prior fusion → PCA + LogReg probes → Gaussian smoothing
"""
import os, platform, sys
print('Python :', sys.version)
print('OS     :', platform.system(), platform.release())
print('CWD    :', os.getcwd())

# ── Install TF 2.20 ──
import subprocess
from pathlib import Path

# Try multiple known wheel locations
_WHL_CANDIDATES = [
    Path('/kaggle/input/notebooks/kdmitrie/bc26-tensorflow-2-20-0/wheel'),
    Path('/kaggle/input/notebooks/kdmitrie/bc26-tensorflow-2-20-0'),
    Path('/kaggle/input/notebooks/ashok205/tf-wheels/tf_wheels'),
    Path('/kaggle/input/notebooks/ashok205/tf-wheels'),
    Path('/kaggle/input/bc26-tensorflow-2-20-0/wheel'),
    Path('/kaggle/input/bc26-tensorflow-2-20-0'),
    Path('/kaggle/input/tf-wheels/tf_wheels'),
    Path('/kaggle/input/tf-wheels'),
]

_WHL = None
for candidate in _WHL_CANDIDATES:
    tb_whl = candidate / 'tensorboard-2.20.0-py3-none-any.whl'
    if tb_whl.exists():
        _WHL = candidate
        print(f'Found TF wheels at: {_WHL}')
        break

if _WHL is None:
    # List what we can see to debug
    print('DEBUG: Listing /kaggle/input/ contents:')
    for p in sorted(Path('/kaggle/input').rglob('*.whl')):
        print(f'  {p}')
    raise RuntimeError('Could not find TF 2.20 wheel files in any known location')

subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-deps',
    str(_WHL / 'tensorboard-2.20.0-py3-none-any.whl')], check=True)

# Find the TF wheel (filename may vary)
tf_whls = list(_WHL.glob('tensorflow-2.20.0*.whl'))
if tf_whls:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-deps',
        str(tf_whls[0])], check=True)
else:
    raise RuntimeError(f'No tensorflow-2.20.0*.whl found in {_WHL}')

print('Installed TF 2.20.0 from Kaggle dataset wheels.')

# ── Imports ──
import gc
import json
import random
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import tensorflow as tf

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # force CPU
tf.experimental.numpy.experimental_enable_numpy_behavior()

print('TensorFlow :', tf.__version__)
print('NumPy      :', np.__version__)
print('Pandas     :', pd.__version__)

# ── Settings ──
class Settings:
    MODE = 'submit'
    SEED = 42

    # Competition paths
    _KAGGLE_BASE = Path('/kaggle/input/competitions/birdclef-2026')
    _LOCAL_BASE  = Path('../dataset')
    BASE = _KAGGLE_BASE if _KAGGLE_BASE.exists() else _LOCAL_BASE

    # Perch v2 SavedModel
    MODEL_DIR = Path(
        '/kaggle/input/models/google/bird-vocalization-classifier'
        '/tensorflow2/perch_v2_cpu/1'
    )

    # Perch cache
    _CACHE_CANDIDATES = [
        Path('/kaggle/input/datasets/jaejohn/perch-meta'),
        Path('../input'),
        Path('/kaggle/working/cache'),
    ]
    CACHE_DIR = next(
        (
            d for d in _CACHE_CANDIDATES
            if (d / 'full_perch_meta.parquet').exists()
            and (d / 'full_perch_arrays.npz').exists()
        ),
        None,
    )
    CACHE_EXISTS = CACHE_DIR is not None
    WORK_CACHE_DIR = Path('/kaggle/working/cache')

    # Audio
    SR             = 32_000
    WINDOW_SEC     = 5
    WINDOW_SAMPLES = SR * WINDOW_SEC
    FILE_SAMPLES   = 60 * SR
    N_WINDOWS      = 12
    BATCH_FILES    = 16
    DRYRUN_N_FILES = 20

    # Prior fusion (frozen from OOF tuning)
    LAMBDA_EVENT         = 0.4
    LAMBDA_TEXTURE       = 1.0
    LAMBDA_PROXY_TEXTURE = 0.8
    SMOOTH_TEXTURE_ALPHA = 0.35

    # Embedding probe (frozen from OOF tuning)
    PROBE_PCA_DIM = 32
    PROBE_MIN_POS = 8
    PROBE_C       = 0.25
    PROBE_ALPHA   = 0.40


CFG = Settings()
CFG.WORK_CACHE_DIR.mkdir(parents=True, exist_ok=True)

print(f'MODE         : {CFG.MODE}')
print(f'BASE         : {CFG.BASE}')
print(f'CACHE_EXISTS : {CFG.CACHE_EXISTS}')
print(f'CACHE_DIR    : {CFG.CACHE_DIR}')

# ── Seed ──
def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

seed_everything(CFG.SEED)

# ── Load Data ──
taxonomy        = pd.read_csv(CFG.BASE / 'taxonomy.csv')
train_meta      = pd.read_csv(CFG.BASE / 'train.csv')
soundscape_raw  = pd.read_csv(CFG.BASE / 'train_soundscapes_labels.csv')
sample_sub      = pd.read_csv(CFG.BASE / 'sample_submission.csv')

soundscape_lbls = soundscape_raw.drop_duplicates().reset_index(drop=True)

PRIMARY_LABELS = sample_sub.columns[1:].tolist()
N_CLASSES      = len(PRIMARY_LABELS)
label_to_idx   = {c: i for i, c in enumerate(PRIMARY_LABELS)}

print(f'taxonomy species     : {len(taxonomy)}')
print(f'train recordings     : {len(train_meta):,}')
print(f'soundscape rows      : {len(soundscape_lbls):,} unique '
      f'(dropped {len(soundscape_raw) - len(soundscape_lbls):,} duplicates)')
print(f'submission classes   : {N_CLASSES}')

# ── Parse Soundscape Labels ──
FNAME_RE = re.compile(
    r'BC2026_(?:Train|Test)_(\d+)_(S\d+)_(\d{8})_(\d{6})\.ogg'
)

def parse_labels(x):
    if pd.isna(x):
        return []
    return [t.strip() for t in str(x).split(';') if t.strip()]

def union_labels(series):
    return sorted(set(lbl for x in series for lbl in parse_labels(x)))

def parse_soundscape_filename(name):
    m = FNAME_RE.match(name)
    if not m:
        return {'site': None, 'hour_utc': -1}
    _, site, _, hms = m.groups()
    return {'site': site, 'hour_utc': int(hms[:2])}

sc_clean = (
    soundscape_lbls
    .groupby(['filename', 'start', 'end'])['primary_label']
    .apply(union_labels)
    .reset_index(name='label_list')
)
sc_clean['end_sec'] = pd.to_timedelta(sc_clean['end']).dt.total_seconds().astype(int)
sc_clean['row_id']  = (
    sc_clean['filename'].str.replace('.ogg', '', regex=False)
    + '_' + sc_clean['end_sec'].astype(str)
)
meta_cols = sc_clean['filename'].apply(parse_soundscape_filename).apply(pd.Series)
sc_clean  = pd.concat([sc_clean, meta_cols], axis=1)

wpf        = sc_clean.groupby('filename').size()
full_files = sorted(wpf[wpf == CFG.N_WINDOWS].index.tolist())
sc_clean['file_fully_labeled'] = sc_clean['filename'].isin(full_files)

Y_SC = np.zeros((len(sc_clean), N_CLASSES), dtype=np.uint8)
for i, labels in enumerate(sc_clean['label_list']):
    for lbl in labels:
        if lbl in label_to_idx:
            Y_SC[i, label_to_idx[lbl]] = 1

full_truth = (
    sc_clean[sc_clean['file_fully_labeled']]
    .sort_values(['filename', 'end_sec'])
    .reset_index(drop=False)
)
Y_FULL_TRUTH = Y_SC[full_truth['index'].to_numpy()]

print(f'Fully-labeled files : {len(full_files)}')
print(f'Trusted windows     : {len(full_truth)}')
print(f'Active classes      : {int((Y_FULL_TRUTH.sum(axis=0) > 0).sum())}')

# ── Load Perch v2 ──
print('Loading Perch model...')
birdclassifier = tf.saved_model.load(str(CFG.MODEL_DIR))
infer_fn       = birdclassifier.signatures['serving_default']
print('Perch loaded.')

bc_labels = (
    pd.read_csv(CFG.MODEL_DIR / 'assets' / 'labels.csv')
    .reset_index()
    .rename(columns={'index': 'bc_index', 'inat2024_fsd50k': 'scientific_name'})
)
NO_LABEL_INDEX = len(bc_labels)

taxonomy_ = taxonomy.copy()
taxonomy_['scientific_name'] = taxonomy_['scientific_name'].astype(str)
mapping = taxonomy_.merge(
    bc_labels[['scientific_name', 'bc_index']],
    on='scientific_name', how='left'
)
mapping['bc_index'] = mapping['bc_index'].fillna(NO_LABEL_INDEX).astype(int)

label_to_bc   = mapping.set_index('primary_label')['bc_index']
BC_INDICES    = np.array([int(label_to_bc.loc[c]) for c in PRIMARY_LABELS], dtype=np.int32)

MAPPED_MASK       = BC_INDICES != NO_LABEL_INDEX
MAPPED_POS        = np.where(MAPPED_MASK)[0].astype(np.int32)
UNMAPPED_POS      = np.where(~MAPPED_MASK)[0].astype(np.int32)
MAPPED_BC_INDICES = BC_INDICES[MAPPED_MASK].astype(np.int32)

print(f'Mapped   : {MAPPED_MASK.sum()} / {N_CLASSES}')
print(f'Unmapped : {(~MAPPED_MASK).sum()}')

# ── Species classification (texture vs event) ──
CLASS_NAME_MAP = taxonomy_.set_index('primary_label')['class_name'].to_dict()
TEXTURE_TAXA   = {'Amphibia', 'Insecta'}
ACTIVE_CLASSES = [PRIMARY_LABELS[i] for i in np.where(Y_SC.sum(axis=0) > 0)[0]]

idx_active_texture = np.array(
    [label_to_idx[c] for c in ACTIVE_CLASSES if CLASS_NAME_MAP.get(c) in TEXTURE_TAXA],
    dtype=np.int32
)
idx_active_event = np.array(
    [label_to_idx[c] for c in ACTIVE_CLASSES if CLASS_NAME_MAP.get(c) not in TEXTURE_TAXA],
    dtype=np.int32
)

idx_mapped_active_texture  = idx_active_texture[MAPPED_MASK[idx_active_texture]]
idx_mapped_active_event    = idx_active_event[MAPPED_MASK[idx_active_event]]
idx_unmapped_active_texture = idx_active_texture[~MAPPED_MASK[idx_active_texture]]
idx_unmapped_active_event   = idx_active_event[~MAPPED_MASK[idx_active_event]]
idx_unmapped_inactive = np.array(
    [i for i in UNMAPPED_POS if PRIMARY_LABELS[i] not in ACTIVE_CLASSES], dtype=np.int32
)

print(f'Active texture classes : {len(idx_active_texture)}')
print(f'Active event classes   : {len(idx_active_event)}')
print(f'Unmapped inactive      : {len(idx_unmapped_inactive)}')

# ── Genus proxy for unmapped amphibians ──
unmapped_df = mapping[mapping['bc_index'] == NO_LABEL_INDEX].copy()
unmapped_non_sonotype = unmapped_df[
    ~unmapped_df['primary_label'].astype(str).str.contains('son', na=False)
].copy()

proxy_map = {}
for _, row in unmapped_non_sonotype.iterrows():
    genus = str(row['scientific_name']).split()[0]
    hits  = bc_labels[
        bc_labels['scientific_name'].str.match(rf'^{re.escape(genus)}\s', na=False)
    ]
    if len(hits) > 0:
        proxy_map[str(row['primary_label'])] = hits['bc_index'].astype(int).tolist()

SELECTED_PROXY_TARGETS   = sorted([t for t in proxy_map if CLASS_NAME_MAP.get(t) == 'Amphibia'])
selected_proxy_pos       = np.array([label_to_idx[c] for c in SELECTED_PROXY_TARGETS], dtype=np.int32)
selected_proxy_pos_to_bc = {
    label_to_idx[t]: np.array(proxy_map[t], dtype=np.int32) for t in SELECTED_PROXY_TARGETS
}

idx_selected_proxy_active_texture  = np.intersect1d(selected_proxy_pos, idx_active_texture)
idx_selected_prioronly_active_texture = np.setdiff1d(idx_unmapped_active_texture, selected_proxy_pos)
idx_selected_prioronly_active_event   = np.setdiff1d(idx_unmapped_active_event, selected_proxy_pos)

print(f'Frog proxy targets : {SELECTED_PROXY_TARGETS}')

# ── Audio & Perch Inference ──
def read_soundscape_60s(path):
    y, sr = sf.read(path, dtype='float32', always_2d=False)
    if y.ndim == 2:
        y = y.mean(axis=1)
    if len(y) < CFG.FILE_SAMPLES:
        y = np.pad(y, (0, CFG.FILE_SAMPLES - len(y)))
    return y[:CFG.FILE_SAMPLES]


def infer_perch_batch(paths, verbose=True):
    paths   = [Path(p) for p in paths]
    n_files = len(paths)
    n_rows  = n_files * CFG.N_WINDOWS

    row_ids    = np.empty(n_rows, dtype=object)
    filenames  = np.empty(n_rows, dtype=object)
    sites      = np.empty(n_rows, dtype=object)
    hours      = np.empty(n_rows, dtype=np.int16)
    scores     = np.zeros((n_rows, N_CLASSES), dtype=np.float32)
    embeddings = np.zeros((n_rows, 1536),      dtype=np.float32)

    write_row = 0
    itr = tqdm(range(0, n_files, CFG.BATCH_FILES), desc='Perch', disable=not verbose)

    for start in itr:
        batch  = paths[start:start + CFG.BATCH_FILES]
        bn     = len(batch)
        x      = np.empty((bn * CFG.N_WINDOWS, CFG.WINDOW_SAMPLES), dtype=np.float32)
        bstart = write_row

        for bi, path in enumerate(batch):
            audio = read_soundscape_60s(path)
            x[bi * CFG.N_WINDOWS:(bi + 1) * CFG.N_WINDOWS] = audio.reshape(
                CFG.N_WINDOWS, CFG.WINDOW_SAMPLES
            )
            meta = parse_soundscape_filename(path.name)
            row_ids[write_row:write_row + CFG.N_WINDOWS]   = [
                f'{path.stem}_{t}' for t in range(5, 65, 5)
            ]
            filenames[write_row:write_row + CFG.N_WINDOWS] = path.name
            sites[write_row:write_row + CFG.N_WINDOWS]     = meta['site']
            hours[write_row:write_row + CFG.N_WINDOWS]     = meta['hour_utc']
            write_row += CFG.N_WINDOWS

        out    = infer_fn(inputs=tf.convert_to_tensor(x))
        logits = out['label'].numpy().astype(np.float32)
        emb    = out['embedding'].numpy().astype(np.float32)

        scores[bstart:write_row, MAPPED_POS] = logits[:write_row - bstart, MAPPED_BC_INDICES]
        embeddings[bstart:write_row]          = emb

        for pos, bc_idx_arr in selected_proxy_pos_to_bc.items():
            scores[bstart:write_row, pos] = logits[:write_row - bstart, bc_idx_arr].max(axis=1)

        del x, out, logits, emb
        gc.collect()

    meta_df = pd.DataFrame({
        'row_id': row_ids, 'filename': filenames,
        'site': sites, 'hour_utc': hours,
    })
    return meta_df, scores, embeddings

# ── Load or Compute Perch Cache ──
if CFG.CACHE_EXISTS:
    print(f'Loading Perch cache from: {CFG.CACHE_DIR}')
    meta_full       = pd.read_parquet(CFG.CACHE_DIR / 'full_perch_meta.parquet')
    arr             = np.load(CFG.CACHE_DIR / 'full_perch_arrays.npz')
    scores_full_raw = arr['scores_full_raw'].astype(np.float32)
    emb_full        = arr['emb_full'].astype(np.float32)
else:
    print('No cache found. Running Perch on fully-labeled training soundscapes...')
    full_paths = [CFG.BASE / 'train_soundscapes' / fn for fn in full_files]
    meta_full, scores_full_raw, emb_full = infer_perch_batch(full_paths)
    meta_full.to_parquet(CFG.WORK_CACHE_DIR / 'full_perch_meta.parquet', index=False)
    np.savez_compressed(
        CFG.WORK_CACHE_DIR / 'full_perch_arrays.npz',
        scores_full_raw=scores_full_raw,
        emb_full=emb_full,
    )
    print(f'Cache saved to {CFG.WORK_CACHE_DIR}')

# Align ground truth to cache row order
full_truth_aligned = (
    full_truth.set_index('row_id')
    .loc[meta_full['row_id']]
    .reset_index(drop=False)
)
Y_FULL = Y_SC[full_truth_aligned['index'].to_numpy()]

print(f'scores_full_raw : {scores_full_raw.shape}  {scores_full_raw.dtype}')
print(f'emb_full        : {emb_full.shape}  {emb_full.dtype}')
print(f'Y_FULL          : {Y_FULL.shape}')

# ── Prior Tables ──
def fit_prior_tables(prior_df, Y_prior):
    prior_df = prior_df.reset_index(drop=True)
    global_p = Y_prior.mean(axis=0).astype(np.float32)

    site_keys = sorted(prior_df['site'].dropna().astype(str).unique())
    hour_keys = sorted(prior_df['hour_utc'].dropna().astype(int).unique())

    site_to_i, site_n, site_p = {}, [], []
    for s in site_keys:
        mask = prior_df['site'].astype(str).values == s
        site_to_i[s] = len(site_n)
        site_n.append(mask.sum())
        site_p.append(Y_prior[mask].mean(axis=0))
    site_n = np.array(site_n, dtype=np.float32)
    site_p = np.stack(site_p).astype(np.float32) if site_p else np.zeros((0, Y_prior.shape[1]), np.float32)

    hour_to_i, hour_n, hour_p = {}, [], []
    for h in hour_keys:
        mask = prior_df['hour_utc'].astype(int).values == h
        hour_to_i[h] = len(hour_n)
        hour_n.append(mask.sum())
        hour_p.append(Y_prior[mask].mean(axis=0))
    hour_n = np.array(hour_n, dtype=np.float32)
    hour_p = np.stack(hour_p).astype(np.float32) if hour_p else np.zeros((0, Y_prior.shape[1]), np.float32)

    sh_to_i, sh_n_list, sh_p_list = {}, [], []
    for (s, h), idx in prior_df.groupby(['site', 'hour_utc']).groups.items():
        sh_to_i[(str(s), int(h))] = len(sh_n_list)
        idx = np.array(list(idx))
        sh_n_list.append(len(idx))
        sh_p_list.append(Y_prior[idx].mean(axis=0))
    sh_n = np.array(sh_n_list, dtype=np.float32)
    sh_p = np.stack(sh_p_list).astype(np.float32) if sh_p_list else np.zeros((0, Y_prior.shape[1]), np.float32)

    return dict(
        global_p=global_p,
        site_to_i=site_to_i, site_n=site_n, site_p=site_p,
        hour_to_i=hour_to_i, hour_n=hour_n, hour_p=hour_p,
        sh_to_i=sh_to_i,     sh_n=sh_n,     sh_p=sh_p,
    )


def prior_logits(sites, hours, tables, eps=1e-4):
    n = len(sites)
    p = np.repeat(tables['global_p'][None, :], n, axis=0).astype(np.float32, copy=True)

    si  = np.fromiter((tables['site_to_i'].get(str(s), -1) for s in sites), np.int32, n)
    hi  = np.fromiter(
        (tables['hour_to_i'].get(int(h), -1) if int(h) >= 0 else -1 for h in hours),
        np.int32, n
    )
    shi = np.fromiter(
        (tables['sh_to_i'].get((str(s), int(h)), -1) if int(h) >= 0 else -1
         for s, h in zip(sites, hours)),
        np.int32, n
    )

    valid = hi >= 0
    if valid.any():
        nh = tables['hour_n'][hi[valid]][:, None]
        p[valid] = nh / (nh + 8.0) * tables['hour_p'][hi[valid]] + (1.0 - nh / (nh + 8.0)) * p[valid]

    valid = si >= 0
    if valid.any():
        ns = tables['site_n'][si[valid]][:, None]
        p[valid] = ns / (ns + 8.0) * tables['site_p'][si[valid]] + (1.0 - ns / (ns + 8.0)) * p[valid]

    valid = shi >= 0
    if valid.any():
        nsh = tables['sh_n'][shi[valid]][:, None]
        p[valid] = nsh / (nsh + 4.0) * tables['sh_p'][shi[valid]] + (1.0 - nsh / (nsh + 4.0)) * p[valid]

    np.clip(p, eps, 1.0 - eps, out=p)
    return (np.log(p) - np.log1p(-p)).astype(np.float32)


def smooth_cols(scores, cols, alpha=0.35):
    """Temporal smoothing: blend each window with its neighbours within the same file."""
    if alpha <= 0 or len(cols) == 0:
        return scores.copy()
    s    = scores.copy()
    view = s.reshape(-1, CFG.N_WINDOWS, s.shape[1])
    x    = view[:, :, cols]
    prev = np.concatenate([x[:, :1, :], x[:, :-1, :]], axis=1)
    nxt  = np.concatenate([x[:, 1:, :], x[:, -1:, :]], axis=1)
    view[:, :, cols] = (1.0 - alpha) * x + 0.5 * alpha * (prev + nxt)
    return s


def fuse_scores(base, sites, hours, tables):
    scores = base.copy()
    prior  = prior_logits(sites, hours, tables)

    if len(idx_mapped_active_event):
        scores[:, idx_mapped_active_event] += CFG.LAMBDA_EVENT * prior[:, idx_mapped_active_event]
    if len(idx_mapped_active_texture):
        scores[:, idx_mapped_active_texture] += CFG.LAMBDA_TEXTURE * prior[:, idx_mapped_active_texture]
    if len(idx_selected_proxy_active_texture):
        scores[:, idx_selected_proxy_active_texture] += (
            CFG.LAMBDA_PROXY_TEXTURE * prior[:, idx_selected_proxy_active_texture]
        )
    if len(idx_selected_prioronly_active_event):
        scores[:, idx_selected_prioronly_active_event] = (
            CFG.LAMBDA_EVENT * prior[:, idx_selected_prioronly_active_event]
        )
    if len(idx_selected_prioronly_active_texture):
        scores[:, idx_selected_prioronly_active_texture] = (
            CFG.LAMBDA_TEXTURE * prior[:, idx_selected_prioronly_active_texture]
        )
    if len(idx_unmapped_inactive):
        scores[:, idx_unmapped_inactive] = -8.0

    scores = smooth_cols(scores, idx_active_texture, alpha=CFG.SMOOTH_TEXTURE_ALPHA)
    return scores.astype(np.float32), prior

# ── OOF Evaluation ──
def macro_auc(y_true, y_score):
    keep = y_true.sum(axis=0) > 0
    return roc_auc_score(y_true[:, keep], y_score[:, keep], average='macro')


gkf    = GroupKFold(n_splits=5)
groups = meta_full['site'].to_numpy()

oof_base  = np.zeros_like(scores_full_raw, dtype=np.float32)
oof_prior = np.zeros_like(scores_full_raw, dtype=np.float32)

for _, va_idx in tqdm(list(gkf.split(scores_full_raw, groups=groups)), desc='OOF folds'):
    va_idx    = np.sort(va_idx)
    val_sites = set(meta_full.iloc[va_idx]['site'].tolist())
    prior_m   = ~sc_clean['site'].isin(val_sites).values
    tables    = fit_prior_tables(
        sc_clean.loc[prior_m].reset_index(drop=True), Y_SC[prior_m]
    )
    oof_base[va_idx], oof_prior[va_idx] = fuse_scores(
        scores_full_raw[va_idx],
        meta_full.iloc[va_idx]['site'].to_numpy(),
        meta_full.iloc[va_idx]['hour_utc'].to_numpy(),
        tables,
    )

print(f'OOF baseline AUC (prior fusion only): {macro_auc(Y_FULL, oof_base):.6f}')

# ── Embedding Features ──
def seq_features_1d(v):
    x    = v.reshape(-1, CFG.N_WINDOWS)
    prev = np.concatenate([x[:, :1], x[:, :-1]], axis=1).reshape(-1)
    nxt  = np.concatenate([x[:, 1:], x[:, -1:]], axis=1).reshape(-1)
    return prev, nxt, np.repeat(x.mean(1), CFG.N_WINDOWS), np.repeat(x.max(1), CFG.N_WINDOWS)


def build_class_features(Z, raw_col, prior_col, base_col):
    p, n, m, mx = seq_features_1d(base_col)
    return np.concatenate(
        [Z, raw_col[:, None], prior_col[:, None], base_col[:, None],
         p[:, None], n[:, None], m[:, None], mx[:, None]],
        axis=1
    ).astype(np.float32)

# ── PCA on Embeddings ──
emb_scaler = StandardScaler()
emb_scaled = emb_scaler.fit_transform(emb_full)

n_comp = min(CFG.PROBE_PCA_DIM, emb_scaled.shape[0] - 1, emb_scaled.shape[1])
emb_pca = PCA(n_components=n_comp)
Z_FULL  = emb_pca.fit_transform(emb_scaled).astype(np.float32)

print(f'PCA components  : {n_comp}')
print(f'Explained var   : {emb_pca.explained_variance_ratio_.sum():.4f}')

# ── Train Probes ──
print(f'Using frozen probe params: pca_dim={CFG.PROBE_PCA_DIM} '
      f'min_pos={CFG.PROBE_MIN_POS} C={CFG.PROBE_C} alpha={CFG.PROBE_ALPHA}')

pos_counts  = Y_FULL.sum(axis=0)
probe_idx   = np.where(pos_counts >= CFG.PROBE_MIN_POS)[0].astype(np.int32)
probe_models = {}

for cls_idx in tqdm(probe_idx, desc='Training probes'):
    y = Y_FULL[:, cls_idx]
    if y.sum() == 0 or y.sum() == len(y):
        continue
    X = build_class_features(
        Z_FULL,
        raw_col=scores_full_raw[:, cls_idx],
        prior_col=oof_prior[:, cls_idx],
        base_col=oof_base[:, cls_idx],
    )
    clf = LogisticRegression(
        C=CFG.PROBE_C, max_iter=400, solver='liblinear', class_weight='balanced'
    )
    clf.fit(X, y)
    probe_models[cls_idx] = clf

print(f'Probe models trained : {len(probe_models)} / {N_CLASSES} classes')

# ── Test Inference ──
final_tables = fit_prior_tables(sc_clean.reset_index(drop=True), Y_SC)

test_paths = sorted((CFG.BASE / 'test_soundscapes').glob('*.ogg'))
if len(test_paths) == 0:
    print(f'No test soundscapes found. Dry-run on {CFG.DRYRUN_N_FILES} train files.')
    test_paths = sorted((CFG.BASE / 'train_soundscapes').glob('*.ogg'))[:CFG.DRYRUN_N_FILES]
else:
    print(f'Test files : {len(test_paths)}')

meta_test, scores_test_raw, emb_test = infer_perch_batch(test_paths)

# Prior fusion
test_base, test_prior = fuse_scores(
    scores_test_raw,
    meta_test['site'].to_numpy(),
    meta_test['hour_utc'].to_numpy(),
    final_tables,
)

# PCA projection
Z_TEST = emb_pca.transform(emb_scaler.transform(emb_test)).astype(np.float32)

# Apply probes with per-class alpha
final_scores = test_base.copy()
for cls_idx, clf in tqdm(probe_models.items(), desc='Applying probes'):
    X = build_class_features(
        Z_TEST,
        raw_col=scores_test_raw[:, cls_idx],
        prior_col=test_prior[:, cls_idx],
        base_col=test_base[:, cls_idx],
    )
    pred = clf.decision_function(X).astype(np.float32)
    n_pos = int(Y_FULL[:, cls_idx].sum())
    alpha = 0.25 + 0.25 * min(1.0, (n_pos - CFG.PROBE_MIN_POS) / 40.0)
    final_scores[:, cls_idx] = (
        (1.0 - alpha) * test_base[:, cls_idx]
        + alpha * pred
    )

print(f'final_scores : {final_scores.shape}')
print(f'Score range  : {final_scores.min():.3f} to {final_scores.max():.3f}')

# ── Build Submission ──
from scipy.ndimage import convolve1d

def gauss_smooth_final(scores, weights=np.array([0.1, 0.2, 0.4, 0.2, 0.1])):
    smoothed = scores.reshape(-1, CFG.N_WINDOWS, scores.shape[1]).copy()
    for i in range(smoothed.shape[0]):
        smoothed[i] = convolve1d(smoothed[i], weights, axis=0, mode='nearest')
    return smoothed.reshape(-1, scores.shape[1])

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

final_scores_smoothed = gauss_smooth_final(final_scores)
submission = pd.DataFrame(sigmoid(final_scores_smoothed), columns=PRIMARY_LABELS)
submission.insert(0, 'row_id', meta_test['row_id'].values)
submission[PRIMARY_LABELS] = submission[PRIMARY_LABELS].astype(np.float32)

assert len(submission) == len(test_paths) * CFG.N_WINDOWS, 'Row count mismatch'
assert submission.columns.tolist() == ['row_id'] + PRIMARY_LABELS, 'Column order mismatch'
assert not submission.isna().any().any(), 'NaNs detected in submission'
assert (submission[PRIMARY_LABELS] >= 0).all().all(), 'Negative probabilities'
assert (submission[PRIMARY_LABELS] <= 1).all().all(), 'Probabilities > 1'

submission.to_csv('/kaggle/working/submission.csv', index=False)
print('submission.csv saved')
print(f'Shape : {submission.shape}')
print(submission.iloc[:3, :8])
