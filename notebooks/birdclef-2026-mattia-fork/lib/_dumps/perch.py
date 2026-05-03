"""Perch v2 backbone loader + ONNX inference cache."""
# ── Cell 4: Load Perch model (ONNX preferred) ─────────────────────────
birdclassifier = tf.saved_model.load(str(MODEL_DIR))
infer_fn       = birdclassifier.signatures["serving_default"]

# ONNX session (150x faster)
ONNX_PERCH_PATH = Path("/kaggle/input/datasets/rishikeshjani/perch-onnx-for-birdclef-2026/perch_v2.onnx")
USE_ONNX = _ONNX_AVAILABLE and ONNX_PERCH_PATH.exists()

if USE_ONNX:
    _so = ort.SessionOptions()
    _so.intra_op_num_threads = 4
    ONNX_SESSION    = ort.InferenceSession(str(ONNX_PERCH_PATH), sess_options=_so,
                                            providers=["CPUExecutionProvider"])
    ONNX_INPUT_NAME = ONNX_SESSION.get_inputs()[0].name
    ONNX_OUT_MAP    = {o.name: i for i, o in enumerate(ONNX_SESSION.get_outputs())}
    print("Using ONNX Perch (150x faster)")
else:
    print("Using TF SavedModel Perch")

bc_labels = (pd.read_csv(MODEL_DIR / "assets" / "labels.csv")
             .reset_index()
             .rename(columns={"index": "bc_index", "inat2024_fsd50k": "scientific_name"}))
NO_LABEL = len(bc_labels)

mapping = (taxonomy
           .merge(bc_labels.rename(columns={"scientific_name": "scientific_name"}),
                  on="scientific_name", how="left"))
mapping["bc_index"] = mapping["bc_index"].fillna(NO_LABEL).astype(int)
lbl2bc = mapping.set_index("primary_label")["bc_index"]

BC_INDICES    = np.array([int(lbl2bc.loc[c]) for c in PRIMARY_LABELS], dtype=np.int32)
MAPPED_MASK   = BC_INDICES != NO_LABEL
MAPPED_POS    = np.where(MAPPED_MASK)[0].astype(np.int32)
MAPPED_BC_IDX = BC_INDICES[MAPPED_MASK].astype(np.int32)

print(f"Mapped: {MAPPED_MASK.sum()} / {N_CLASSES} species have a Perch logit")
# --- 4b: Genus proxy logits ---
# ── Cell 4b: Genus proxy logits for unmapped species ──────────────────
import re as _re

# Find which species have no direct Perch mapping
UNMAPPED_POS  = np.where(~MAPPED_MASK)[0].astype(np.int32)

CLASS_NAME_MAP = taxonomy.set_index("primary_label")["class_name"].to_dict()
TEXTURE_TAXA   = {"Amphibia", "Insecta"}

# For each unmapped species, find genus-level matches in Perch vocab
proxy_map = {}   # label_idx -> list of bc_indices

unmapped_df = (taxonomy[taxonomy["primary_label"]
               .isin([PRIMARY_LABELS[i] for i in UNMAPPED_POS])]
               .copy())

for _, row in unmapped_df.iterrows():
    target = row["primary_label"]
    sci    = str(row["scientific_name"])
    genus  = sci.split()[0]
    
    # Find all Perch labels from the same genus
    hits = bc_labels[
        bc_labels["scientific_name"]
        .astype(str)
        .str.match(rf"^{_re.escape(genus)}\s", na=False)
    ]
    
    if len(hits) > 0:
        proxy_map[label_to_idx[target]] = hits["bc_index"].astype(int).tolist()

# Only use proxies for biologically meaningful taxa
PROXY_TAXA = {"Amphibia", "Insecta", "Aves"}
proxy_map  = {
    idx: bc_idxs
    for idx, bc_idxs in proxy_map.items()
    if CLASS_NAME_MAP.get(PRIMARY_LABELS[idx]) in PROXY_TAXA
}

print(f"Unmapped species total:        {len(UNMAPPED_POS)}")
print(f"Species with genus proxy:      {len(proxy_map)}")
print(f"Species still without signal:  {len(UNMAPPED_POS) - len(proxy_map)}")
print("\nProxy targets:")
for idx, bc_idxs in list(proxy_map.items())[:8]:
    label = PRIMARY_LABELS[idx]
    cls   = CLASS_NAME_MAP.get(label, "?")
    print(f"  {label:12s} ({cls:10s}) ← {len(bc_idxs)} Perch genus matches")
# --- 5: Perch inference engine ---
# ── Cell 5: Perch inference engine (ONNX + multithreaded I/O) ─────────
import concurrent.futures

def read_60s(path):
    y, sr = sf.read(path, dtype="float32", always_2d=False)
    if y.ndim == 2: y = y.mean(axis=1)
    if len(y) < FILE_SAMPLES: y = np.pad(y, (0, FILE_SAMPLES - len(y)))
    else:                      y = y[:FILE_SAMPLES]
    return y

def run_perch(paths, batch_files=16, verbose=True):
    paths  = [Path(p) for p in paths]
    n_rows = len(paths) * N_WINDOWS

    row_ids   = np.empty(n_rows, dtype=object)
    filenames = np.empty(n_rows, dtype=object)
    sites     = np.empty(n_rows, dtype=object)
    hours     = np.zeros(n_rows, dtype=np.int16)
    scores    = np.zeros((n_rows, N_CLASSES), dtype=np.float32)
    embs      = np.zeros((n_rows, 1536),      dtype=np.float32)

    wr  = 0
    itr = tqdm(range(0, len(paths), batch_files), desc="Perch") if verbose else range(0, len(paths), batch_files)

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as io_executor:
        # Prefetch first batch
        next_paths   = paths[0:batch_files]
        future_audio = [io_executor.submit(read_60s, p) for p in next_paths]

        for start in itr:
            batch_paths  = next_paths
            batch_n      = len(batch_paths)
            batch_audio  = [f.result() for f in future_audio]

            # Prefetch next batch immediately
            next_start = start + batch_files
            if next_start < len(paths):
                next_paths   = paths[next_start:next_start + batch_files]
                future_audio = [io_executor.submit(read_60s, p) for p in next_paths]

            x  = np.empty((batch_n * N_WINDOWS, WINDOW_SAMPLES), dtype=np.float32)
            br = wr

            for bi, path in enumerate(batch_paths):
                y    = batch_audio[bi]
                meta = parse_fname(path.name)
                stem = path.stem
                x[bi * N_WINDOWS:(bi + 1) * N_WINDOWS] = y.reshape(N_WINDOWS, WINDOW_SAMPLES)
                row_ids  [wr:wr + N_WINDOWS] = [f"{stem}_{t}" for t in range(5, 65, 5)]
                filenames[wr:wr + N_WINDOWS] = path.name
                sites    [wr:wr + N_WINDOWS] = meta["site"]
                hours    [wr:wr + N_WINDOWS] = meta["hour_utc"]
                wr += N_WINDOWS

            # ── ONNX or TF inference ───────────────────────────────────
            if USE_ONNX:
                outs   = ONNX_SESSION.run(None, {ONNX_INPUT_NAME: x})
                logits = outs[ONNX_OUT_MAP["label"]].astype(np.float32)
                emb    = outs[ONNX_OUT_MAP["embedding"]].astype(np.float32)
            else:
                out    = infer_fn(inputs=tf.convert_to_tensor(x))
                logits = out["label"].numpy().astype(np.float32)
                emb    = out["embedding"].numpy().astype(np.float32)

            scores[br:wr, MAPPED_POS] = logits[:, MAPPED_BC_IDX]
            embs  [br:wr]             = emb

            for pos_idx, bc_idxs in proxy_map.items():
                bc_arr = np.array(bc_idxs, dtype=np.int32)
                scores[br:wr, pos_idx] = logits[:, bc_arr].max(axis=1)

            del x, logits, emb, batch_audio
            gc.collect()

    meta_df = pd.DataFrame({"row_id": row_ids, "filename": filenames,
                             "site": sites, "hour_utc": hours})
    return meta_df, scores, embs

print("✅ Perch inference engine (ONNX + multithreaded I/O) defined")
# --- 6: Build/load Perch training cache ---
# ── Cell 6: Build-or-load Perch training cache ────────────────────────
print(f"USE_ONNX = {USE_ONNX}  "
      f"(cache will be built with {'ONNX' if USE_ONNX else 'TF SavedModel'})")

# Add any external cache locations here if you want to reuse pre-built data
EXTERNAL_CACHE_DIRS = [
    Path("/kaggle/input/notebooks/vyankteshdwivedi/notebook1b25083f0d"),
    Path("/kaggle/input/datasets/jaejohn/perch-meta"),
]

CACHE_META_LOCAL = WORK_DIR / "perch_meta.parquet"
CACHE_NPZ_LOCAL  = WORK_DIR / "perch_arrays.npz"


def _find_external_cache():
    for d in EXTERNAL_CACHE_DIRS:
        meta = d / "perch_meta.parquet"
        npz  = d / "perch_arrays.npz"
        if meta.exists() and npz.exists():
            return meta, npz
    return None, None


SCORE_KEYS = ["scores", "sc", "logits", "perch_scores", "preds", "arr_0"]
EMB_KEYS   = ["embs", "emb", "embeddings", "features", "perch_embs", "arr_1"]


def _pick_array(arr, candidates, shape_hint_cols):
    for k in candidates:
        if k in arr.files:
            return arr[k], k

    for k in arr.files:
        v = arr[k]
        if v.ndim == 2 and v.shape[1] == shape_hint_cols:
            return v, k

    raise KeyError(f"None of {candidates} found in npz. Available keys: {arr.files}")


def _build_cache():
    print(f"Building Perch cache from {len(full_files)} training files…")

    train_paths = [BASE / "train_soundscapes" / fn for fn in full_files]
    train_paths = [p for p in train_paths if p.exists()]

    t0 = time.time()

    meta_built, sc_built, emb_built = run_perch(
        train_paths,
        batch_files=CFG["batch_files"],
        verbose=True
    )

    print(f"  Perch pass done in {time.time()-t0:.1f}s  "
          f"scores={sc_built.shape} embs={emb_built.shape}")

    meta_built.to_parquet(CACHE_META_LOCAL)

    np.savez(
        CACHE_NPZ_LOCAL,
        scores=sc_built.astype(np.float32),
        embs=emb_built.astype(np.float32),
        primary_labels=np.array(PRIMARY_LABELS)
    )

    print(f"  Cache saved to {WORK_DIR}")

    return CACHE_META_LOCAL, CACHE_NPZ_LOCAL


ext_meta, ext_npz = _find_external_cache()

if ext_meta is not None:
    CACHE_META, CACHE_NPZ = ext_meta, ext_npz
    print(f"Using external cache: {CACHE_META.parent}")

elif CACHE_META_LOCAL.exists() and CACHE_NPZ_LOCAL.exists():
    CACHE_META, CACHE_NPZ = CACHE_META_LOCAL, CACHE_NPZ_LOCAL
    print(f"Using local cache: {WORK_DIR}")

else:
    print("No cache found — building from scratch (~1.5 min)")
    CACHE_META, CACHE_NPZ = _build_cache()


print("Loading Perch cache from:", CACHE_META.parent)

meta_tr = pd.read_parquet(CACHE_META)
_arr    = np.load(CACHE_NPZ)


sc_tr_raw,  sk = _pick_array(_arr, SCORE_KEYS, N_CLASSES)
emb_tr_raw, ek = _pick_array(_arr, EMB_KEYS,   1536)

print(f"  scores ← '{sk}'  shape={sc_tr_raw.shape}")
print(f"  embs   ← '{ek}'  shape={emb_tr_raw.shape}")


sc_tr  = sc_tr_raw.astype(np.float32)
emb_tr = emb_tr_raw.astype(np.float32)


if "primary_labels" in _arr.files:
    if _arr["primary_labels"].tolist() != PRIMARY_LABELS:
        print("  WARNING: cached primary_labels differ — scores columns may not align!")
    else:
        print("  primary_labels schema OK")


if "row_id" not in meta_tr.columns:
    print("  row_id missing — reconstructing")

    if "end_sec" in meta_tr.columns:
        end_sec = meta_tr["end_sec"].astype(int)

    elif "window_idx" in meta_tr.columns:
        end_sec = (meta_tr["window_idx"].astype(int) + 1) * 5

    else:
        end_sec = np.tile(np.arange(5, 65, 5), len(meta_tr) // N_WINDOWS)

    meta_tr["row_id"] = (
        meta_tr["filename"].str.replace(".ogg", "", regex=False)
        + "_" + end_sec.astype(str)
    )


row_id_to_index = full_rows.set_index("row_id")["index"]

missing_rows = set(meta_tr["row_id"]) - set(row_id_to_index.index)

if missing_rows:
    raise RuntimeError(
        f"Cache has {len(missing_rows)} row_ids not in labeled set. "
        f"Delete {CACHE_META_LOCAL} and {CACHE_NPZ_LOCAL} to rebuild."
    )


Y_FULL_aligned = Y_SC[
    row_id_to_index.loc[meta_tr["row_id"]].to_numpy()
]

print(f"sc_tr: {sc_tr.shape}  emb_tr: {emb_tr.shape}  Y_FULL_aligned: {Y_FULL_aligned.shape}")