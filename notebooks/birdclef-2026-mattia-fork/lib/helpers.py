"""Validation + post-processing helpers."""
# ── Cell 7: Metric helpers ─────────────────────────────────────────────
def macro_auc(y_true, y_score):
    """
    Exact replica of the competition metric:
    macro-averaged ROC-AUC, skipping classes with no positive labels.
    This is the ONLY number you should track locally.
    """
    keep = y_true.sum(axis=0) > 0
    return roc_auc_score(y_true[:, keep], y_score[:, keep], average="macro")
 
 
def honest_oof_auc(scores, Y, meta_df, n_splits=5, label="scores"):
    """
    GroupKFold by filename — files never split across folds.
    This is the only correct way to estimate LB performance locally.
    Leaking a file across train/val inflates AUC by ~0.01–0.03.
    """
    groups = meta_df["filename"].to_numpy()
    gkf    = GroupKFold(n_splits=n_splits)
    oof    = np.zeros_like(scores, dtype=np.float32)
 
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(scores, groups=groups), 1):
        oof[va_idx] = scores[va_idx]
 
    auc = macro_auc(Y, oof)
    print(f"[{label}] honest OOF macro-AUC: {auc:.6f}")
    return auc, oof
# ── Cell 7b: Temporal smoothing helper ─────────────────────────────────
def smooth_predictions(probs, n_windows=12, alpha=0.3):
    """
    For each file's 12 windows, blend each window with its neighbors.
    
    new[t] = (1 - alpha) * old[t] + 0.5*alpha * (old[t-1] + old[t+1])
    
    alpha=0: no smoothing (your current baseline)
    alpha=0.3: moderate smoothing (good starting point)
    
    Shape: (n_files * 12, n_classes) → same shape output
    """
    N, C = probs.shape
    assert N % n_windows == 0, f"Expected multiple of {n_windows}, got {N}"
    
    # Reshape to (n_files, 12, 234) so we can work file-by-file
    view = probs.reshape(-1, n_windows, C).copy()
    
    # Shift left and right (with edge padding = repeat boundary)
    prev_w = np.concatenate([view[:, :1, :],  view[:, :-1, :]], axis=1)  # t-1
    next_w = np.concatenate([view[:, 1:,  :], view[:, -1:, :]], axis=1)  # t+1
    
    smoothed = (1 - alpha) * view + 0.5 * alpha * (prev_w + next_w)
    
    return smoothed.reshape(N, C)


print("✅ Temporal smoothing helper defined")
# ── Cell 7c: Prior table builder ───────────────────────────────────────
def build_prior_tables(sc_df, Y_labels):
    """
    Build site-level and hour-level species frequency tables.
    
    These answer: "How often is species X observed at site S at hour H?"
    
    We use these as a soft prior: add them to raw Perch logits.
    """
    sc_df = sc_df.reset_index(drop=True)
    global_p = Y_labels.mean(axis=0).astype(np.float32)  # overall frequency
    
    # ── Site-level frequencies ──────────────────────────────────────────
    site_keys = sorted(sc_df["site"].dropna().astype(str).unique())
    site_to_i = {k: i for i, k in enumerate(site_keys)}
    site_p    = np.zeros((len(site_keys), Y_labels.shape[1]), dtype=np.float32)
    site_n    = np.zeros(len(site_keys), dtype=np.float32)
    
    for s in site_keys:
        i     = site_to_i[s]
        mask  = sc_df["site"].astype(str).values == s
        site_n[i] = mask.sum()
        site_p[i] = Y_labels[mask].mean(axis=0)
    
    # ── Hour-level frequencies ──────────────────────────────────────────
    hour_keys = sorted(sc_df["hour_utc"].dropna().astype(int).unique())
    hour_to_i = {h: i for i, h in enumerate(hour_keys)}
    hour_p    = np.zeros((len(hour_keys), Y_labels.shape[1]), dtype=np.float32)
    hour_n    = np.zeros(len(hour_keys), dtype=np.float32)
    
    for h in hour_keys:
        i     = hour_to_i[h]
        mask  = sc_df["hour_utc"].astype(int).values == h
        hour_n[i] = mask.sum()
        hour_p[i] = Y_labels[mask].mean(axis=0)
    
    return {
        "global_p": global_p,
        "site_to_i": site_to_i, "site_p": site_p, "site_n": site_n,
        "hour_to_i": hour_to_i, "hour_p": hour_p, "hour_n": hour_n,
    }


def apply_prior(scores, sites, hours, tables, lambda_prior=0.4):
    """
    Add a scaled prior logit to the raw Perch scores.
    
    lambda_prior=0: no effect (your baseline)
    lambda_prior=0.4: moderate influence from location/time
    
    The prior is converted to a logit (log-odds) before adding.
    This is mathematically correct — you add logits, not probabilities.
    """
    eps = 1e-4
    n   = len(scores)
    out = scores.copy()
    
    # Start from global average
    p = np.tile(tables["global_p"], (n, 1))  # (n, 234)
    
    # Override with hour-level estimate (if enough data)
    for i, h in enumerate(hours):
        h = int(h)
        if h in tables["hour_to_i"]:
            j   = tables["hour_to_i"][h]
            nh  = tables["hour_n"][j]
            w   = nh / (nh + 8.0)   # shrink toward global if little data
            p[i] = w * tables["hour_p"][j] + (1 - w) * tables["global_p"]
    
    # Override with site-level estimate (if enough data)
    for i, s in enumerate(sites):
        s = str(s)
        if s in tables["site_to_i"]:
            j   = tables["site_to_i"][s]
            ns  = tables["site_n"][j]
            w   = ns / (ns + 8.0)   # same shrinkage logic
            p[i] = w * tables["site_p"][j] + (1 - w) * p[i]
    
    # Convert prior probability to logit and add
    p      = np.clip(p, eps, 1 - eps)
    logit_prior = np.log(p) - np.log1p(-p)
    out   += lambda_prior * logit_prior
    
    return out.astype(np.float32)


print("✅ Prior table functions defined")
# ── Cell 7d: File-level confidence scaling ─────────────────────────────
def file_confidence_scale(probs, n_windows=12, top_k=2, power=0.4):
    """
    Scale each window's predictions by how confident the file is overall.
    
    Steps:
    1. For each file, find the top-k highest scores across all 12 windows
    2. Compute their mean → "file confidence"
    3. Multiply every window's scores by (file_confidence ** power)
    
    power=0: no effect (baseline)
    power=0.4: moderate suppression of uncertain files
    
    Why top-k and not max?
    Max is noisy (one lucky spike). Top-2 mean is more robust.
    """
    N, C = probs.shape
    assert N % n_windows == 0
    
    view      = probs.reshape(-1, n_windows, C)       # (n_files, 12, 234)
    sorted_v  = np.sort(view, axis=1)                 # sort across time
    top_k_mean = sorted_v[:, -top_k:, :].mean(axis=1, keepdims=True)  # (n_files, 1, 234)
    
    scale  = np.power(top_k_mean, power)              # (n_files, 1, 234)
    scaled = view * scale                             # broadcast across 12 windows
    
    return scaled.reshape(N, C)


print("✅ File-level confidence scaling defined")
# ── Cell 7e: Per-taxon temperature scaling ─────────────────────────────
# Build lookup: which species class are they?
CLASS_NAME_MAP = taxonomy.set_index("primary_label")["class_name"].to_dict()
TEXTURE_TAXA   = {"Amphibia", "Insecta"}   # continuous callers

# Build per-class temperature vector
temperatures = np.ones(N_CLASSES, dtype=np.float32)
for ci, label in enumerate(PRIMARY_LABELS):
    cls = CLASS_NAME_MAP.get(label, "Aves")
    if cls in TEXTURE_TAXA:
        temperatures[ci] = 0.95   # frogs/insects: slightly sharper
    else:
        temperatures[ci] = 1.10   # birds: slightly softer

n_texture = (temperatures < 1.0).sum()
n_event   = (temperatures > 1.0).sum()
print(f"✅ Temperatures: {n_event} event species (T=1.10), {n_texture} texture species (T=0.95)")