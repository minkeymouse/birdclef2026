"""
exp33 — probe architecture sweep (CPU).

Gap: exp22/24/28 all used LogReg probes. MLP / temporal stacking / Platt calibration never
systematically tested under Val-A with proper folds.

Tests (per class, Val-A 5-fold file-stratified):
  P1  LogReg C=0.25          — exp28 baseline
  P2  MLP [32→16→1] dropout 0.3  — small non-linear
  P3  MLP [64→32→1] dropout 0.5  — larger
  P4  Temporal stack: concat(prev, curr, next) PCA32 → LogReg — 3x input
  P5  P1 + Platt calibration on fold
  P6  P4 + Platt calibration
"""
from __future__ import annotations
import json, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression

warnings.filterwarnings("ignore")

ROOT = Path("/data/birdclef2026")
CACHE = ROOT / "experiments/exp21_outputs/perch_cache"
DATA = ROOT / "data/birdclef-2026"
OUT = ROOT / "experiments/exp33_outputs"
OUT.mkdir(parents=True, exist_ok=True)


def load_all():
    sc_raw = pd.read_csv(DATA / "train_soundscapes_labels.csv").drop_duplicates().reset_index(drop=True)
    sample_sub = pd.read_csv(DATA / "sample_submission.csv")
    primary = sample_sub.columns[1:].tolist()
    lab2idx = {c: i for i, c in enumerate(primary)}

    def parse(x):
        if pd.isna(x): return []
        return [t.strip() for t in str(x).split(";") if t.strip()]

    sc = (sc_raw.groupby(["filename", "start", "end"])["primary_label"]
          .apply(lambda s: sorted({l for x in s for l in parse(x)})).reset_index(name="lbls"))
    sc["end_sec"] = pd.to_timedelta(sc["end"]).dt.total_seconds().astype(int)
    sc["row_id"] = sc["filename"].str.replace(".ogg", "", regex=False) + "_" + sc["end_sec"].astype(str)

    meta = pd.read_parquet(CACHE / "full_perch_meta.parquet")
    Y = np.zeros((len(meta), len(primary)), dtype=np.uint8)
    by_rowid = sc.set_index("row_id")
    for i, rid in enumerate(meta["row_id"]):
        if rid in by_rowid.index:
            for l in by_rowid.loc[rid, "lbls"]:
                if l in lab2idx:
                    Y[i, lab2idx[l]] = 1

    arr = np.load(CACHE / "full_perch_arrays.npz")
    return meta, Y, primary, arr["scores"], arr["emb"]


def val_a_folds(meta):
    files = meta.drop_duplicates("filename").reset_index(drop=True)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    f2f = {}
    for fold, (_, vi) in enumerate(skf.split(files["filename"], files["site"])):
        for f in files.iloc[vi]["filename"].values:
            f2f[f] = fold
    return meta["filename"].map(f2f).values.astype(int)


def macro_auc(y_true, y_score):
    keep = y_true.sum(0) > 0
    return float(roc_auc_score(y_true[:, keep], y_score[:, keep], average="macro"))


def temporal_stack(X, meta):
    """Stack (prev, curr, next) PCA emb along axis=1. Pad edges with self."""
    out = np.zeros((len(X), X.shape[1] * 3), dtype=X.dtype)
    for fn, g in meta.groupby("filename", sort=False):
        idx = g.index.values
        x = X[idx]  # (12, D)
        prev = np.vstack([x[:1], x[:-1]])
        next_ = np.vstack([x[1:], x[-1:]])
        stacked = np.concatenate([prev, x, next_], axis=1)
        out[idx] = stacked
    return out


def platt_calibrate(s_train, y_train, s_val):
    """Fit LogReg on (s_train → y_train), apply to s_val. Returns calibrated decision values."""
    if y_train.sum() < 2 or (1 - y_train).sum() < 2:
        return s_val
    try:
        lr = LogisticRegression(max_iter=200, C=1.0)
        lr.fit(s_train.reshape(-1, 1), y_train)
        return lr.decision_function(s_val.reshape(-1, 1))
    except Exception:
        return s_val


MIN_POS = 8


def train_evaluate(X, Y, folds, make_model, calibrate=False):
    """Per-class 5-fold training, returns full OOF predictions and macro AUC."""
    preds = np.zeros_like(Y, dtype=np.float32)
    n_classes = Y.shape[1]
    n_trained = 0
    for c in range(n_classes):
        if Y[:, c].sum() < MIN_POS:
            continue
        for f in range(5):
            ti = folds != f
            vi = ~ti
            if Y[ti, c].sum() < 3:
                continue
            try:
                clf = make_model()
                clf.fit(X[ti], Y[ti, c])
                if hasattr(clf, "decision_function"):
                    s = clf.decision_function(X[vi])
                    s_tr = clf.decision_function(X[ti])
                else:
                    s = clf.predict_proba(X[vi])[:, 1]
                    s_tr = clf.predict_proba(X[ti])[:, 1]
                if calibrate:
                    s = platt_calibrate(s_tr, Y[ti, c], s)
                preds[vi, c] = s
            except Exception:
                pass
        n_trained += 1
    return macro_auc(Y, preds), n_trained, preds


def main():
    t0 = time.time()
    meta, Y, primary, scores_raw, emb = load_all()
    folds = val_a_folds(meta)

    pca = PCA(n_components=32, random_state=42).fit(emb)
    X32 = pca.transform(emb).astype(np.float32)
    X_stack = temporal_stack(X32, meta)  # (708, 96)
    print(f"Base X32 {X32.shape}, temporal stack {X_stack.shape}")
    scaler = StandardScaler().fit(X_stack)
    X_stack_sc = scaler.transform(X_stack).astype(np.float32)

    experiments = [
        ("P1_LogReg_C0.25", X32, lambda: LogisticRegression(max_iter=500, C=0.25), False),
        ("P2_MLP_32_16", X32, lambda: MLPClassifier(hidden_layer_sizes=(32, 16),
                                                      activation="relu", solver="adam",
                                                      alpha=0.001, max_iter=200,
                                                      early_stopping=True, validation_fraction=0.15,
                                                      random_state=42), False),
        ("P3_MLP_64_32", X32, lambda: MLPClassifier(hidden_layer_sizes=(64, 32),
                                                      activation="relu", solver="adam",
                                                      alpha=0.01, max_iter=200,
                                                      early_stopping=True, validation_fraction=0.15,
                                                      random_state=42), False),
        ("P4_Temporal_LogReg", X_stack, lambda: LogisticRegression(max_iter=500, C=0.25), False),
        ("P5_LogReg_Platt", X32, lambda: LogisticRegression(max_iter=500, C=0.25), True),
        ("P6_Temporal_Platt", X_stack, lambda: LogisticRegression(max_iter=500, C=0.25), True),
    ]

    results = []
    for name, X, mk, cal in experiments:
        t_e = time.time()
        auc, n, _ = train_evaluate(X, Y, folds, mk, calibrate=cal)
        dt = time.time() - t_e
        print(f"  {name:25s}  Val-A {auc:.4f}  ({n} cls, {dt:.1f}s)")
        results.append({"name": name, "val_a": auc, "n_trained": n, "time_s": dt})

    # Also test: MLP on temporal stack (bigger input)
    for name, X, mk, cal in [
        ("P7_MLP_temporal", X_stack_sc, lambda: MLPClassifier(hidden_layer_sizes=(64, 32),
                                                                alpha=0.01, max_iter=200,
                                                                early_stopping=True, validation_fraction=0.15,
                                                                random_state=42), False),
    ]:
        t_e = time.time()
        auc, n, _ = train_evaluate(X, Y, folds, mk, calibrate=cal)
        dt = time.time() - t_e
        print(f"  {name:25s}  Val-A {auc:.4f}  ({n} cls, {dt:.1f}s)")
        results.append({"name": name, "val_a": auc, "n_trained": n, "time_s": dt})

    (OUT / "results.json").write_text(json.dumps({
        "elapsed_s": time.time() - t0,
        "reference_exp28_LB910freeze": 0.8891,
        "results": results,
    }, indent=2))
    print(f"\nDone in {(time.time()-t0)/60:.1f} min. Best: {max(results, key=lambda r: r['val_a'])}")


if __name__ == "__main__":
    main()
