#!/usr/bin/env python3
"""
inference.py — ensemble inference + ML-KFHE Kalman smoothing

* Loads process.yaml for metadata path.
* Discovers all model checkpoints under models/.
* Applies ensemble inference and multivariate Bernoulli KF smoothing to soundscape chunks.
"""
import logging
from pathlib import Path
import yaml
import numpy as np
import pandas as pd
import torch
from inference_model import InferenceModel, MultivariateBernoulliKalmanFilter

# ─── Load config ────────────────────────────────────────────
project_root = Path(__file__).resolve().parents[2]
config_path  = project_root / "config" / "process.yaml"
with open(config_path, "r", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)
paths_cfg = CFG["paths"]
# Convert to Path objects
paths_cfg["meta_data"] = Path(paths_cfg["meta_data"])
paths_cfg["mel_dir"]   = Path(paths_cfg["mel_dir"])
paths_cfg["label_dir"] = Path(paths_cfg["label_dir"])

# Models directory
models_dir = project_root / "models"

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("inference")

from typing import Union

def apply_power_to_low_ranked_cols(
    p: np.ndarray,
    top_k: int = 30,
    exponent: Union[int, float] = 2,
    inplace: bool = True
) -> np.ndarray:
    """
    Rank columns by their column‑wise maximum and raise every column whose
    rank falls below `top_k` to a given power.

    Parameters
    ----------
    p : np.ndarray
        A 2‑D array of shape **(n_chunks, n_classes)**.

        - **n_chunks** is the number of fixed‑length time chunks obtained
          after slicing the input audio (or other sequential data).  
          *Example:* In the BirdCLEF `test_soundscapes` set, each file is
          60 s long. If you extract non‑overlapping 5 s windows,  
          `n_chunks = 60 s / 5 s = 12`.
        - **n_classes** is the number of classes being predicted.
        - Each element `p[i, j]` is the score or probability of class *j*
          in chunk *i*.

    top_k : int, default=30
        The highest‑ranked columns (by their maximum value) that remain
        unchanged.

    exponent : int or float, default=2
        The power applied to the selected low‑ranked columns  
        (e.g. `2` squares, `0.5` takes the square root, `3` cubes).

    inplace : bool, default=True
        If `True`, modify `p` in place.  
        If `False`, operate on a copy and leave the original array intact.

    Returns
    -------
    np.ndarray
        The transformed array. It is the same object as `p` when
        `inplace=True`; otherwise, it is a new array.

    """
    if not inplace:
        p = p.copy()

    # Identify columns whose max value ranks below `top_k`
    tail_cols = np.argsort(-p.max(axis=0))[top_k:]

    # Apply the power transformation to those columns
    p[:, tail_cols] = p[:, tail_cols] ** exponent
    return p

def load_ensemble(models_dir: Path, num_classes: int, device: torch.device) -> list:
    models = []
    for ckpt in sorted(models_dir.glob("*.pth")):
        name = ckpt.stem.lower()
        if "efficientnet" in name:
            arch = 'efficientnet_b0'
        elif "regnety" in name:
            arch = 'regnety_008'
        elif "focal" in name:
            arch = 'efficientnet_b0'
        else:
            logger.warning("Skipping unknown checkpoint: %s", ckpt.name)
            continue

        model = InferenceModel(arch, in_chans=1, num_classes=num_classes).to(device)
        # load the raw state_dict
        bundle = torch.load(ckpt, map_location=device, weights_only=False)
        sd     = bundle['model_state_dict']

        # strip only the DDP "module." and then re-prefix under backbone
        cleaned = {}
        for k, v in sd.items():

            k2 = k.replace("module.", "")

            if k2.startswith("backbone."):
                cleaned[k2] = v
            else:
                cleaned[f"backbone.{k2}"] = v

        # now these keys exactly match what InferenceModel.backbone expects
        model.load_state_dict(cleaned)
        model.eval()
        models.append(model)
        logger.info("Loaded checkpoint: %s as %s", ckpt.name, arch)
    return models


def update_labels_for_group(df_group: pd.DataFrame, models: list, device: torch.device) -> None:
    df = df_group.sort_values('end_sec').reset_index(drop=True)
    # stack mel-spectrograms
    specs = [np.load(p) for p in df['mel_path']]
    specs = np.stack([s if s.ndim==3 else s[None,...] for s in specs], axis=0)

    # ensemble predictions
    all_preds = []
    batch = torch.tensor(specs, dtype=torch.float32, device=device)
    with torch.no_grad():
        for model in models:
            all_preds.append(model(batch).cpu().numpy())
    all_preds = np.stack(all_preds, axis=0)  # (M, T, C)
    M, T, C = all_preds.shape
    ensemble = all_preds.mean(axis=0)       # (T, C)
    ensemble = apply_power_to_low_ranked_cols(
        ensemble,
        top_k = 30,
        exponent=2,
        inplace=True
    )

    # Kalman Filter for probabilities
    kf = MultivariateBernoulliKalmanFilter(
        num_states=C,
        Q_method="kappa",
        kappa=0.05,
        constant_q=1e-3,
        include_offdiag=False,
        r_min=1e-3,
        missing_tau=0.01
    )
    kf.compute_R(ensemble)
    kf.compute_Q()
    kf.initialize(ensemble[0], P0=1e6)

    # Kalman Filter for model trust
    kf_w = MultivariateBernoulliKalmanFilter(
        num_states=M,
        Q_method="constant",
        constant_q=1e-2,
        include_offdiag=False,
        r_min=1e-3,
        missing_tau=0.0
    )
    init_trust = np.full(M, 0.5, dtype=np.float32)
    kf_w.initialize(init_trust, P0=1e2)

    # smoothing loop
    for t in range(T):
        kf.predict()
        kf_w.predict()
        state_pred = kf.probabilities  # (C,)

        # update trust
        residuals = np.mean(np.abs(all_preds[:,t,:] - state_pred), axis=1).astype(np.float32)
        kf_w.update(residuals)
        trust = kf_w.probabilities  # (M,)

        # sequential measurement updates
        for m in range(M):
            prev = all_preds[m, t-1] if t>0 else all_preds[m, t]
            curr = all_preds[m, t]
            nxt  = all_preds[m, t+1] if t<T-1 else all_preds[m, t]
            z_tm = 0.2*prev + 0.6*curr + 0.2*nxt
            z_he = 0.5*z_tm + 0.5*state_pred
            base_R = kf.Ry.diagonal()
            R_eff  = base_R / (trust[m] + 1e-3)
            kf.update(z_he, obs_var=R_eff)

        # save smoothed
        sm = kf.probabilities
        label_path = Path(df.loc[t, 'label_path'])
        np.save(label_path, sm)
        logger.info("Updated labels for %s", label_path.name)


def main():
    logger.info("Starting inference and smoothing...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # load metadata
    meta_df = pd.read_csv(paths_cfg["meta_data"])
    sc_df   = meta_df[meta_df['source']=="train_soundscape"]
    if sc_df.empty:
        logger.warning("No soundscape entries found in metadata.")
        return

    # infer num_classes
    num_classes = int(np.load(sc_df.iloc[0]['label_path']).shape[0])

    # load ensemble
    models = load_ensemble(models_dir, num_classes, device)
    if not models:
        logger.error("No models loaded. Exiting.")
        return

    # group by file
    for fname, grp in sc_df.groupby('filename'):
        logger.info("Processing soundscape: %s (%d chunks)", fname, len(grp))
        update_labels_for_group(grp, models, device)

    logger.info("Inference complete.")

if __name__ == "__main__":
    main()
