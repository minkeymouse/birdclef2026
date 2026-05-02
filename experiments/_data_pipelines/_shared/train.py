"""Generic SED training loop with masked BCE support."""
import json
import time
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from .constants import N_CLS, BATCH_SIZE


def macro_auc(y, p, n_cls=N_CLS):
    aucs = []
    for c in range(n_cls):
        if y[:, c].sum() == 0 or y[:, c].sum() == len(y): continue
        try: aucs.append(roc_auc_score(y[:, c], p[:, c]))
        except ValueError: pass
    return float(np.mean(aucs)) if aucs else float('nan'), len(aucs)


def per_taxon_auc(y, p, taxon_array, n_cls=N_CLS):
    """Compute per-taxon macro AUC."""
    res = {}
    for tx in ["Aves", "Amphibia", "Insecta", "Mammalia", "Reptilia"]:
        mask = (taxon_array == tx)
        if mask.sum() == 0:
            res[tx] = float('nan'); continue
        aucs = []
        for c in np.where(mask)[0]:
            if y[:, c].sum() == 0 or y[:, c].sum() == len(y): continue
            try: aucs.append(roc_auc_score(y[:, c], p[:, c]))
            except ValueError: pass
        res[tx] = float(np.mean(aucs)) if aucs else float('nan')
    return res


def train_sed_loop(model, train_loader, val_loader_ta, val_loader_ss,
                    bg_pool, taxon_array,
                    optim, scheduler, epochs, device,
                    out_dir, mixup_fn=None,
                    use_masked_bce=True, log_per_taxon=True):
    """Generic SED training loop with optional mask-aware BCE.

    Each batch yields (x, y, mask, primary_idx, is_ta).
    BCE loss: -[y log p + (1-y) log (1-p)] * mask, summed and divided by sum(mask).
    """
    best_val_ss = 0.0
    best_state = None
    history = []
    print(f"[train_sed_loop] Starting {epochs} epochs", flush=True)

    for ep in range(epochs):
        model.train()
        ep_loss = 0.0
        ep_mask_pct = 0.0  # diagnostic
        n_batches = 0
        t0 = time.time()
        print(f"[ep {ep}] iterating train_loader (len={len(train_loader)} batches)", flush=True)
        for batch_i, batch in enumerate(train_loader):
            if batch_i == 0:
                print(f"[ep {ep}] first batch received", flush=True)
            if batch_i % 200 == 0 and batch_i > 0:
                print(f"[ep {ep}] batch {batch_i}/{len(train_loader)} elapsed {(time.time()-t0)/60:.1f} min", flush=True)
            x, y, mask, primary_idx, is_ta = batch  # type: ignore[assignment]
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            if mixup_fn is not None:
                x_m, y_m = mixup_fn(x, y, primary_idx, bg_pool, taxon_array)
            else:
                x_m, y_m = x, y

            optim.zero_grad()
            clip, fmax = model(x_m, train=True)
            if use_masked_bce:
                # Per-cell BCE then masked mean
                bce_clip = F.binary_cross_entropy_with_logits(clip, y_m, reduction='none')
                bce_fmax = F.binary_cross_entropy_with_logits(fmax, y_m, reduction='none')
                # mask shape: (B, n_cls)
                eff_mask = mask
                # Mean over masked cells per row, then mean across batch
                loss_clip = (bce_clip * eff_mask).sum() / (eff_mask.sum() + 1e-9)
                loss_fmax = (bce_fmax * eff_mask).sum() / (eff_mask.sum() + 1e-9)
            else:
                loss_clip = F.binary_cross_entropy_with_logits(clip, y_m)
                loss_fmax = F.binary_cross_entropy_with_logits(fmax, y_m)
            loss = 0.5 * loss_clip + 0.5 * loss_fmax
            loss.backward()
            optim.step()
            ep_loss += loss.item()
            ep_mask_pct += float(mask.mean())
            n_batches += 1
        scheduler.step()

        model.eval()
        all_y_ta, all_p_ta = [], []
        all_y_ss, all_p_ss = [], []
        with torch.no_grad():
            for x, y, _, _, _ in val_loader_ta:
                x = x.to(device); clip, _ = model(x, train=False)
                all_y_ta.append(y.numpy()); all_p_ta.append(torch.sigmoid(clip).cpu().numpy())
            for x, y, _, _, _ in val_loader_ss:
                x = x.to(device); clip, _ = model(x, train=False)
                all_y_ss.append(y.numpy()); all_p_ss.append(torch.sigmoid(clip).cpu().numpy())
        all_y_ta = np.concatenate(all_y_ta); all_p_ta = np.concatenate(all_p_ta)
        all_y_ss = np.concatenate(all_y_ss); all_p_ss = np.concatenate(all_p_ss)

        val_ta, _ = macro_auc(all_y_ta, all_p_ta)
        val_ss, _ = macro_auc(all_y_ss, all_p_ss)

        if log_per_taxon:
            pt = per_taxon_auc(all_y_ss, all_p_ss, taxon_array)
        else:
            pt = {}

        elapsed = time.time() - t0
        avg_mask = ep_mask_pct / max(n_batches, 1)
        print(f"ep {ep:02d}  loss {ep_loss/n_batches:.4f}  mask_avg {avg_mask:.3f}  "
              f"val_TA {val_ta:.4f}  val_SS {val_ss:.4f}  "
              + (" ".join(f"{tx[:2]} {pt.get(tx, float('nan')):.3f}" for tx in ["Aves","Amphibia","Insecta","Mammalia"]))
              + f"  ({elapsed/60:.1f} min)", flush=True)

        if val_ss > best_val_ss:
            best_val_ss = val_ss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            torch.save({"state_dict": best_state, "val_TA": val_ta, "val_SS": val_ss,
                          "epoch": ep},
                         out_dir / "best_ckpt.pt")
            print(f"  -> saved best ckpt @ ep{ep:02d}")
        history.append({"epoch": ep, "loss": ep_loss/n_batches, "mask_avg": avg_mask,
                          "val_TA": val_ta, "val_SS": val_ss, **pt})

    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    return best_val_ss, best_state, history
