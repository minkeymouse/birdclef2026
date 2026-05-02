# ─────────────────────────────────────────────────────────────────────────
# v43 P_NEW3 hybrid blend (exp103-106)
#
# Adds a small (w=0.10) blend with P_NEW3, a Perch-init hybrid head trained
# on TA + labeled SS. Architecture: frozen Perch-init Linear (extracted from
# Perch's 14k ProtoPNet head, mapped to our 234) + trainable correction MLP
# (init=0). At init, output = Perch baseline; correction adds learned signal.
#
# Local validation:
#   Same-site eval (122): macro 0.870 (P_NEW3 alone) vs Perch 0.622
#   LOSO mean: 0.767 vs P_NEW1 (rand) 0.751 — generalization improved
#   Aves LOSO: 0.831 (transferable), Insecta LOSO: 0.516 (site shortcut)
#
# Blend test on 122 eval (vs v33 base):
#   v33 + P_NEW3 w=0.10 → macro_d +0.125, sp_row 0.995, Aves +0.057
#   All taxa positive. sp_row higher than v34 (0.99) and v36 (0.999).
#
# Mechanism: training-time supervised correction with Perch prior — different
# category from the 11 inference-time levers tested previously. Risk: P_NEW3
# correction MLP fits 5-site labeled SS (same data that v34/v36 used).
# w=0.10 is the most conservative blend: macro swing smaller, sp_row safest.
# ─────────────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path

PNEW3_CKPT_PATHS = [
    Path('/kaggle/input/datasets/ultimatumgame/birdclef2026-model-weights/p_new3_hybrid.pt'),
    Path('/kaggle/input/birdclef2026-model-weights/p_new3_hybrid.pt'),
]
PERCH_HEAD_NPZ_PATHS = [
    Path('/kaggle/input/datasets/ultimatumgame/birdclef2026-model-weights/perch_head_extracted.npz'),
    Path('/kaggle/input/birdclef2026-model-weights/perch_head_extracted.npz'),
]
PNEW3_AUX_PATHS = [
    Path('/kaggle/input/datasets/ultimatumgame/birdclef2026-model-weights/p_new3_aux.npz'),
    Path('/kaggle/input/birdclef2026-model-weights/p_new3_aux.npz'),
]
_pnew3_ckpt = next((p for p in PNEW3_CKPT_PATHS if p.exists()), None)
_perch_head = next((p for p in PERCH_HEAD_NPZ_PATHS if p.exists()), None)
_pnew3_aux = next((p for p in PNEW3_AUX_PATHS if p.exists()), None)

# v43 hyperparameter — conservative blend weight
W_PNEW3 = 0.10

if _pnew3_ckpt is None or _perch_head is None or _pnew3_aux is None:
    print(f"v43 P_NEW3: artifacts missing (ckpt={_pnew3_ckpt}, head={_perch_head}, aux={_pnew3_aux}), skipping")
elif emb_test is None:
    print(f"v43 P_NEW3: emb_test missing — skipping")
else:
    print(f"\nv43 P_NEW3 hybrid: applying with W_PNEW3={W_PNEW3}", flush=True)

    # Load Perch head extraction
    _head = np.load(_perch_head)
    _W_proto = _head["W"]   # (1536, 14795, 4)
    _B = _head["B"]         # (14795,)
    _S = _head["S"]         # (14795, 4)
    _W_eff = (_W_proto * _S[None, :, :]).sum(axis=-1).astype(np.float32)  # (1536, 14795)

    # Load precomputed mapping (sorted PRIMARY_LABELS + perch_idx for each)
    _aux = np.load(_pnew3_aux)
    _mapped_perch_idx = _aux["perch_mapping"]  # (234,) int64
    _aux_labels = _aux["primary_labels_sorted"].tolist()
    # PRIMARY_LABELS from notebook should match the sample_submission column order
    if list(PRIMARY_LABELS) != _aux_labels:
        print(f"v43 P_NEW3: WARNING — PRIMARY_LABELS != aux labels order. Using aux ordering.")
    _n_mapped = int((_mapped_perch_idx >= 0).sum())
    print(f"  Perch mapping (precomputed): {_n_mapped}/{N_CLASSES} mapped")

    # Build init weights for our 234-class head (matches exp106 PerchHybrid)
    _rng = np.random.RandomState(42)
    _W_init = np.zeros((1536, N_CLASSES), dtype=np.float32)
    _b_init = np.zeros(N_CLASSES, dtype=np.float32)
    for _c in range(N_CLASSES):
        if _mapped_perch_idx[_c] >= 0:
            _W_init[:, _c] = _W_eff[:, _mapped_perch_idx[_c]]
            _b_init[_c] = _B[_mapped_perch_idx[_c]]
        else:
            _bound = np.sqrt(6.0 / 1536)
            _W_init[:, _c] = _rng.uniform(-_bound, _bound, size=1536)
    if True:

        class _PerchHybrid(nn.Module):
            def __init__(self, W_init_, b_init_, hidden=768, dropout=0.3):
                super().__init__()
                self.perch_fc = nn.Linear(1536, N_CLASSES)
                with torch.no_grad():
                    self.perch_fc.weight.copy_(torch.from_numpy(W_init_.T))
                    self.perch_fc.bias.copy_(torch.from_numpy(b_init_))
                for _p in self.perch_fc.parameters():
                    _p.requires_grad_(False)
                self.bn = nn.BatchNorm1d(1536)
                self.fc1 = nn.Linear(1536, hidden)
                self.dropout = nn.Dropout(dropout)
                self.fc2 = nn.Linear(hidden, N_CLASSES)

            def forward(self, x):
                x_norm = F.normalize(x, dim=-1, eps=1e-6)
                perch_logit = self.perch_fc(x_norm)
                h = self.bn(x)
                h = F.gelu(self.fc1(h))
                h = self.dropout(h)
                return perch_logit + self.fc2(h)

        _model = _PerchHybrid(_W_init, _b_init).eval()
        _ckpt = torch.load(str(_pnew3_ckpt), map_location='cpu', weights_only=False)
        _model.load_state_dict(_ckpt["state_dict"])

        # Inference on test embeddings (CPU since Kaggle runtime is CPU-only)
        _emb_t = torch.from_numpy(emb_test.astype(np.float32))
        with torch.inference_mode():
            _logits = _model(_emb_t)
            _pnew3_pred = torch.sigmoid(_logits).numpy().astype(np.float32)
        print(f"  P_NEW3 predictions: {_pnew3_pred.shape}, range [{_pnew3_pred.min():.5f}, {_pnew3_pred.max():.5f}]")

        # Blend: probs (currently v33 state) + small P_NEW3 component
        probs = (1.0 - W_PNEW3) * probs + W_PNEW3 * _pnew3_pred
        probs = np.clip(probs, 0.0, 1.0).astype(np.float32)
        print(f"  v43 P_NEW3 blend applied. probs range: [{probs.min():.5f}, {probs.max():.5f}]")
        del _model, _ckpt, _emb_t, _logits, _pnew3_pred
