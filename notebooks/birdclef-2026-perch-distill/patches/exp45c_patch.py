# ─────────────────────────────────────────────────────────────────────────
# exp45c V9 patch — Soft taxon gating (additive 0.1 offset).
#
# Replaces the exp44c 27-species overlay. Instead of overwriting 27 columns,
# re-weights ALL 234 species columns:
#   final_prob[:, c] = probs[:, c] * (taxon_prob_for_species(c) + 0.1)
#
# Taxon head: 2-layer MLP 1536 → 256 → 5 (Aves/Amphibia/Insecta/Mammalia/Reptilia)
# Trained on train_audio 35549 + labeled SS 617 windows (~36k samples).
# Val-A_v2 (40 classes): raw Perch 0.622 → taxon-gated V9 0.767 (+0.145).
#
# Runtime impact: negligible (~1s on Kaggle CPU).
# ─────────────────────────────────────────────────────────────────────────

# --- Load taxon head ---
_TAXON_CKPT_PATHS = [
    Path('/kaggle/input/datasets/ultimatumgame/birdclef2026-model-weights/exp45a_taxon_head.pt'),
    Path('/kaggle/input/birdclef2026-model-weights/exp45a_taxon_head.pt'),
]
_taxon_ckpt_path = next((p for p in _TAXON_CKPT_PATHS if p.exists()), None)

if _taxon_ckpt_path is None:
    print("exp45c: taxon head NOT found, skipping taxon gating")
else:
    import torch, torch.nn as nn
    _TAXA = ["Aves", "Amphibia", "Insecta", "Mammalia", "Reptilia"]

    class _TaxonHead(nn.Module):
        def __init__(self, in_dim=1536, hidden=256, n_taxa=5):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(0.2),
                nn.Linear(hidden, n_taxa),
            )
        def forward(self, x): return self.net(x)

    _ckpt = torch.load(_taxon_ckpt_path, map_location='cpu', weights_only=False)
    _taxon_model = _TaxonHead()
    _taxon_model.load_state_dict(_ckpt["state_dict"])
    _taxon_model.eval()
    _species_to_taxon = np.asarray(_ckpt["species_to_taxon"], dtype=np.int64)
    assert len(_species_to_taxon) == N_CLASSES, \
        f"taxon map len {len(_species_to_taxon)} != N_CLASSES {N_CLASSES}"

    print(f"exp45c: taxon head loaded  taxa={_TAXA}")

    # --- Predict per-row taxon probs from Perch embeddings ---
    # emb_test is (n_rows, 1536) aligned with test_paths × N_WINDOWS
    import time as _t
    _t0 = _t.time()
    with torch.no_grad():
        _emb_t = torch.from_numpy(emb_test.astype(np.float32))
        _taxon_logit = _taxon_model(_emb_t)
        _taxon_prob = torch.sigmoid(_taxon_logit).numpy()          # (n_rows, 5)
    print(f"exp45c: taxon inference {_t.time()-_t0:.1f}s  shape={_taxon_prob.shape}")

    # --- Soft gating V9: probs[:, c] *= (taxon_prob[:, taxon(c)] + 0.1) ---
    _tprobs_per_sp = _taxon_prob[:, _species_to_taxon]             # (n_rows, 234)
    _gate = np.clip(_tprobs_per_sp + 0.1, 0.0, 1.0)
    probs = probs * _gate

    print(f"exp45c: applied V9 soft taxon gating (offset 0.1)")
    print(f"  taxon mean prob per taxon: {dict(zip(_TAXA, _taxon_prob.mean(0).round(3)))}")
    print(f"  probs range after gating: [{probs.min():.6f}, {probs.max():.6f}]")
