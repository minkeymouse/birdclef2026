"""OOF evaluation + full inference pipeline orchestration."""
# ── Cell 8: OOF evaluation (train mode only) ──────────────────────────
baseline_auc = None
oof_raw      = None
 
if CFG["run_oof"]:
    print("Running honest OOF evaluation on training data…")
    baseline_auc, oof_raw = honest_oof_auc(
        sc_tr, Y_FULL_aligned, meta_tr,
        n_splits=CFG["oof_n_splits"],
        label="raw Perch"
    )
    print(f"\nBaseline OOF AUC: {baseline_auc:.6f}  ← your starting point")
else:
    print("Submit mode: skipping OOF evaluation")
# ── Cell 8b: Full Pipeline OOF ─────────────────────────────────────────

def run_pipeline_oof(emb_full, sc_full, Y_full, meta_full, n_splits=5):
    """
    Proper full-pipeline OOF.
    Trains ProtoSSM + MLP on K-1 folds, predicts on held-out fold.
    ~3-4 min total on CPU. Use this instead of the raw-Perch OOF.
    """
    file_meta = (
        meta_full.drop_duplicates("filename")
        .reset_index(drop=True)
    )

    gkf = GroupKFold(n_splits=n_splits)
    oof_probs = np.zeros((len(sc_full), N_CLASSES), dtype=np.float32)

    for fold, (tr_f, va_f) in enumerate(
        gkf.split(file_meta, groups=file_meta["filename"]), 1
    ):
        tr_fnames = set(file_meta.iloc[tr_f]["filename"])
        va_fnames = set(file_meta.iloc[va_f]["filename"])

        tr_mask = meta_full["filename"].isin(tr_fnames).values
        va_mask = meta_full["filename"].isin(va_fnames).values

        emb_tr_f = emb_full[tr_mask]
        sc_tr_f = sc_full[tr_mask]
        Y_tr_f = Y_full[tr_mask]
        meta_tr_f = meta_full[tr_mask].reset_index(drop=True)

        emb_va_f = emb_full[va_mask]
        sc_va_f = sc_full[va_mask]
        meta_va_f = meta_full[va_mask].reset_index(drop=True)

        # ── Train ProtoSSM on train fold ───────────────────────────────
        proto_model, site2i = train_light_proto_ssm(
            emb_tr_f,
            sc_tr_f,
            Y_tr_f,
            meta_tr_f,
            n_epochs=40,
            patience=8,
            lr=1e-3,
            verbose=False,
        )

        # ── ProtoSSM predict on val fold ───────────────────────────────
        n_va = len(emb_va_f) // N_WINDOWS

        va_fn_list = (
            meta_va_f.drop_duplicates("filename")["filename"].tolist()
        )

        va_site_ids = np.array(
            [
                min(
                    site2i.get(
                        meta_va_f.loc[
                            meta_va_f["filename"] == fn, "site"
                        ].iloc[0],
                        0,
                    ),
                    19,
                )
                for fn in va_fn_list
            ],
            dtype=np.int64,
        )

        va_hour_ids = np.array(
            [
                int(
                    meta_va_f.loc[
                        meta_va_f["filename"] == fn, "hour_utc"
                    ].iloc[0]
                )
                % 24
                for fn in va_fn_list
            ],
            dtype=np.int64,
        )

        proto_model.eval()
        with torch.no_grad():
            proto_va = proto_model(
                torch.tensor(
                    emb_va_f.reshape(n_va, N_WINDOWS, -1),
                    dtype=torch.float32,
                ),
                torch.tensor(
                    sc_va_f.reshape(n_va, N_WINDOWS, -1),
                    dtype=torch.float32,
                ),
                site_ids=torch.tensor(va_site_ids, dtype=torch.long),
                hours=torch.tensor(va_hour_ids, dtype=torch.long),
            ).numpy().reshape(-1, N_CLASSES)

        # ── Train MLP on train fold ────────────────────────────────────
        probe_models, emb_scaler, emb_pca, alpha_blend = train_mlp_probes(
            emb_tr_f,
            sc_tr_f,
            Y_tr_f,
            min_pos=5,
            pca_dim=64,
            alpha_blend=0.4,
        )

        sc_va_mlp = apply_mlp_probes_vectorized(
            emb_va_f,
            sc_va_f,
            probe_models,
            emb_scaler,
            emb_pca,
            alpha_blend,
        )

        # ── Ensemble + sigmoid ─────────────────────────────────────────
        first_pass = 0.5 * proto_va + 0.5 * sc_va_mlp
        probs_va = 1.0 / (1.0 + np.exp(-np.clip(first_pass, -30, 30)))
        oof_probs[va_mask] = probs_va

        fold_auc = macro_auc(Y_full[va_mask], probs_va)
        print(
            f"  Fold {fold}/{n_splits}  val files={len(va_fnames)}  AUC={fold_auc:.6f}"
        )

    overall = macro_auc(Y_full, oof_probs)
    print(f"\nFull pipeline OOF AUC: {overall:.6f}")
    return overall, oof_probs


if CFG["run_oof"]:
    pipeline_auc, oof_pipeline = run_pipeline_oof(
        emb_tr,
        sc_tr,
        Y_FULL_aligned,
        meta_tr,
        n_splits=5,
    )
# ── Cell 9: Test inference ─────────────────────────────────────────────
test_paths = sorted((BASE / "test_soundscapes").glob("*.ogg"))
 
if not test_paths:
    n = CFG["dryrun_n_files"] or 20
    print(f"No hidden test — dry-run on {n} train files")
    test_paths = sorted((BASE / "train_soundscapes").glob("*.ogg"))[:n]
else:
    print(f"Hidden test files: {len(test_paths)}")
 
meta_te, sc_te, emb_te = run_perch(test_paths, CFG["batch_files"], verbose=CFG["verbose"])
print(f"Test scores: {sc_te.shape}")
# ── Cell 10: Full pipeline with ProtoSSM + ResidualSSM ─────────────────

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

# ── Step A: Train LightProtoSSM ────────────────────────────────────────
t0 = time.time()
proto_model, site2i_tr = train_light_proto_ssm(
    emb_tr, sc_tr, Y_FULL_aligned, meta_tr,
    n_epochs=40, patience=8, lr=1e-3, verbose=False)
print(f"ProtoSSM training: {time.time()-t0:.1f}s")

# ── Step B: Run ProtoSSM on TEST ───────────────────────────────────────
n_test_files  = len(sc_te) // N_WINDOWS
emb_te_f      = emb_te.reshape(n_test_files, N_WINDOWS, -1)
sc_te_f       = sc_te.reshape(n_test_files, N_WINDOWS, -1)

test_fnames   = meta_te.drop_duplicates("filename")["filename"].tolist()
n_sites_cap   = 20
test_site_ids = np.array([
    min(site2i_tr.get(
        meta_te.loc[meta_te["filename"]==fn,"site"].iloc[0], 0),
        n_sites_cap-1)
    for fn in test_fnames], dtype=np.int64)
test_hour_ids = np.array([
    int(meta_te.loc[meta_te["filename"]==fn,"hour_utc"].iloc[0]) % 24
    for fn in test_fnames], dtype=np.int64)

proto_model.eval()
with torch.no_grad():
    proto_out = proto_model(
        torch.tensor(emb_te_f, dtype=torch.float32),
        torch.tensor(sc_te_f,  dtype=torch.float32),
        site_ids=torch.tensor(test_site_ids, dtype=torch.long),
        hours   =torch.tensor(test_hour_ids, dtype=torch.long),
    ).numpy()
proto_scores_flat = proto_out.reshape(-1, N_CLASSES).astype(np.float32)

# ── Step C: Prior tables ───────────────────────────────────────────────
prior_tables   = build_prior_tables(sc, Y_SC)
sc_te_adjusted = apply_prior(
    sc_te,
    sites=meta_te["site"].to_numpy(),
    hours=meta_te["hour_utc"].to_numpy(),
    tables=prior_tables,
    lambda_prior=0.4,
)

# ── Step D: MLP probes ─────────────────────────────────────────────────
probe_models, emb_scaler, emb_pca, alpha_blend = train_mlp_probes(
    emb=emb_tr, scores_raw=sc_tr, Y=Y_FULL_aligned,
    min_pos=5, pca_dim=64, alpha_blend=0.4,
)
sc_te_adjusted = apply_mlp_probes_vectorized(
    emb_te, sc_te_adjusted,
    probe_models, emb_scaler, emb_pca, alpha_blend,
)

# ── Step E: First-pass ensemble (ProtoSSM + MLP) ───────────────────────
ENSEMBLE_W      = 0.5
first_pass_flat = (ENSEMBLE_W * proto_scores_flat
                   + (1.0 - ENSEMBLE_W) * sc_te_adjusted)

# ── Step F: ResidualSSM (second-pass correction) ───────────────────────
# Build training-data first-pass scores for residual training
n_tr_files    = len(sc_tr) // N_WINDOWS
emb_tr_f      = emb_tr.reshape(n_tr_files, N_WINDOWS, -1)
sc_tr_f       = sc_tr.reshape(n_tr_files, N_WINDOWS, -1)

tr_fnames     = meta_tr.drop_duplicates("filename")["filename"].tolist()
tr_site_ids   = np.array([
    min(site2i_tr.get(
        meta_tr.loc[meta_tr["filename"]==fn,"site"].iloc[0], 0),
        n_sites_cap-1)
    for fn in tr_fnames], dtype=np.int64)
tr_hour_ids   = np.array([
    int(meta_tr.loc[meta_tr["filename"]==fn,"hour_utc"].iloc[0]) % 24
    for fn in tr_fnames], dtype=np.int64)


# Get ProtoSSM scores on training data
# CORRECT — using emb_tr_f, sc_tr_f, tr_site_ids (train data)
proto_tr_out = run_tta_proto(
    proto_model, emb_tr_f, sc_tr_f,
    site_t=torch.tensor(tr_site_ids, dtype=torch.long),
    hour_t=torch.tensor(tr_hour_ids, dtype=torch.long),
    shifts=[0, 1, -1, 2, -2],
)

proto_tr_flat = proto_tr_out.reshape(-1, N_CLASSES).astype(np.float32)

# Get MLP scores on training data
sc_tr_prior   = apply_prior(
    sc_tr,
    sites=meta_tr["site"].to_numpy(),
    hours=meta_tr["hour_utc"].to_numpy(),
    tables=prior_tables,
    lambda_prior=0.4,
)
sc_tr_mlp = apply_mlp_probes_vectorized(
    emb_tr, sc_tr_prior,
    probe_models, emb_scaler, emb_pca, alpha_blend,
)
first_pass_tr = (ENSEMBLE_W * proto_tr_flat
                 + (1.0 - ENSEMBLE_W) * sc_tr_mlp)

train_probs_for_calib = sigmoid(first_pass_tr)
PER_CLASS_THRESHOLDS = calibrate_and_optimize_thresholds(
    oof_probs=train_probs_for_calib,
    Y_FULL=Y_FULL_aligned,
    threshold_grid=[0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70],
    n_windows=N_WINDOWS,
)


# Train ResidualSSM on training errors
t0 = time.time()
res_model, correction_weight = train_residual_ssm(
    emb_full=emb_tr,
    first_pass_flat=first_pass_tr,
    Y_full=Y_FULL_aligned,
    site_ids=tr_site_ids,
    hour_ids=tr_hour_ids,
    n_epochs=30,
    patience=8,
    lr=1e-3,
    correction_weight=0.30,
    verbose=False,
)
print(f"ResidualSSM training: {time.time()-t0:.1f}s")

# Apply ResidualSSM correction to TEST scores
first_pass_te_f  = first_pass_flat.reshape(n_test_files, N_WINDOWS, -1)
res_model.eval()
with torch.no_grad():
    test_correction = res_model(
        torch.tensor(emb_te_f,         dtype=torch.float32),
        torch.tensor(first_pass_te_f,  dtype=torch.float32),
        site_ids=torch.tensor(test_site_ids, dtype=torch.long),
        hours   =torch.tensor(test_hour_ids, dtype=torch.long),
    ).numpy()

correction_flat = test_correction.reshape(-1, N_CLASSES).astype(np.float32)
final_scores    = (first_pass_flat
                   + correction_weight * correction_flat)

print(f"Correction applied — "
      f"mean_abs={np.abs(correction_flat).mean():.4f}  "
      f"score range [{final_scores.min():.3f}, {final_scores.max():.3f}]")

# ── Step G: Temperature scaling ────────────────────────────────────────
final_scores = final_scores / temperatures[None, :]

# ── Step H: Sigmoid → probabilities ───────────────────────────────────
probs = sigmoid(final_scores)

# ── Step I: Post-processing pipeline ──────────────────────────────────
probs = file_confidence_scale(probs, n_windows=N_WINDOWS,
                               top_k=2,       power=0.4)
probs = rank_aware_scaling(   probs, n_windows=N_WINDOWS,
                               power=0.4)
probs = adaptive_delta_smooth(probs, n_windows=N_WINDOWS,
                               base_alpha=0.20)
probs = np.clip(probs, 0.0, 1.0)

# probs = apply_per_class_thresholds(probs, PER_CLASS_THRESHOLDS)

# ── Step J: Build submission ───────────────────────────────────────────
sub = pd.DataFrame(probs.astype(np.float32), columns=PRIMARY_LABELS)
sub.insert(0, "row_id", meta_te["row_id"].values)
assert list(sub.columns) == ["row_id"] + PRIMARY_LABELS
assert len(sub) == len(test_paths) * N_WINDOWS
assert not sub.isna().any().any()
sub.to_csv("submission_protossm.csv", index=False)                                                                                        
protossm_sub = sub.copy()

print(f"\nsubmission.csv saved — shape {sub.shape}")
print(f"Total wall time: {(time.time() - _WALL_START)/60:.1f} min")

del emb_tr_f, sc_tr_f, proto_model, res_model                                                                                             
gc.collect()                                                                                                                              
print("Memory freed. Ready for SED cell.")