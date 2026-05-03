"""LightProtoSSM (state-space + cross-attention) + ResidualSSM."""
# ── Cell 7i: LightProtoSSM WITH Cross-Attention ────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F

class SelectiveSSM(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.in_proj = nn.Linear(d_model, 2 * d_model, bias=False)
        self.conv1d = nn.Conv1d(
            d_model, d_model, d_conv, padding=d_conv - 1, groups=d_model
        )
        self.dt_proj = nn.Linear(d_model, d_model, bias=True)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(d_model, -1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_model))
        self.B_proj = nn.Linear(d_model, d_state, bias=False)
        self.C_proj = nn.Linear(d_model, d_state, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B_sz, T, D = x.shape
        xz = self.in_proj(x)
        x_ssm, z = xz.chunk(2, dim=-1)
        x_conv = self.conv1d(x_ssm.transpose(1, 2))[:, :, :T].transpose(1, 2)
        x_conv = F.silu(x_conv)
        dt = F.softplus(self.dt_proj(x_conv))
        A = -torch.exp(self.A_log)
        B = self.B_proj(x_conv)
        C = self.C_proj(x_conv)
        h = torch.zeros(B_sz, D, self.d_state)
        ys = []
        for t in range(T):
            dA = torch.exp(A[None] * dt[:, t, :, None])
            dB = dt[:, t, :, None] * B[:, t, None, :]
            h = h * dA + x[:, t, :, None] * dB
            ys.append((h * C[:, t, None, :]).sum(-1))
        y = torch.stack(ys, dim=1)
        return y + x * self.D[None, None, :]


class LightProtoSSM(nn.Module):
    def __init__(self, d_input=1536, d_model=128, d_state=16,
                 n_classes=234, n_windows=12, dropout=0.15,
                 n_sites=20, meta_dim=16,
                 use_cross_attn=True, cross_attn_heads=2):
        super().__init__()
        self.n_classes = n_classes
        self.n_windows = n_windows
        self.use_cross_attn = use_cross_attn

        self.input_proj = nn.Sequential(
            nn.Linear(d_input, d_model),
            nn.LayerNorm(d_model), nn.GELU(), nn.Dropout(dropout))
        self.pos_enc  = nn.Parameter(torch.randn(1, n_windows, d_model) * 0.02)
        self.site_emb = nn.Embedding(n_sites, meta_dim)
        self.hour_emb = nn.Embedding(24, meta_dim)
        self.meta_proj = nn.Linear(2 * meta_dim, d_model)

        self.ssm_fwd  = nn.ModuleList([SelectiveSSM(d_model, d_state) for _ in range(2)])
        self.ssm_bwd  = nn.ModuleList([SelectiveSSM(d_model, d_state) for _ in range(2)])
        self.ssm_merge= nn.ModuleList([nn.Linear(2 * d_model, d_model) for _ in range(2)])
        self.ssm_norm = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(2)])
        self.drop     = nn.Dropout(dropout)

        if use_cross_attn:
            self.cross_attn = nn.ModuleList([
                nn.MultiheadAttention(d_model, num_heads=cross_attn_heads,
                                      dropout=dropout, batch_first=True)
                for _ in range(2)])
            self.cross_norm = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(2)])

        self.prototypes   = nn.Parameter(torch.randn(n_classes, d_model) * 0.02)
        self.proto_temp   = nn.Parameter(torch.tensor(5.0))
        self.class_bias   = nn.Parameter(torch.zeros(n_classes))
        self.fusion_alpha = nn.Parameter(torch.zeros(n_classes))

    def init_prototypes(self, emb_tensor, labels_tensor):
        with torch.no_grad():
            h = self.input_proj(emb_tensor)
            for c in range(self.n_classes):
                mask = labels_tensor[:, c] > 0.5
                if mask.sum() > 0:
                    self.prototypes.data[c] = F.normalize(h[mask].mean(0), dim=0)

    def forward(self, emb, perch_logits=None, site_ids=None, hours=None):
        B, T, _ = emb.shape
        h = self.input_proj(emb) + self.pos_enc[:, :T, :]
        if site_ids is not None and hours is not None:
            meta = self.meta_proj(torch.cat(
                [self.site_emb(site_ids), self.hour_emb(hours)], dim=-1))
            h = h + meta[:, None, :]

        for i, (fwd, bwd, merge, norm) in enumerate(zip(
                self.ssm_fwd, self.ssm_bwd, self.ssm_merge, self.ssm_norm)):
            res = h
            h_f = fwd(h); h_b = bwd(h.flip(1)).flip(1)
            h   = self.drop(merge(torch.cat([h_f, h_b], dim=-1)))
            h   = norm(h + res)
            if self.use_cross_attn:
                attn_out, _ = self.cross_attn[i](h, h, h)
                h = self.cross_norm[i](h + attn_out)

        h_n = F.normalize(h, dim=-1)
        p_n = F.normalize(self.prototypes, dim=-1)
        sim = (torch.matmul(h_n, p_n.T) * F.softplus(self.proto_temp)
               + self.class_bias[None, None, :])
        if perch_logits is not None:
            alpha = torch.sigmoid(self.fusion_alpha)[None, None, :]
            out   = alpha * sim + (1 - alpha) * perch_logits
        else:
            out = sim
        return out

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def train_light_proto_ssm(emb_full, scores_full, Y_full, meta_full,
                           n_epochs=40, patience=8, lr=1e-3,
                           n_sites=20, verbose=False):
    """Train LightProtoSSM with cross-attention + SWA."""
    n_files = len(emb_full) // N_WINDOWS
    emb_f   = emb_full.reshape(n_files, N_WINDOWS, -1)
    log_f   = scores_full.reshape(n_files, N_WINDOWS, -1)
    lab_f   = Y_full.reshape(n_files, N_WINDOWS, -1).astype(np.float32)

    fnames  = meta_full["filename"].unique()
    sites_u = sorted(meta_full["site"].unique())
    site2i  = {s: i + 1 for i, s in enumerate(sites_u)}

    site_ids = np.array([
        min(site2i.get(meta_full.loc[meta_full["filename"]==fn,"site"].iloc[0], 0), n_sites-1)
        for fn in fnames], dtype=np.int64)
    hour_ids = np.array([
        int(meta_full.loc[meta_full["filename"]==fn,"hour_utc"].iloc[0]) % 24
        for fn in fnames], dtype=np.int64)

    model = LightProtoSSM(n_classes=N_CLASSES, n_sites=n_sites,
                          use_cross_attn=True, cross_attn_heads=2)
    model.init_prototypes(
        torch.tensor(emb_full, dtype=torch.float32),
        torch.tensor(Y_full,   dtype=torch.float32))
    print(f"LightProtoSSM params: {model.count_parameters():,}")

    emb_t  = torch.tensor(emb_f,    dtype=torch.float32)
    log_t  = torch.tensor(log_f,    dtype=torch.float32)
    lab_t  = torch.tensor(lab_f,    dtype=torch.float32)
    site_t = torch.tensor(site_ids, dtype=torch.long)
    hour_t = torch.tensor(hour_ids, dtype=torch.long)

    pos_cnt    = lab_t.sum(dim=(0, 1))
    total      = lab_t.shape[0] * lab_t.shape[1]
    pos_weight = ((total - pos_cnt) / (pos_cnt + 1)).clamp(max=25.0)

    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, epochs=n_epochs, steps_per_epoch=1,
        pct_start=0.1, anneal_strategy="cos")

    best_loss, best_state, wait = float("inf"), None, 0

    # ── SWA setup ──────────────────────────────────────────────────────
    swa_model = torch.optim.swa_utils.AveragedModel(model)
    swa_start = int(n_epochs * 0.65)
    swa_sched = torch.optim.swa_utils.SWALR(opt, swa_lr=4e-4)

    for ep in range(n_epochs):
        model.train()
        out  = model(emb_t, log_t, site_ids=site_t, hours=hour_t)
        loss = (F.binary_cross_entropy_with_logits(
                    out, lab_t, pos_weight=pos_weight[None, None, :])
                + 0.15 * F.mse_loss(out, log_t))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        # ── SWA update ─────────────────────────────────────────────────
        if ep >= swa_start:
            swa_model.update_parameters(model)
            swa_sched.step()
        else:
            sched.step()

        if loss.item() < best_loss:
            best_loss  = loss.item()
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        if wait >= patience:
            if verbose: print(f"  Early stop ep {ep+1}")
            break

    # ── Use SWA model if we reached swa_start, else best checkpoint ────
    if ep >= swa_start:
        torch.optim.swa_utils.update_bn(emb_t.unsqueeze(0), swa_model)
        model = swa_model
    else:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        out = model(emb_t, log_t, site_ids=site_t, hours=hour_t)
    print(f"LightProtoSSM trained — best loss={best_loss:.4f}")
    return model, site2i


print("✅ CHANGE 4: LightProtoSSM with cross-attention (2 heads) + SWA defined")
# ── Cell 7i-2: TTA — Circular Shift Test-Time Augmentation ───────────
# CHANGE 3: Average ProtoSSM predictions across 5 time shifts
# Expected gain: +0.003–0.005 on public LB

def run_tta_proto(proto_model, emb_files, sc_files,
                  site_t, hour_t, shifts=[0, 1, -1, 2, -2]):
    """
    CHANGE 3: TTA by circular-shifting 12-window sequences.
    
    For each shift s:
      1. Roll embeddings and perch logits by s windows
      2. Run ProtoSSM → get predictions
      3. Roll predictions back by -s (undo shift)
    
    Finally average all predictions across shifts.
    
    Why this works:
      - ProtoSSM sees temporal context across all 12 windows
      - Different starting points expose different context patterns
      - Averaging over 5 views reduces temporal boundary artifacts
    """
    proto_model.eval()
    all_preds = []
    
    emb_t  = torch.tensor(emb_files, dtype=torch.float32)
    sc_t   = torch.tensor(sc_files,  dtype=torch.float32)
    
    for shift in shifts:
        if shift == 0:
            e_shifted = emb_t
            s_shifted = sc_t
        else:
            e_shifted = torch.roll(emb_t, shift, dims=1)
            s_shifted = torch.roll(sc_t,  shift, dims=1)
        
        with torch.no_grad():
            out = proto_model(
                e_shifted, s_shifted,
                site_ids=site_t, hours=hour_t
            ).numpy()   # (n_files, 12, 234)
        
        if shift != 0:
            out = np.roll(out, -shift, axis=1)  # undo shift
        
        all_preds.append(out)
    
    return np.mean(all_preds, axis=0)  # (n_files, 12, 234)

print("✅ CHANGE 3: TTA with 5 circular shifts defined")

# ── Cell 7j: Residual SSM (second-pass error correction) ──────────────
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualSSM(nn.Module):
    """
    Lightweight second-pass model that learns to correct
    systematic errors from the first-pass ensemble.
    
    Input:  embeddings + first-pass scores (concatenated)
    Output: additive correction to first-pass scores
    
    Key design: output head initialized to zero
    so corrections start small and only grow if helpful.
    ~25s training on 59 files.
    """
    def __init__(self, d_input=1536, d_scores=234,
                 d_model=64, d_state=8,
                 n_classes=234, n_windows=12,
                 dropout=0.1, n_sites=20, meta_dim=8):
        super().__init__()
        self.n_classes = n_classes

        self.input_proj = nn.Sequential(
            nn.Linear(d_input + d_scores, d_model),
            nn.LayerNorm(d_model), nn.GELU(), nn.Dropout(dropout))

        self.site_emb  = nn.Embedding(n_sites, meta_dim)
        self.hour_emb  = nn.Embedding(24,      meta_dim)
        self.meta_proj = nn.Linear(2 * meta_dim, d_model)
        self.pos_enc   = nn.Parameter(
            torch.randn(1, n_windows, d_model) * 0.02)

        self.ssm_fwd   = SelectiveSSM(d_model, d_state)
        self.ssm_bwd   = SelectiveSSM(d_model, d_state)
        self.ssm_merge = nn.Linear(2 * d_model, d_model)
        self.ssm_norm  = nn.LayerNorm(d_model)
        self.ssm_drop  = nn.Dropout(dropout)

        self.output_head = nn.Linear(d_model, n_classes)
        # Zero init — corrections start at zero, only grow if helpful
        nn.init.zeros_(self.output_head.weight)
        nn.init.zeros_(self.output_head.bias)

    def forward(self, emb, first_pass, site_ids=None, hours=None):
        B, T, _ = emb.shape
        x = torch.cat([emb, first_pass], dim=-1)
        h = self.input_proj(x) + self.pos_enc[:, :T, :]

        if site_ids is not None and hours is not None:
            meta = self.meta_proj(torch.cat(
                [self.site_emb(site_ids.clamp(0, self.site_emb.num_embeddings-1)),
                 self.hour_emb(hours.clamp(0, 23))], dim=-1))
            h = h + meta.unsqueeze(1)

        res = h
        h_f = self.ssm_fwd(h)
        h_b = self.ssm_bwd(h.flip(1)).flip(1)
        h   = self.ssm_drop(self.ssm_merge(
            torch.cat([h_f, h_b], dim=-1)))
        h   = self.ssm_norm(h + res)

        return self.output_head(h)   # (B, T, n_classes)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters()
                   if p.requires_grad)


def train_residual_ssm(emb_full, first_pass_flat, Y_full,
                       site_ids, hour_ids,
                       n_epochs=30, patience=8, lr=1e-3,
                       correction_weight=0.30,
                       verbose=False):
    """
    Train ResidualSSM to predict (Y - sigmoid(first_pass)).
    Returns corrected flat scores (n_rows, n_classes).
    ~20s on CPU.
    """
    n_files    = len(emb_full) // N_WINDOWS
    emb_f      = emb_full.reshape(n_files, N_WINDOWS, -1)
    fp_f       = first_pass_flat.reshape(n_files, N_WINDOWS, -1)
    lab_f      = Y_full.reshape(n_files, N_WINDOWS, -1).astype(np.float32)

    # Residual target = label - sigmoid(first_pass)
    fp_prob    = 1.0 / (1.0 + np.exp(-np.clip(fp_f, -30, 30)))
    residuals  = lab_f - fp_prob   # values in [-1, 1]

    print(f"Residuals: mean={residuals.mean():.4f}  "
          f"std={residuals.std():.4f}  "
          f"abs_mean={np.abs(residuals).mean():.4f}")

    # Train / val split (file level, no shuffle leakage)
    n_val    = max(1, int(n_files * 0.15))
    rng      = torch.Generator(); rng.manual_seed(42)
    perm     = torch.randperm(n_files, generator=rng).numpy()
    val_i    = perm[:n_val];  train_i = perm[n_val:]

    emb_t    = torch.tensor(emb_f,    dtype=torch.float32)
    fp_t     = torch.tensor(fp_f,     dtype=torch.float32)
    res_t    = torch.tensor(residuals, dtype=torch.float32)
    site_t   = torch.tensor(site_ids, dtype=torch.long)
    hour_t   = torch.tensor(hour_ids, dtype=torch.long)

    model    = ResidualSSM(n_classes=N_CLASSES)
    print(f"ResidualSSM params: {model.count_parameters():,}")

    opt      = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=1e-3)
    sched    = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, epochs=n_epochs, steps_per_epoch=1,
        pct_start=0.1, anneal_strategy="cos")

    best_loss, best_state, wait = float("inf"), None, 0

    for ep in range(n_epochs):
        model.train()
        corr = model(emb_t[train_i], fp_t[train_i],
                     site_ids=site_t[train_i],
                     hours   =hour_t[train_i])
        loss = F.mse_loss(corr, res_t[train_i])
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()

        model.eval()
        with torch.no_grad():
            val_corr = model(emb_t[val_i], fp_t[val_i],
                             site_ids=site_t[val_i],
                             hours   =hour_t[val_i])
            val_loss = F.mse_loss(val_corr, res_t[val_i])

        if val_loss.item() < best_loss:
            best_loss  = val_loss.item()
            best_state = {k: v.clone()
                          for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        if wait >= patience:
            if verbose: print(f"  Early stop ep {ep+1}")
            break

    model.load_state_dict(best_state)
    print(f"ResidualSSM trained — best val MSE={best_loss:.6f}")

    # Apply correction to ALL training data (for verification)
    model.eval()
    with torch.no_grad():
        all_corr = model(emb_t, fp_t,
                         site_ids=site_t,
                         hours   =hour_t).numpy()
    print(f"Correction magnitude: "
          f"mean_abs={np.abs(all_corr).mean():.4f}  "
          f"max={np.abs(all_corr).max():.4f}")

    return model, correction_weight


print("✅ ResidualSSM defined (~439K params, ~20s training)")