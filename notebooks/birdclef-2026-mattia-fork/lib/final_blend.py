"""Final rank-percentile blend + 3 conditional rescue rules."""
# Cell 3 — SED-smoothed rank ensemble + Proto fat-tail continuity gates

import numpy as np
import pandas as pd

PROTOSSM_CSV = "submission_protossm.csv"
SED_CSV     = "submission_sed.csv"
OUT_CSV     = "submission.csv"

EPS = 1e-5
SED_W = 0.40

# calibrated PROTOSSM/ProtoSSM rescue
FAKE_ONLY_THR   = 0.50
SED_LOW_THR     = 0.05
FAKE_ONLY_BLEND = 0.12

# Proto temporal continuity rescue: wider fat-tailed context
PROTO_CONT_RADIUS     = 3      # +/-3 windows = soft 35s context
PROTO_CONT_DF         = 2.0
PROTO_CONT_SCALE      = 1.20
PROTO_CONT_RANK_THR   = 0.88
PROTO_LOCAL_RANK_THR  = 0.75
SED_CONT_LOW_THR      = 0.12
PROTO_CONT_BLEND      = 0.15

# rare local SED spike rescue
SED_ONLY_RANK_THR = 0.95
FAKE_RANK_LOW_THR = 0.80
SED_ONLY_BLEND    = 0.12

a = pd.read_csv(PROTOSSM_CSV)
b = pd.read_csv(SED_CSV)

cols = [c for c in a.columns if c != "row_id"]
b = b.set_index("row_id").loc[a["row_id"]].reset_index()

pa = np.clip(a[cols].to_numpy(np.float32), EPS, 1.0 - EPS)
pb = np.clip(b[cols].to_numpy(np.float32), EPS, 1.0 - EPS)

row_ids = a["row_id"].astype(str).to_numpy()
file_ids = np.array(["_".join(r.split("_")[:-1]) for r in row_ids])

# rank blend
xa = pd.DataFrame(pa).rank(axis=0, pct=True).to_numpy(np.float32)
xb = pd.DataFrame(pb).rank(axis=0, pct=True).to_numpy(np.float32)

pred = xa * (1.0 - SED_W) + xb * SED_W

# 1) Proto/PROTOSSM calibrated rescue
fake_only = (pa > FAKE_ONLY_THR) & (pb < SED_LOW_THR)
pred = np.where(fake_only, (1.0 - FAKE_ONLY_BLEND) * pred + FAKE_ONLY_BLEND * xa, pred)

# 2) Proto temporal-continuity rescue with fat-tailed context
offs = np.arange(-PROTO_CONT_RADIUS, PROTO_CONT_RADIUS + 1, dtype=np.float32)
proto_kernel = (1.0 + (offs / PROTO_CONT_SCALE) ** 2 / PROTO_CONT_DF) ** (-(PROTO_CONT_DF + 1.0) / 2.0)
proto_kernel = (proto_kernel / proto_kernel.sum()).astype(np.float32)

pa_ctx = pa.copy()
R = PROTO_CONT_RADIUS

for fid in pd.unique(file_ids):
    m = file_ids == fid
    x = pa[m]
    if len(x) > 1:
        xp = np.pad(x, ((R, R), (0, 0)), mode="edge")
        pa_ctx[m] = sum(proto_kernel[i] * xp[i:i + len(x)] for i in range(2 * R + 1))

xctx = pd.DataFrame(pa_ctx).rank(axis=0, pct=True).to_numpy(np.float32)

proto_cont = (
    (xctx > PROTO_CONT_RANK_THR) &
    (xa > PROTO_LOCAL_RANK_THR) &
    (pb < SED_CONT_LOW_THR) &
    (~fake_only)
)

pred = np.where(
    proto_cont,
    (1.0 - PROTO_CONT_BLEND) * pred + PROTO_CONT_BLEND * np.maximum(xa, xctx),
    pred,
)

# 3) rare SED local spike rescue
sed_only = (
    (xb > SED_ONLY_RANK_THR) &
    (xa < FAKE_RANK_LOW_THR) &
    (~fake_only) &
    (~proto_cont)
)

pred = np.where(sed_only, (1.0 - SED_ONLY_BLEND) * pred + SED_ONLY_BLEND * xb, pred)

sub = a.copy()
sub[cols] = pred.astype(np.float32)
sub.to_csv(OUT_CSV, index=False)
