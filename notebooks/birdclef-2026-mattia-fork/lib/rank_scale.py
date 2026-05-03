"""Rank-aware file_max scaling + adaptive delta smoothing."""
# ── Cell 7g: Rank-aware scaling ────────────────────────────────────────
def rank_aware_scaling(probs, n_windows=12, power=0.4):
    """
    CHANGE 6: Scale each window by the file's single peak confidence.

    How it works:
      1. For each file, find the MAX score across all 12 windows (per species)
      2. Raise it to power → scale factor
      3. Multiply every window's score by that scale factor

    Example for one species across 12 windows:
      Confident file:  max=0.90 → scale=0.90^0.4=0.96 → mild boost
      Uncertain file:  max=0.10 → scale=0.10^0.4=0.40 → strong suppression

    How this differs from Change 3 (file_confidence_scale):
      Change 3 uses top-2 MEAN → smoother, less aggressive
      Change 6 uses single MAX  → asks "does ANY window have strong evidence?"

    power=0.0 → no effect (baseline)
    power=0.4 → moderate suppression of uncertain files (recommended start)
    power=1.0 → multiply directly by file max (very aggressive)
    """
    N, C = probs.shape
    assert N % n_windows == 0, f"Expected multiple of {n_windows}, got {N}"

    view     = probs.reshape(-1, n_windows, C)              # (n_files, 12, 234)
    file_max = view.max(axis=1, keepdims=True)              # (n_files, 1, 234)

    scale  = np.power(file_max, power)                      # (n_files, 1, 234)
    scaled = view * scale                                   # broadcast to all 12 windows

    return scaled.reshape(N, C)


print("✅ Rank-aware scaling defined")
# ── Cell 7h: Adaptive delta smoothing ─────────────────────────────────
def adaptive_delta_smooth(probs, n_windows=12, base_alpha=0.20):
    """
    CHANGE 7: Smooth uncertain windows toward their neighbors,
    while leaving confident windows almost untouched.

    How it works:
      For each window t:
        conf  = max probability across all 234 species at window t
        alpha = base_alpha * (1 - conf)   ← KEY: adapts to confidence
        new[t] = (1 - alpha) * old[t] + alpha * avg(old[t-1], old[t+1])

    Why alpha adapts to confidence:
      Confident window (max=0.90):
        alpha = 0.20 * (1 - 0.90) = 0.02  → barely smoothed, peak preserved
      Uncertain window (max=0.10):
        alpha = 0.20 * (1 - 0.10) = 0.18  → smoothed more, noise reduced

    This is exactly why your Change 1 hurt (-0.005) but this one should help:
      Change 1 used fixed alpha=0.3 → diluted confident peaks equally
      Change 7 uses adaptive alpha  → protects confident peaks, smooths noise

    base_alpha=0.0  → no smoothing (baseline)
    base_alpha=0.20 → recommended starting point
    """
    N, C = probs.shape
    assert N % n_windows == 0, f"Expected multiple of {n_windows}, got {N}"

    result = probs.copy()
    view   = probs.reshape(-1, n_windows, C)    # (n_files, 12, 234) original
    out    = result.reshape(-1, n_windows, C)   # (n_files, 12, 234) to modify

    for t in range(n_windows):

        # Confidence at this window = max prob across all species
        # Shape: (n_files, 1) — one confidence value per file per window
        conf = view[:, t, :].max(axis=-1, keepdims=True)   # (n_files, 1)

        # Adaptive alpha — low confidence = more smoothing
        alpha = base_alpha * (1.0 - conf)                  # (n_files, 1)

        # Neighbor average with edge padding
        if t == 0:
            # First window: left neighbor = itself
            neighbor_avg = (view[:, t, :] + view[:, t+1, :]) / 2.0
        elif t == n_windows - 1:
            # Last window: right neighbor = itself
            neighbor_avg = (view[:, t-1, :] + view[:, t, :]) / 2.0
        else:
            neighbor_avg = (view[:, t-1, :] + view[:, t+1, :]) / 2.0

        # Blend: confident windows barely change, uncertain ones smooth more
        out[:, t, :] = (1.0 - alpha) * view[:, t, :] + alpha * neighbor_avg

    return result


print("✅ Adaptive delta smoothing defined")