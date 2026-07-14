"""Re-score CPO room candidates by a ROBUST statistic of their per-point color
residuals instead of the mean (D26 follow-up). A residual grid is 101 percentiles
(p0..p100) of one (pano, room) residual vector. Fair: reads only residuals, no GT (D5)."""

def _stat(grid, stat, **params):
    if len(grid) != 101:
        raise ValueError(f"grid must be 101 percentiles, got {len(grid)}")
    if stat == "median":
        return float(grid[50])
    if stat == "low_percentile":
        q = int(params["q"])
        return float(grid[q])
    if stat == "trimmed_mean":
        k = int(params["k"])                      # drop the top-k% highest residuals
        kept = grid[:101 - k] if k > 0 else grid
        return float(sum(kept) / len(kept))
    raise ValueError(f"unknown stat {stat!r}")

def robust_scores(grids, stat, **params):
    """grids: {pano: {room: [101 floats]}} -> {pano: {room: score}} (lower = better)."""
    return {p: {r: _stat(g, stat, **params) for r, g in rooms.items()}
            for p, rooms in grids.items()}


def raw_mean_scores(grids):
    """RAW UNWEIGHTED per-room residual mean (== mean of the residual grid, i.e. trimmed_mean
    k=0). Beats CPO's deployed match_color+weighted loss on the D27/D28 gate metric. Superseded
    for room assignment by `low_percentile_scores` (D30): the mean lets a loss-sink room's low
    core drag its score down via the good-point majority, so window/occlusion panos flip to the
    sink. Kept for the D28 reproduction. Fair: reads only residuals."""
    return robust_scores(grids, "trimmed_mean", k=0)


DEPLOY_Q = 20   # low-percentile pin: stable plateau q in [5,25] (deploy_regime_qpin, D30)

def low_percentile_scores(grids, q=DEPLOY_Q):
    """RECOMMENDED deployable room-assignment score (D30): the q-th percentile of a room's
    per-point residuals -- "how well do the best-matching q% of points align". The true room
    has a strong low core (many well-matched points); a loss-sink room matches mediocrely
    everywhere, so its best q% is still worse. This suppresses the loss-sink tail that raw-mean
    lets the sink exploit.

    Validated OFFLINE on TWO independent pose caches (deploy_regime / _xcheck / _qpin):
    dominates raw-mean and CPO-loss at every candidate-set size k in {3..6, 23}; the advantage
    GROWS with room count (loss-sink risk); a WIDE stable plateau over q in [5,25] (not an
    n=12 overfit). Unlike calibrate.assign it needs NO cross-pano stats -- it scores each pano
    INDEPENDENTLY, so it fits the fast per-pano coarse-seed mandate and works with few panos.
    Feed straight to argmin (calibration adds ~nothing on top -- deploy_regime_gate, Q4).
    Fair: reads only residuals, no GT (D5). Caveat: n=12; a larger-pano GPU run should confirm."""
    return robust_scores(grids, "low_percentile", q=q)
