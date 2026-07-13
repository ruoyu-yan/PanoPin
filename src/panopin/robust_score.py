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
