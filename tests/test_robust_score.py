import numpy as np
from panopin import robust_score as rs

# grid = percentiles p0..p100 of a residual vector
def _grid(vals):
    return np.percentile(np.asarray(vals, float), np.arange(101)).tolist()

def test_stat_values():
    g = _grid(list(range(101)))          # residuals 0..100 -> percentiles ~ 0..100
    grids = {"pA": {"r": g}}
    assert abs(rs.robust_scores(grids, "median")["pA"]["r"] - 50.0) < 1e-6
    assert abs(rs.robust_scores(grids, "low_percentile", q=10)["pA"]["r"] - 10.0) < 1e-6
    # trimmed_mean k=0 == mean of full grid (~50); k=50 drops top half -> mean of p0..p50 (~25)
    assert abs(rs.robust_scores(grids, "trimmed_mean", k=0)["pA"]["r"] - 50.0) < 1.0
    assert abs(rs.robust_scores(grids, "trimmed_mean", k=50)["pA"]["r"] - 25.0) < 1.0

def test_raw_mean_scores_is_grid_mean():
    g = _grid([0.05] * 80 + [0.8] * 20)
    grids = {"p": {"r": g}}
    s = rs.raw_mean_scores(grids)["p"]["r"]
    assert abs(s - rs.robust_scores(grids, "trimmed_mean", k=0)["p"]["r"]) < 1e-9
    assert abs(s - (sum(g) / len(g))) < 1e-9   # == mean of the 101-pt grid

def test_low_percentile_scores_default_q_and_supersedes_mean():
    # true room: strong low core + window outliers; loss-sink: mediocre everywhere.
    true = _grid([0.05] * 80 + [0.8] * 20)
    sink = _grid([0.30] * 100)
    grids = {"p": {"true": true, "sink": sink}}
    # default pin q=DEPLOY_Q must equal an explicit low_percentile at that q.
    lp = rs.low_percentile_scores(grids)["p"]
    assert lp == rs.robust_scores(grids, "low_percentile", q=rs.DEPLOY_Q)["p"]
    # low-pct assigns the true room; raw-mean flips to the loss-sink (mean 0.20 core vs 0.30 hidden
    # by the 0.8 window tail -> true mean ~0.20+ vs sink 0.30, but the window mass narrows it).
    assert min(lp, key=lambda r: lp[r]) == "true"


def test_discrimination_true_room_beats_loss_sink():
    # true room: strong low core (most points match ~0.05) + minority window outliers (~0.8)
    true = _grid([0.05]*80 + [0.8]*20)
    # loss-sink: mediocre everywhere (~0.30) -> higher low-core, similar mean
    sink = _grid([0.30]*100)
    grids = {"p": {"true": true, "sink": sink}}
    # under low_percentile the true room's core wins; under plain mean it would NOT (0.20 vs 0.30 is close,
    # and window-heavy panos can flip it) -- this is the whole hypothesis.
    lp = rs.robust_scores(grids, "low_percentile", q=20)["p"]
    assert lp["true"] < lp["sink"]
    tm = rs.robust_scores(grids, "trimmed_mean", k=30)["p"]   # drop the window outliers
    assert tm["true"] < tm["sink"]
