"""Deployable PanoPin -> FGPL room seeding (D32/D33).

Given a scene's panoramas + candidate room clouds, return ONE seed (pano + coarse pose) per
room via room-anchored low-percentile colour scoring: each room is seeded by its best-matching
pano (`coverage.room_anchored_seeds`), which is loss-sink-immune and, in the all-covered
deployment, covers every room at high precision (COVERAGE_RESULTS.md / D32). Composes the
validated primitives — cpo_adapter (localize + per-point residuals) + robust_score (low-pct,
D30) + coverage (room-anchored, D32). Fair: reads only panos + clouds, no GT (D5).

Split so the pure assembly (`seeds_from_scores`) is testable without GPU; only
`localize_and_score` touches CPO. Default cfg matches the validated caches (load_cfg
sample_rate=30, raw residual path)."""
from collections import namedtuple
import numpy as np

from panopin.cpo_config import load_cfg
from panopin.cpo_adapter import localize_pair, residuals_at_pose
from panopin import robust_score, coverage
from panopin.determinism import pin

RoomSeed = namedtuple("RoomSeed", "room pano t R score")
SeedResult = namedtuple("SeedResult", "seeds confidence scores")


def _grid(resid):
    """101-percentile summary of a per-point residual vector (empty -> degenerate high)."""
    if len(resid) == 0:
        return [999.0] * 101
    return np.percentile(resid, np.arange(101)).astype(float).tolist()


def localize_and_score(panos, candidate_clouds, cfg, q=robust_score.DEPLOY_Q):
    """GPU. panos {pano_id: pano_path} x candidate_clouds {room: cloud_path} ->
    (score_matrix {pano:{room: low-pct score}}, poses {pano:{room: (t, R)}}).

    Pins determinism.pin() first: this is the single chokepoint every shipped
    caller passes through (panopin.cli.seed_from_clouds calls this directly;
    seed_rooms below calls it too), and data_utils.read_txt_pcd draws an
    np.random permutation on every call whenever cfg.sample_rate>1 (the deployed
    sample_rate=30 always hits this). Without pinning here, two runs over the
    same inputs can sample different point subsets and disagree on scores/poses
    (see tests/test_determinism.py::test_localize_and_score_is_reproducible_across_runs).
    One pin() here is sufficient for the whole loop below: residuals_at_pose
    already reseeds np.random to a fixed value before its own read on every
    iteration, so it re-anchors the RNG state after the first pair regardless."""
    pin()
    grids, poses = {}, {}
    for pid, ppath in panos.items():
        grids[pid], poses[pid] = {}, {}
        for room, cloud in candidate_clouds.items():
            t, R, _ = localize_pair(cfg, ppath, cloud)
            resid = residuals_at_pose(cfg, ppath, cloud, t, R)
            grids[pid][room] = _grid(resid)
            poses[pid][room] = (t, R)
    return robust_score.low_percentile_scores(grids, q=q), poses


def seeds_from_scores(score_matrix, poses):
    """Pure (no GPU): room-anchored assembly -> {room: RoomSeed(room, pano, t, R, score)}."""
    out = {}
    for room, (pano, score) in coverage.room_anchored_seeds(score_matrix).items():
        t, R = poses[pano][room]
        out[room] = RoomSeed(room, pano, t, R, score)
    return out


def seed_rooms(panos, candidate_clouds, cfg=None, q=robust_score.DEPLOY_Q):
    """The deployable hand-off. Returns SeedResult(seeds, confidence, scores):
      seeds      {room: RoomSeed(room, pano, t, R, score)} -- one coarse seed/room for FGPL
      confidence {pano: (picked_room, confidence)} -- for optional gating (higher = more confident)
      scores     {pano: {room: low-pct score}} -- the underlying score matrix
    """
    if cfg is None:
        cfg = load_cfg(sample_rate=30)
    score_matrix, poses = localize_and_score(panos, candidate_clouds, cfg, q=q)
    return SeedResult(seeds_from_scores(score_matrix, poses),
                      coverage.pano_confidence(score_matrix), score_matrix)
