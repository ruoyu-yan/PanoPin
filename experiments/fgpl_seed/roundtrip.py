# experiments/fgpl_seed/roundtrip.py
"""PanoPin<->FGPL validation round-trip (spec 2026-07-15-fgpl-roundtrip). Seed FGPL from the
D34 fgpl_export module, run the estimator on the 6-room area3_seed_ablation subset, score
refined poses vs S3DIS GT + per-room coverage. Reference: cached oracle/p1 (work/poses/).

Offline pieces (seed build, pre-check, coverage) run in `panopin`; run_arm internally spawns
the FGPL estimator in `panopin-gpu`. Run the full round-trip:
    conda run -n panopin python -m experiments.fgpl_seed.roundtrip
Fair (D5): the seed is built only from cached color scores/poses, never GT."""
import os
import json
import sys
import numpy as np

# panopin lives at src/ (not pip-installed); mirror the sibling-script path setup
# (cpo_seeds.py et al. / docs/HANDOVER.md "panopin package is not pip-installed").
_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, "..", "..", "src"))   # panopin.*
sys.path.insert(0, os.path.join(_HERE, "..", ".."))          # experiments.* (repo root)

from experiments.fgpl_seed import subset, seed_and_config as sc, run_arm, score, paths
from eval import s3dis_gt
from panopin import robust_score, fgpl_export

C = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float)   # FGPL equirect signed-perm
SEEDS = paths.WORK / "seeds"


def _fgpl_rot_to_cw(R):
    """FGPL output rotation (C @ R_wc) -> camera->world, for scoring vs GT R_cw."""
    return (np.array(R).T @ C).tolist()


def load_scores_poses():
    """Cached low-pct score matrix + per-(pano,room) (t,R) poses for the subset (no GT)."""
    grids = json.load(open(SEEDS / "residuals.json"))
    scores = robust_score.low_percentile_scores(grids)
    cpo = json.load(open(SEEDS / "cpo_cache.json"))
    poses = {p: {r: (v["t"], v["R"]) for r, v in cpo[p]["poses"].items()} for p in cpo}
    return scores, poses


def build_export_seed(rows, md_path, tau=0.10):
    """Deployment seed via the D34 module -> (seed_path, admitted pano names)."""
    scores, poses = load_scores_poses()
    room_order = sorted({r["room"] for r in rows})
    seed_path = SEEDS / "fgpl_export.json"
    admitted = fgpl_export.export_alignment(scores, poses, room_order, md_path, seed_path, tau=tau)
    return seed_path, admitted


def precheck_seed(seed_path, md_path, admitted, rows):
    """GPU-free: load the seed through FGPL's OWN load_panorama_positions; assert positions
    recover to the cached t[:2] and every subset room is seeded. Returns (n_seed, rooms)."""
    sys.path.insert(0, str(paths.FGPL_ROOT / "src" / "pose_estimation"))
    from multiroom_pose_estimation import load_panorama_positions
    positions = load_panorama_positions(str(seed_path), str(md_path), list(admitted))
    cpo = json.load(open(SEEDS / "cpo_cache.json"))
    matches = json.load(open(seed_path))["matches"]
    for m in matches:
        exp = np.array(cpo[m["pano_name"]]["poses"][m["room_label"]]["t"])[:2]
        assert np.allclose(positions[m["pano_name"]], exp, atol=1e-6), m["pano_name"]
    seeded_rooms = {m["room_label"] for m in matches}
    all_rooms = {r["room"] for r in rows}
    assert seeded_rooms == all_rooms, (sorted(seeded_rooms), sorted(all_rooms))
    return len(matches), sorted(all_rooms)


def per_room_coverage(poses_cw, admitted_rows, all_rooms, cents):
    """Rooms whose refined FGPL pose lands in the correct room. A room r is covered iff >=1
    admitted pano whose TRUE room is r has a pose whose nearest centroid == r. Denominator =
    all_rooms (the full subset room set). Returns (n_covered, n_rooms, {room: bool})."""
    true_room = {r["pano_name"]: r["room"] for r in admitted_rows}
    covered = {r: False for r in all_rooms}
    for u, p in poses_cw.items():
        if p is None:
            continue
        tr = true_room[u]
        if score._nearest_room(p["translation"], cents) == tr:
            covered[tr] = True
    return sum(covered.values()), len(all_rooms), covered


def load_cached_arm(arm, rows):
    """Load a previously-run arm's per-pano FGPL poses from work/poses/<arm>/ (no re-run)."""
    base = paths.WORK / "poses" / arm
    out = {}
    for r in rows:
        cp = base / r["pano_name"] / "camera_pose.json"
        if cp.exists():
            d = json.load(open(cp))
            out[r["pano_name"]] = {"translation": d["translation"], "rotation": d["rotation"]}
        else:
            out[r["pano_name"]] = None
    return out
