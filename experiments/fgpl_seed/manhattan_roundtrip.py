"""Strictly-Manhattan PanoPin -> FGPL pose round-trip (2026-07-16 demo).

Seeds FGPL from PanoPin's cached color scores for the 5-room Manhattan pool, runs FGPL's real
`multiroom_pose_estimation` on the Manhattan-only map, and scores the refined poses vs S3DIS GT.
Unlike D35 (whose 6-room map contained 2 non-Manhattan rooms), BOTH the candidate rooms and the
map are confined to Manhattan world.

Every pano is seeded (tau admits all) because the question is "estimated pose for each pano vs
GT"; the D34 tau=0.10 gate is reported as an offline slice of the same seed, not run separately
(the Voronoi depends on the seed set, so a gated arm is a different run, not a subset).

Prereq: `manhattan_map.py` (ply + line map + features). Fair (D5): the seed is built only from
cached color scores/poses, never GT — except the explicit `oracle` reference arm.

    conda run -n panopin python -m experiments.fgpl_seed.manhattan_roundtrip           # PanoPin arm
    conda run -n panopin python -m experiments.fgpl_seed.manhattan_roundtrip oracle    # GT-seeded ref
"""
import os
import sys
import json
import statistics

import numpy as np

_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, "..", "..", "src"))
sys.path.insert(0, os.path.join(_HERE, "..", ".."))

from experiments.fgpl_seed import (manhattan, seed_and_config as sc, run_arm, score, paths,
                                   roundtrip)
from eval import s3dis_gt
from panopin import robust_score, fgpl_export, coverage

SEEDS = paths.WORK / "seeds"
TAU_ADMIT_ALL = 1.0     # every pano seeded: the demo asks for a pose per pano
TAU_GATE = 0.10         # the D34 deployment gate, reported as an offline slice

# arm -> (seed kind, upright rotation prior). The prior is a property of the ESTIMATOR and the
# seed kind is a property of PANOPIN, so they are orthogonal: any fair comparison across seed
# kinds must hold the prior fixed (a table mixing prior-on and prior-off arms changes two
# variables at once).
ARM_SPEC = {
    "manhattan_global":            ("global", False),
    "manhattan_global_upright":    ("global", True),
    "manhattan_export":            ("export", False),
    "manhattan_upright":           ("export", True),
    "manhattan_oracle":            ("oracle", False),
    "manhattan_oracle_upright":    ("oracle", True),
    "manhattan_anchored":          ("anchored", False),
    "manhattan_anchored_upright":  ("anchored", True),
    "manhattan_aliased_solo":      ("solo", False),
}


def load_scores_poses(rows):
    """Cached low-pct scores + per-(pano,room) poses, confined to Manhattan panos x rooms."""
    grids = json.load(open(SEEDS / "largeval_residuals.json"))
    cache = json.load(open(SEEDS / "largeval_cache.json"))
    panos = [r["pano_name"] for r in rows]
    rooms = manhattan.POOL_ROOMS
    full = robust_score.low_percentile_scores(grids, q=20)
    scores = {p: {r: full[p][r] for r in rooms} for p in panos}
    poses = {p: {r: (cache[p]["poses"][r]["t"], cache[p]["poses"][r]["R"]) for r in rooms}
             for p in panos}
    return scores, poses


def build_seed(rows, md_path, arm, gt=None):
    """PanoPin arm: low-pct argmin seed for every pano. Oracle arm: GT position, true room."""
    scores, poses = load_scores_poses(rows)
    room_order = manhattan.POOL_ROOMS
    R_meta = np.array(json.load(open(md_path))["rotation_matrix"], float)
    kind = ARM_SPEC[arm][0]
    path = SEEDS / f"{arm}.json"
    if kind == "global":
        # Seed is written for schema completeness only — global mode never reads it.
        path = SEEDS / f"{arm}_seed_unused.json"
        admitted = fgpl_export.export_alignment(scores, poses, room_order, md_path, path,
                                                tau=TAU_ADMIT_ALL)
        return path, admitted, scores
    if kind == "solo":
        # Confirmation test: take panos that ALIASED in the 22-pano run despite a CORRECT
        # room seed, one per room, and re-run them with only 4 seeds so each gets ~its whole
        # room. Same seed positions as the 22-pano run -- the ONLY change is how many other
        # panos compete for the map. If they now lock, pano-count-per-room is causal.
        solo = {"conferenceRoom_1": "481b93c52f5144eb8557fbccf59262c9",
                "hallway_1": "d0834679c83f4ecf997c4c808d151482",
                "lounge_1": "80e1f6ae6c4b4e6dac81676ac86cacfb",
                "WC_1": "f0e54fcd44df46cea3ac3bd97eab0bef"}
        matches = []
        for room, pano in solo.items():
            t, _R = poses[pano][room]
            matches.append({"pano_name": pano, "room_idx": room_order.index(room),
                            "room_label": room, "score": float(scores[pano][room]),
                            "rotation_deg": 0.0,
                            "camera_position": fgpl_export.raw_t_to_camera_position(t, R_meta)})
        admitted = [m["pano_name"] for m in matches]
        fgpl_export.write_alignment_json(matches, admitted, path,
                                         extra_meta={"pipeline": "panopin:aliased-solo"})
        return path, admitted, scores
    if kind == "anchored":
        # The ACTUAL deployment config (D32/D33): one seed per room, room-anchored.
        # Also an unconfounded test of the geometry hypothesis -- 5 seeds means each
        # pano's Voronoi cell is ~its whole room, without touching FGPL's filter code.
        anchored = coverage.room_anchored_seeds(scores)      # {room: (pano, score)}
        matches = []
        for room, (pano, sc_) in anchored.items():
            t, _R = poses[pano][room]
            matches.append({"pano_name": pano, "room_idx": room_order.index(room),
                            "room_label": room, "score": float(sc_), "rotation_deg": 0.0,
                            "camera_position": fgpl_export.raw_t_to_camera_position(t, R_meta)})
        admitted = [m["pano_name"] for m in matches]
        assert len(set(admitted)) == len(admitted), "a pano anchored two rooms; not handled"
        fgpl_export.write_alignment_json(matches, admitted, path,
                                         extra_meta={"pipeline": "panopin:room-anchored"})
        return path, admitted, scores
    if kind == "oracle":
        matches = [{"pano_name": r["pano_name"], "room_idx": room_order.index(r["room"]),
                    "room_label": r["room"], "score": 0.0, "rotation_deg": 0.0,
                    "camera_position": [float(x) for x in gt[r["pano_name"]]["location"][:2]]}
                   for r in rows]
        admitted = [m["pano_name"] for m in matches]
        fgpl_export.write_alignment_json(matches, admitted, path,
                                         extra_meta={"pipeline": "gt-oracle:manhattan"})
        return path, admitted, scores
    admitted = fgpl_export.export_alignment(scores, poses, room_order, md_path, path,
                                            tau=TAU_ADMIT_ALL)
    return path, admitted, scores


def main(arm="manhattan_export"):
    rows = manhattan.build_pool()
    gt = s3dis_gt.load_gt("Area_3", s3dis_gt.load_config(None))
    cents = sc.room_centroids(rows)
    md = sc.write_identity_metadata()
    ply = paths.WORK / "clouds" / f"{manhattan.SCENE}.ply"
    line_map = paths.WORK / manhattan.LINEMAP_SUBDIR / "3d_line_map.pkl"
    assert line_map.exists(), f"run manhattan_map first: {line_map}"
    all_rooms = sorted({r["room"] for r in rows})

    seed_path, admitted, scores = build_seed(rows, md, arm, gt)
    admitted_rows = [r for r in rows if r["pano_name"] in admitted]
    seed_room = {m["pano_name"]: m["room_label"]
                 for m in json.load(open(seed_path))["matches"]}
    print(f"[{arm}] {len(admitted)} seeds; rooms {all_rooms}", flush=True)

    # manhattan_global = FGPL's ORIGINAL global mode: no Voronoi, no seed read at all
    # (load_panorama_positions is only called under use_local). This is the FGPL-alone
    # baseline AND the test of whether Voronoi local filtering starves panos of 3D lines.
    kind, upright = ARM_SPEC[arm]
    use_local = kind != "global"
    extra = {"point_cloud_name": manhattan.SCENE, "point_cloud_path": str(ply)}
    if upright:
        # S3DIS raw frame is Z-up, so gravity is [0,0,1] in the cloud frame.
        extra.update({"upright_prior": True, "up_world": [0.0, 0.0, 1.0],
                      "max_tilt_deg": 10.0})
    cfg = sc.write_config(arm, admitted_rows, seed_path, line_map, md,
                          paths.WORK / "features", paths.WORK / "panos",
                          use_local=use_local, extra=extra)
    poses = run_arm.run_arm(cfg, admitted_rows)
    poses_cw, s = roundtrip._score_cw(poses, gt, admitted_rows, cents)
    n_cov, n_rooms, covered = roundtrip.per_room_coverage(poses_cw, admitted_rows, all_rooms,
                                                          cents)

    pu = s["translation"]["per_uuid"]
    right = [r["pano_name"] for r in admitted_rows if seed_room.get(r["pano_name"]) == r["room"]]
    rr = [pu[u] for u in right if u in pu]

    res = {"arm": arm, "scored": s, "n_admitted": len(admitted_rows),
           "coverage": {"n_covered": n_cov, "n_rooms": n_rooms, "covered": covered},
           "trans_median_right_room": statistics.median(rr) if rr else None,
           "n_right_room": len(rr),
           "seed_room": seed_room,
           "true_room": {r["pano_name"]: r["room"] for r in admitted_rows},
           "per_uuid_trans": pu,
           "gate_would_admit": sorted(p for p in scores
                                      if min(scores[p].values()) <= TAU_GATE)}
    out = paths.subdir("results") / f"{arm}.json"
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    tm = s["translation"]["median"]
    print(f"\n[{arm}] localized {s['n_localized']}/{len(admitted_rows)}  coverage {n_cov}/{n_rooms}"
          f"  trans_med={'n/a' if tm is None else round(tm, 3)}"
          f"  right-room_med={res['trans_median_right_room']}"
          f"  wrong_room={s['wrong_room_rate']}\nwrote {out}", flush=True)
    return res


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "manhattan_export")
