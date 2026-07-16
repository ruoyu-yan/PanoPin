"""Per-room architecture: PanoPin assigns the room, FGPL localizes inside that room alone.

The 22-seed arm starves each pano: the Voronoi splits a room's 3D lines among the panos sharing
it, and the rotation search cannot then tell the true rotation from its 180° twin. The 5-seed
anchored arm avoids that (each pano gets ~a whole room) but only poses 5 panos.

This gets both. For each room: build a line map from THAT ROOM's cloud only, then run the
estimator in GLOBAL mode (use_local_filtering=False) over the panos PanoPin assigned to it.
Global mode on a single-room map means no Voronoi at all — every pano sees the entire room,
however many panos share it. That is also the configuration the TMB 180° fix was validated in
(the single-room pipeline).

PanoPin's ONLY contribution here is the room assignment; no seed positions are used (global mode
never reads the alignment file). Assignment comes from PanoPin's low-percentile argmin, NOT from
ground truth — so its ~4/22 misassignments localize against the WRONG room's map and are scored
as the failures they are.

    conda run -n panopin python -m experiments.fgpl_seed.manhattan_perroom
"""
import os
import sys
import json
import math
import time
import statistics as st

_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, "..", "..", "src"))
sys.path.insert(0, os.path.join(_HERE, "..", ".."))

from experiments.fgpl_seed import (manhattan, manhattan_roundtrip as mr, build_ply, build_linemap,
                                   seed_and_config as sc, run_arm, roundtrip, score, paths)
from eval import s3dis_gt, metrics

ARM = "manhattan_perroom"


def panopin_assignment(rows):
    """{pano: room} from PanoPin's low-pct argmin. No GT (D5)."""
    scores, _poses = mr.load_scores_poses(rows)
    return {p: min(scores[p], key=lambda r: scores[p][r]) for p in scores}


def room_map(room, cloud_txt):
    """Line map built from ONE room's cloud. Cached: skip if already built."""
    scene = f"area3_{room}"
    sub = f"linemap_{room}"
    lm = paths.WORK / sub / "3d_line_map.pkl"
    ply = paths.WORK / "clouds" / f"{scene}.ply"
    if lm.exists() and ply.exists():
        print(f"  [{room}] map cached", flush=True)
        return ply, lm
    ply = build_ply.build_combined_ply([{"cloud_txt": cloud_txt}], scene=scene)
    lm = build_linemap.build_linemap(ply, scene=scene, out_subdir=sub)
    return ply, lm


def main():
    t0 = time.time()
    rows = manhattan.build_pool()
    gt = s3dis_gt.load_gt("Area_3", s3dis_gt.load_config(None))
    cents = sc.room_centroids(rows)
    md = sc.write_identity_metadata()
    clouds = manhattan.pool_clouds()
    assign = panopin_assignment(rows)
    true = {r["pano_name"]: r["room"] for r in rows}

    n_bad = sum(1 for p in assign if assign[p] != true[p])
    print(f"[perroom] PanoPin assignment: {len(assign)-n_bad}/{len(assign)} correct "
          f"({n_bad} panos will be localized against the WRONG room's map)", flush=True)

    poses = {}
    for room in manhattan.POOL_ROOMS:
        mine = [r for r in rows if assign[r["pano_name"]] == room]
        if not mine:
            print(f"  [{room}] no panos assigned; skipping", flush=True)
            continue
        print(f"\n[perroom] {room}: {len(mine)} panos assigned "
              f"({(time.time()-t0)/60:.1f} min)", flush=True)
        ply, lm = room_map(room, clouds[room])
        cfg = sc.write_config(f"{ARM}_{room}", mine, paths.WORK / "seeds" / "manhattan_export.json",
                              lm, md, paths.WORK / "features", paths.WORK / "panos",
                              use_local=False,            # single-room map -> no Voronoi
                              extra={"point_cloud_name": f"area3_{room}",
                                     "point_cloud_path": str(ply),
                                     "output_dir": str(paths.WORK / "poses" / ARM / room),
                                     "upright_prior": True, "up_world": [0.0, 0.0, 1.0],
                                     "max_tilt_deg": 10.0})
        got = run_arm.run_arm(cfg, mine)
        poses.update({u: v for u, v in got.items()})

    # --- score every pano against GT ---
    all_rows = [r for r in rows if r["pano_name"] in poses and poses[r["pano_name"]]]
    poses_cw, s = roundtrip._score_cw({u: poses[u] for u in poses}, gt, rows, cents)
    n_cov, n_rooms, covered = roundtrip.per_room_coverage(poses_cw, rows, sorted(clouds), cents)

    per = {}
    for r in rows:
        u = r["pano_name"]
        p = poses.get(u)
        if not p:
            continue
        eR = roundtrip._fgpl_rot_to_cw(p["rotation"])
        per[u] = {"dt": math.dist(p["translation"], gt[u]["location"]),
                  "dr": metrics.rotation_errors({u: eR}, {u: gt[u]["R_cw"]})["per_uuid"][u],
                  "assigned": assign[u], "true": true[u]}

    res = {"arm": ARM, "scored": s, "n_admitted": len(rows),
           "coverage": {"n_covered": n_cov, "n_rooms": n_rooms, "covered": covered},
           "assignment_correct": len(assign) - n_bad, "per_pano": per,
           "true_room": true, "seed_room": assign,
           "per_uuid_trans": {u: v["dt"] for u, v in per.items()},
           "trans_median_right_room": None, "n_right_room": 0}
    right = [v["dt"] for v in per.values() if v["assigned"] == v["true"]]
    res["trans_median_right_room"] = st.median(right) if right else None
    res["n_right_room"] = len(right)
    out = paths.subdir("results") / f"{ARM}.json"
    with open(out, "w") as f:
        json.dump(res, f, indent=2)

    dts = [v["dt"] for v in per.values()]
    drs = [v["dr"] for v in per.values()]
    print(f"\n[{ARM}] localized {len(per)}/{len(rows)}  coverage {n_cov}/{n_rooms}  "
          f"trans_med={st.median(dts):.3f}  rot_med={st.median(drs):.1f}  "
          f"locked={sum(1 for x in drs if x <= 45)}/{len(drs)}  "
          f"wrong_room={s['wrong_room_rate']:.2f}  ({(time.time()-t0)/60:.1f} min)\nwrote {out}",
          flush=True)


if __name__ == "__main__":
    main()
