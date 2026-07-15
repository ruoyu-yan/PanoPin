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


def _score_cw(poses, gt, rows, cents):
    poses_cw = {u: (None if p is None else
                    {"translation": p["translation"], "rotation": _fgpl_rot_to_cw(p["rotation"])})
                for u, p in poses.items()}
    return poses_cw, score.score_arm(poses_cw, gt, rows, cents)


def main():
    rows = subset.build_subset()
    gt = s3dis_gt.load_gt("Area_3", s3dis_gt.load_config(None))
    cents = sc.room_centroids(rows)
    md = sc.write_identity_metadata()
    line_map = paths.WORK / "linemap" / "3d_line_map.pkl"
    feat, panos = paths.WORK / "features", paths.WORK / "panos"
    all_rooms = sorted({r["room"] for r in rows})

    # --- build + pre-check the fgpl_export seed (offline) ---
    seed_path, admitted = build_export_seed(rows, md)
    n_seed, _ = precheck_seed(seed_path, md, admitted, rows)
    admitted_rows = [r for r in rows if r["pano_name"] in admitted]
    seed_room = {m["pano_name"]: m["room_label"]
                 for m in json.load(open(seed_path))["matches"]}
    print(f"[precheck] {n_seed} seeds admitted; rooms {all_rooms}", flush=True)

    # --- run FGPL on the admitted panos (the expensive step) ---
    cfg = sc.write_config("fgpl_export", admitted_rows, seed_path, line_map, md, feat, panos)
    poses = run_arm.run_arm(cfg, admitted_rows)
    poses_cw, s = _score_cw(poses, gt, admitted_rows, cents)
    n_cov, n_rooms, covered = per_room_coverage(poses_cw, admitted_rows, all_rooms, cents)

    # right-room slice: median trans over admitted panos whose SEED room == true room
    pu = s["translation"]["per_uuid"]
    right = [r["pano_name"] for r in admitted_rows
             if seed_room.get(r["pano_name"]) == r["room"]]
    import statistics
    rr = [pu[u] for u in right if u in pu]
    trans_right = statistics.median(rr) if rr else None

    # --- cached reference arms, scored on the SAME admitted set ---
    refs = {}
    for arm in ("oracle", "p1"):
        _, rs = _score_cw(load_cached_arm(arm, admitted_rows), gt, admitted_rows, cents)
        refs[arm] = rs

    results = {"fgpl_export": s, "coverage": {"n_covered": n_cov, "n_rooms": n_rooms,
               "covered": covered}, "trans_median_right_room": trans_right,
               "n_right_room": len(rr), "n_admitted": len(admitted_rows),
               "references": refs}
    out = paths.subdir("results") / "roundtrip.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)

    _write_report(results, all_rooms)
    tm = s["translation"]["median"]
    print(f"\n[roundtrip] coverage {n_cov}/{n_rooms}  "
          f"trans_med={'n/a' if tm is None else round(tm, 3)}  "
          f"right-room_med={trans_right if trans_right is None else round(trans_right, 3)}  "
          f"wrong_room={s['wrong_room_rate']}\nwrote {out}", flush=True)


def _write_report(res, all_rooms):
    s = res["fgpl_export"]; t, r = s["translation"], s["rotation"]
    cov = res["coverage"]
    def f(x, p=3):
        return "n/a" if x is None else f"{x:.{p}f}"
    per_room = ", ".join(f"{r}={'OK' if v else 'MISS'}" for r, v in cov["covered"].items())
    lines = [
        "# PanoPin<->FGPL validation round-trip results",
        "",
        f"Scene: area3_seed_ablation (6-room subset, 12 in-frame panos). "
        f"fgpl_export arm gated at tau=0.10; FGPL estimator run on the {res['n_admitted']} "
        f"admitted panos. Fair: seed from cached color scores only (no GT).",
        "",
        f"**Per-room coverage: {cov['n_covered']}/{cov['n_rooms']}** "
        f"(a room counts iff >=1 admitted pano of that TRUE room refines into it). "
        f"Per room: {per_room}.",
        "",
        "| arm | n_localized | trans median | trans median (right-room) | trans mean | trans max | rot median | wrong-room |",
        "|-----|-------------|--------------|---------------------------|------------|-----------|-----------|-----------|",
        f"| fgpl_export (gated) | {s['n_localized']}/{res['n_admitted']} | {f(t['median'])} | "
        f"{f(res['trans_median_right_room'])} ({res['n_right_room']}) | {f(t['mean'])} | "
        f"{f(t['max'])} | {f(r['median'],1)} | {f(s['wrong_room_rate'],2)} |",
    ]
    for arm in ("oracle", "p1"):
        rs = res["references"][arm]; rt, rr2 = rs["translation"], rs["rotation"]
        lines.append(
            f"| {arm} (cached, ref) | {rs['n_localized']}/{res['n_admitted']} | {f(rt['median'])} "
            f"| n/a | {f(rt['mean'])} | {f(rt['max'])} | {f(rr2['median'],1)} | "
            f"{f(rs['wrong_room_rate'],2)} |")
    lines += [
        "",
        "Caveats: oracle/p1 cached poses ran with a 12-pano Voronoi vs this arm's "
        f"{res['n_admitted']}-pano Voronoi (reference context, not a controlled ablation); "
        "Area_3 subset only; tau=0.10 is the D34 Area_3-tuned value (this run also serves as "
        "its through-FGPL precision check).",
    ]
    (paths.HERE / "ROUNDTRIP_RESULTS.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
