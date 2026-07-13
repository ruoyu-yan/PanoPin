"""Run the ablation arms on the subset, score each vs GT, emit a comparison table.
Arms: oracle / p1 / p1_cal / wrong_room.
  - oracle     : GT position seed (upper bound).
  - p1         : CPO raw min-loss room's position seed.
  - p1_cal     : CPO v1-minmax-calibrated room's position seed (D24) -- the new lever.
  - wrong_room : sibling-room centroid (catastrophic floor).
P2 (translation-grid narrowing) and P3 (CPO-yaw prior) were dropped after D25: P2 was a
no-op (Voronoi already constrains the search) and P3 hurt (CPO rotation is convention-
broken). The flag-gated FGPL narrowing edit still lives on scan2measure
feat/panopin-seed-narrowing / patches/fgpl_narrowing.patch if ever needed.

Rotation scoring: FGPL emits rotation in the equirect signed-perm convention
C=[[0,0,1],[-1,0,0],[0,-1,0]] (validated: Rp.T @ C == camera->world, ~0.5 deg on
precisely-localized oracle panos). We convert to camera->world before scoring so the
rotation numbers are comparable to GT R_cw."""
import json, time, statistics, numpy as np
from experiments.fgpl_seed import (subset, seed_and_config as sc, run_arm, score, paths)
from eval import s3dis_gt

C = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float)   # FGPL equirect signed-perm
ARMS = ["oracle", "p1", "p1_cal", "wrong_room"]


def _fgpl_rot_to_cw(R):
    """FGPL output rotation (C @ R_wc) -> camera->world, for scoring vs GT R_cw."""
    return (np.array(R).T @ C).tolist()


def _seed_room(arm, name, cpo):
    """The room whose CPO pose seeded this arm (== 'correct region' when it matches GT).
    p1 uses raw min-loss; p1_cal uses the calibrated room."""
    return cpo[name]["room_cal"] if arm == "p1_cal" else cpo[name]["room"]


def main():
    rows = subset.build_subset()
    gt = s3dis_gt.load_gt("Area_3", s3dis_gt.load_config(None))
    cpo = json.load(open(paths.WORK / "seeds" / "cpo_cache.json"))
    cents = sc.room_centroids(rows)
    md = sc.write_identity_metadata()
    line_map = paths.WORK / "linemap" / "3d_line_map.pkl"
    feat, panos = paths.WORK / "features", paths.WORK / "panos"

    results = {}
    for arm in ARMS:
        seed = sc.write_seed(arm, rows, gt, cpo, centroids=cents)
        cfg = sc.write_config(arm, rows, seed, line_map, md, feat, panos)
        t0 = time.time()
        poses = run_arm.run_arm(cfg, rows)
        dt = time.time() - t0
        poses_cw = {u: (None if p is None else
                        {"translation": p["translation"], "rotation": _fgpl_rot_to_cw(p["rotation"])})
                    for u, p in poses.items()}
        s = score.score_arm(poses_cw, gt, rows, cents)
        s["runtime_s"] = dt
        results[arm] = s
        print(f"[{arm}] trans_med={s['translation']['median']:.3f} "
              f"rot_med={s['rotation']['median']:.1f} wrong={s['wrong_room_rate']} {dt:.0f}s",
              flush=True)

    # Headline slice: median translation over the panos where the ARM's SEED picked the
    # TRUE room (== region-selection correct). This is the load-bearing number: when the
    # room is right, the color position seed should be ~= oracle. p1 uses raw min-loss,
    # p1_cal uses the calibrated room -- so the two slices cover different pano sets.
    for arm in ARMS:
        if arm in ("oracle", "wrong_room"):
            results[arm]["trans_median_right_room"] = None
            results[arm]["n_right_room"] = None
            continue
        right = [r["pano_name"] for r in rows if _seed_room(arm, r["pano_name"], cpo) == r["room"]]
        pu = results[arm]["translation"]["per_uuid"]
        rr = [pu[u] for u in right if u in pu]
        results[arm]["trans_median_right_room"] = statistics.median(rr) if rr else None
        results[arm]["n_right_room"] = len(rr)

    raw_recall = sum(1 for r in rows if cpo[r["pano_name"]]["room"] == r["room"])
    cal_recall = sum(1 for r in rows if cpo[r["pano_name"]]["room_cal"] == r["room"])

    out = paths.subdir("results") / "ablation.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)

    def _f(x, p=3):
        return f"{x:.{p}f}" if x is not None else "n/a"
    n = len(rows)
    print(f"\nroom recall@1: raw min-loss (p1) = {raw_recall}/{n}   calibrated (p1_cal) = {cal_recall}/{n}")
    print("\n| arm | n_loc | trans median | trans median (right-room) | trans mean | trans max | rot median | wrong-room | runtime s |")
    print("|-----|-------|--------------|---------------------------|------------|-----------|-----------|-----------|-----------|")
    for arm in ARMS:
        s = results[arm]
        t, r = s["translation"], s["rotation"]
        nrr = s["n_right_room"]
        rr_str = f"{_f(s['trans_median_right_room'])} ({nrr})" if nrr is not None else "n/a"
        print(f"| {arm} | {s['n_localized']}/{n} | {_f(t['median'])} | "
              f"{rr_str} | {_f(t['mean'])} | {_f(t['max'])} | {_f(r['median'],1)} | "
              f"{_f(s['wrong_room_rate'],2)} | {_f(s['runtime_s'],0)} |")
    print("\nwrote", out)


if __name__ == "__main__":
    main()
