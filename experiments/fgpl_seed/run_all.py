"""Run all 5 arms on the subset, score each vs GT, emit a comparison table.
Arms: oracle / p1 / p2 / p3 / wrong_room. P1/P2/P3 read the CPO cache; P2 adds a
translation radius; P3 adds P2 + a yaw prior derived from CPO's rotation, mapped into
FGPL's rotation convention.

Rotation scoring: FGPL emits rotation in the equirect signed-perm convention
C=[[0,0,1],[-1,0,0],[0,-1,0]] (validated: Rp.T @ C == camera->world, ~0.5 deg on
precisely-localized oracle panos). We convert to camera->world before scoring so the
rotation numbers are comparable to GT R_cw.

CAVEAT (P3): the estimator runs all panos in one process with a single seed_yaw_deg,
so P3 uses ONE global (median) yaw prior. Panos whose true yaw is >tol from it fall
back to the full rotation set (== P2). Per-pano yaw would need per-pano runs; out of
scope for this first cut. Headline metrics = translation + wrong-room."""
import json, time, math, numpy as np
from experiments.fgpl_seed import (subset, seed_and_config as sc, run_arm, score, paths)
from eval import s3dis_gt

P2_RADIUS = 2.0            # meters (design §7)
P3_YAW_TOL = 30.0          # degrees
C = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float)   # FGPL equirect signed-perm
ARMS = ["oracle", "p1", "p2", "p3", "wrong_room"]


def _fgpl_rot_to_cw(R):
    """FGPL output rotation (C @ R_wc) -> camera->world, for scoring vs GT R_cw."""
    return (np.array(R).T @ C).tolist()


def _fgpl_yaw_from_cpo_R(cpo_R):
    """CPO rotation (camera->world, S3DIS frame) -> yaw in FGPL's rotation convention.
    FGPL rot = C @ R_wc = C @ (camera->world).T; yaw = atan2(Rf[1,0], Rf[0,0])."""
    Rf = C @ np.array(cpo_R).T
    return math.degrees(math.atan2(Rf[1, 0], Rf[0, 0]))


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
        narrowing = None
        if arm == "p2":
            narrowing = {"seed_trans_radius": P2_RADIUS}
        elif arm == "p3":
            yaws = [_fgpl_yaw_from_cpo_R(cpo[r["pano_name"]]["R"]) for r in rows]
            narrowing = {"seed_trans_radius": P2_RADIUS,
                         "seed_yaw_deg": float(np.median(yaws)), "seed_yaw_tol": P3_YAW_TOL}
        seed = sc.write_seed(arm, rows, gt, cpo, centroids=cents)
        cfg = sc.write_config(arm, rows, seed, line_map, md, feat, panos, narrowing=narrowing)
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

    out = paths.subdir("results") / "ablation.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)

    print("\n| arm | n_loc | trans median (m) | trans mean | trans max | rot median (deg) | wrong-room | runtime s |")
    print("|-----|-------|------------------|------------|-----------|------------------|-----------|-----------|")
    for arm in ARMS:
        s = results[arm]
        t, r = s["translation"], s["rotation"]
        print(f"| {arm} | {s['n_localized']}/{len(rows)} | {t['median']:.3f} | {t['mean']:.3f} | "
              f"{t['max']:.3f} | {r['median']:.1f} | {s['wrong_room_rate']:.2f} | {s['runtime_s']:.0f} |")
    print("\nwrote", out)


if __name__ == "__main__":
    main()
