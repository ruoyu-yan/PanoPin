"""Generate FGPL's multi-room candidate pool per pano by running the estimator in GLOBAL
mode (whole-scene search -> top-k spans same-shape rooms) with the flag-gated candidate
dump (Task 3). Global mode intentionally lets cross-room false minima populate the pool —
that IS the pool color must disambiguate (InLoc pattern). Run in panopin-gpu.
-> work/seeds/fusion_pool.json (+ per-pano distinct-room coverage)."""
import os, sys, json
_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, "..", "..", "src"))
sys.path.insert(0, os.path.join(_HERE, "..", ".."))
from experiments.fgpl_seed import subset, seed_and_config as sc, run_arm, paths
from eval import s3dis_gt

TOP_K = 30


def _nearest_room(t_xyz, centroids):
    return min(centroids, key=lambda room: sum((t_xyz[i]-centroids[room][i])**2 for i in range(3)))


def main():
    rows = subset.build_subset()
    gt = s3dis_gt.load_gt("Area_3", s3dis_gt.load_config(None))
    cents = sc.room_centroids(rows)
    md = sc.write_identity_metadata()
    line_map = paths.WORK / "linemap" / "3d_line_map.pkl"
    feat, panos = paths.WORK / "features", paths.WORK / "panos"
    # Seed content is irrelevant in global mode (the estimator only reads camera_position
    # under use_local); write a trivial alignment so the config path resolves.
    paths.subdir("seeds")
    dummy_seed = paths.WORK / "seeds" / "pool_dummy.json"
    json.dump({"metadata": {}, "matches": []}, open(dummy_seed, "w"))
    cfg = sc.write_config("fusion_pool", rows, dummy_seed, line_map, md, feat, panos,
                          use_local=False, extra={"dump_candidates": True, "top_k": TOP_K})
    res = run_arm.run_arm_pool(cfg, rows)

    pool = {}
    for r in rows:
        entry = res[r["pano_name"]]
        cands = entry["candidates"]
        rooms = {_nearest_room(c["t"], cents) for c in cands}
        gt_in = r["room"] in rooms
        pool[r["pano_name"]] = {"best": entry["best"], "candidates": cands}
        print(f"{r['pano_name']} true={r['room']:11s} n_cand={len(cands):2d} "
              f"rooms={len(rooms)} gt_in_pool={gt_in} {'OK' if gt_in else 'MISS'}", flush=True)

    out = paths.subdir("seeds") / "fusion_pool.json"
    json.dump(pool, open(out, "w"))
    gt_cov = sum(1 for r in rows if r["room"] in {_nearest_room(c["t"], cents)
                 for c in pool[r["pano_name"]]["candidates"]})
    print(f"\nGT-room in pool: {gt_cov}/{len(rows)}   wrote {out}")


if __name__ == "__main__":
    main()
