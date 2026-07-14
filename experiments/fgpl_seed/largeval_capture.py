"""Larger-n validation (D33): per-point residual grids at the largeval cached poses (32 panos
x 8 pool rooms), the raw low-pct path (match_color=False, seed=0). Run in panopin-gpu.
-> work/seeds/largeval_residuals.json (101 percentiles per pair, wholearea_residuals format)."""
import os, sys, json, time, numpy as np
_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, "..", "..", "src"))
sys.path.insert(0, os.path.join(_HERE, "..", ".."))
from experiments.fgpl_seed import largeval, paths
from panopin.determinism import pin
from panopin.cpo_config import load_cfg
from panopin import cpo_adapter


def main(sample_rate=30):
    pin()
    cfg = load_cfg(sample_rate=sample_rate)
    rows = largeval.build_pool()
    clouds = largeval.pool_clouds()
    cache = json.load(open(paths.WORK / "seeds" / "largeval_cache.json"))
    grids, t0 = {}, time.time()
    for r in rows:
        name = r["pano_name"]
        grids[name] = {}
        for room, cloud in clouds.items():
            pose = cache[name]["poses"][room]
            resid = cpo_adapter.residuals_at_pose(cfg, r["pano_jpg"], cloud, pose["t"], pose["R"])
            grid = ([999.0] * 101 if len(resid) == 0
                    else np.percentile(resid, np.arange(101)).astype(float).tolist())
            grids[name][room] = grid
        print(f"{name[:8]} done ({len(clouds)} rooms)", flush=True)
    out = paths.subdir("seeds") / "largeval_residuals.json"
    with open(out, "w") as f:
        json.dump(grids, f)
    print(f"\nwrote {out}  ({(time.time()-t0)/60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
