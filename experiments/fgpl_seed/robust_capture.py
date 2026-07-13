"""Capture per-point residual percentile grids at each room's cached refined pose.
Reuses cpo_cache poses -> one sampling forward per (pano,room) (no Adam/search), ~3 min.
Output: work/seeds/residuals.json = {pano: {room: [101 percentiles]}}. Run in panopin-gpu."""
import os, sys, json, time, numpy as np
_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, "..", "..", "src"))
sys.path.insert(0, os.path.join(_HERE, "..", ".."))
from experiments.fgpl_seed import subset, paths
from panopin.determinism import pin
from panopin.cpo_config import load_cfg
from panopin import cpo_adapter

def main(sample_rate=30):
    pin()
    cfg = load_cfg(sample_rate=sample_rate)
    rows = subset.build_subset()
    clouds = {r["room"]: r["cloud_txt"] for r in rows}
    cache = json.load(open(paths.WORK / "seeds" / "cpo_cache.json"))
    grids, t0 = {}, time.time()
    for r in rows:
        pano_name, pano_jpg = r["pano_name"], r["pano_jpg"]
        grids[pano_name] = {}
        for room, cloud in clouds.items():
            pose = cache[pano_name]["poses"][room]
            resid = cpo_adapter.residuals_at_pose(cfg, pano_jpg, cloud, pose["t"], pose["R"])
            if len(resid) == 0:
                grid = [999.0] * 101
            else:
                grid = np.percentile(resid, np.arange(101)).astype(float).tolist()
            grids[pano_name][room] = grid
        print(f"{pano_name} done ({len(clouds)} rooms)", flush=True)
    out = paths.subdir("seeds") / "residuals.json"
    with open(out, "w") as f:
        json.dump(grids, f)
    print(f"\nwrote {out}  ({(time.time()-t0)/60:.1f} min)", flush=True)

if __name__ == "__main__":
    main()
