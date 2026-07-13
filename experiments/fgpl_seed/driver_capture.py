"""D28 driver isolation: capture residual percentile grids for match_color / subsample
toggles at the SAME cached poses used by D27, to attribute the a(3/12)->a'(9/12)
prefix_correct gap. All unweighted (weighting is the expensive factor, tested only by
elimination unless needed). Reuses cpo_cache poses -> one sampling forward per pair.
Run in panopin-gpu. Writes work/seeds/residuals_<tag>.json."""
import os, sys, json, time, numpy as np
_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, "..", "..", "src"))
sys.path.insert(0, os.path.join(_HERE, "..", ".."))
from experiments.fgpl_seed import subset, paths
from panopin.determinism import pin
from panopin.cpo_config import load_cfg
from panopin import cpo_adapter

VARIANTS = [  # (tag, match_color, seed)
    ("mc",    True,  0),   # a'_mc: match_color ON, else like a'  -> isolates match_color
    ("seed1", False, 1),   # a'_seed1: raw, different subsample   -> isolates subsample
    ("seed2", False, 2),   # a'_seed2: raw, another subsample     -> subsample robustness
]


def main(sample_rate=30):
    pin()
    cfg = load_cfg(sample_rate=sample_rate)
    rows = subset.build_subset()
    clouds = {r["room"]: r["cloud_txt"] for r in rows}
    cache = json.load(open(paths.WORK / "seeds" / "cpo_cache.json"))
    for tag, mc, seed in VARIANTS:
        grids, t0 = {}, time.time()
        for r in rows:
            pano_name, pano_jpg = r["pano_name"], r["pano_jpg"]
            grids[pano_name] = {}
            for room, cloud in clouds.items():
                pose = cache[pano_name]["poses"][room]
                resid = cpo_adapter.residuals_at_pose(cfg, pano_jpg, cloud, pose["t"], pose["R"],
                                                      match_color=mc, seed=seed)
                grid = ([999.0] * 101 if len(resid) == 0
                        else np.percentile(resid, np.arange(101)).astype(float).tolist())
                grids[pano_name][room] = grid
            print(f"[{tag}] {pano_name} done ({len(clouds)} rooms)", flush=True)
        out = paths.subdir("seeds") / f"residuals_{tag}.json"
        with open(out, "w") as f:
            json.dump(grids, f)
        print(f"[{tag}] wrote {out}  ({(time.time()-t0)/60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
