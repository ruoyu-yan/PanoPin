"""Color-score every FGPL candidate pose: color = mean(residuals_at_pose) at the candidate's
FGPL rotation Rp FED AS-IS (FGPL's Rp = C @ R_wc is already the equirect convention
residuals_at_pose expects — derived in Task 1's fusion_convention_probe.py; do NOT apply
C^T). Scored against the candidate's nearest-room cloud. Fair: reads only the pool + clouds,
no GT (D5). Run in panopin-gpu. -> work/seeds/fusion_pool_colored.json."""
import os, sys, json
_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, "..", "..", "src"))
sys.path.insert(0, os.path.join(_HERE, "..", ".."))
from experiments.fgpl_seed import subset, seed_and_config as sc, paths
from panopin.determinism import pin
from panopin.cpo_config import load_cfg
from panopin.cpo_adapter import residuals_at_pose


def _nearest_room(t, cents):
    return min(cents, key=lambda r: sum((t[i]-cents[r][i])**2 for i in range(3)))


def main(sample_rate=30):
    pin()
    cfg = load_cfg(sample_rate=sample_rate)
    rows = subset.build_subset()
    clouds = {r["room"]: r["cloud_txt"] for r in rows}
    cents = sc.room_centroids(rows)
    pool = json.load(open(paths.WORK / "seeds" / "fusion_pool.json"))
    row_by_name = {r["pano_name"]: r for r in rows}

    for name, entry in pool.items():
        pano_jpg = row_by_name[name]["pano_jpg"]
        for c in entry["candidates"]:
            room = _nearest_room(c["t"], cents)
            res = residuals_at_pose(cfg, pano_jpg, clouds[room], c["t"], c["R"])  # Rp as-is
            c["color"] = float(res.mean())
            c["room"] = room
            c["geom"] = float(c["n_tight"])          # alias the geom score the selectors read
        print(f"{name}: scored {len(entry['candidates'])} candidates", flush=True)

    out = paths.subdir("seeds") / "fusion_pool_colored.json"
    json.dump(pool, open(out, "w"))
    print("wrote", out)


if __name__ == "__main__":
    main()
