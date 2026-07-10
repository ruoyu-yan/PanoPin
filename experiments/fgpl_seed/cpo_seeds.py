"""Per pano: CPO localize_pair vs each of the 6 candidate rooms -> assign min-loss
room; keep that room's (t,R). This is the seed for P1/P2/P3. Run in panopin-gpu."""
import os, sys, json, time
# panopin lives at src/ (not pip-installed); mirror the smoke-script path setup.
_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, "..", "..", "src"))   # panopin.*
sys.path.insert(0, os.path.join(_HERE, "..", ".."))          # experiments.* (repo root)
from experiments.fgpl_seed import subset, paths
from panopin.determinism import pin
from panopin.cpo_config import load_cfg
from panopin.cpo_adapter import localize_pair


def main(sample_rate=30):
    pin()
    cfg = load_cfg(sample_rate=sample_rate)
    rows = subset.build_subset()
    clouds = {r["room"]: r["cloud_txt"] for r in rows}   # room -> its cloud .txt (6 unique)
    cache = {}
    t_start = time.time()
    for r in rows:
        per_room, best = {}, None
        for room, cloud in clouds.items():
            t, R, loss = localize_pair(cfg, r["pano_jpg"], cloud)
            per_room[room] = loss
            if best is None or loss < best[2]:
                best = (t, R, loss, room)
        t, R, loss, room = best
        cache[r["pano_name"]] = {"t": [float(x) for x in t],
                                 "R": [[float(x) for x in row] for row in R],
                                 "loss": float(loss), "room": room, "per_room": per_room}
        print(f"{r['pano_name']} true={r['room']:11s} -> assigned={room:11s} "
              f"loss={loss:.3f}  {'OK' if room == r['room'] else 'X'}", flush=True)
    out = paths.subdir("seeds") / "cpo_cache.json"
    with open(out, "w") as f:
        json.dump(cache, f, indent=2)
    hits = sum(1 for r in rows if cache[r["pano_name"]]["room"] == r["room"])
    print(f"\nroom recall@1 = {hits}/{len(rows)}   wrote {out}   ({(time.time()-t_start)/60:.1f} min)",
          flush=True)


if __name__ == "__main__":
    main()
