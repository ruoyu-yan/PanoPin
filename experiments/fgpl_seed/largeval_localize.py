"""Larger-n validation (D33): CPO localize each of the 32 pool panos vs each of the 8 pool
rooms -> per_room loss + per-room pose. ~1.5 h GPU (32 x 8 localizations). Incremental save
(crash-safe). Run in panopin-gpu. -> work/seeds/largeval_cache.json (wholearea_cache format)."""
import os, sys, json, time
_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, "..", "..", "src"))
sys.path.insert(0, os.path.join(_HERE, "..", ".."))
from experiments.fgpl_seed import largeval, paths
from panopin.determinism import pin
from panopin.cpo_config import load_cfg
from panopin.cpo_adapter import localize_pair


def main(sample_rate=30):
    pin()
    cfg = load_cfg(sample_rate=sample_rate)
    rows = largeval.build_pool()
    clouds = largeval.pool_clouds()
    print(f"{len(rows)} panos x {len(clouds)} rooms", flush=True)
    out = paths.subdir("seeds") / "largeval_cache.json"
    cache, t0 = {}, time.time()
    for i, r in enumerate(rows):
        per_room, poses, best = {}, {}, None
        for room, cloud in clouds.items():
            t, R, loss = localize_pair(cfg, r["pano_jpg"], cloud)
            tl = [float(x) for x in t]
            Rl = [[float(x) for x in row] for row in R]
            per_room[room] = float(loss)
            poses[room] = {"t": tl, "R": Rl}
            if best is None or loss < best[2]:
                best = (tl, Rl, float(loss), room)
        t, R, loss, room = best
        cache[r["pano_name"]] = {"t": t, "R": R, "loss": loss, "room": room,
                                 "per_room": per_room, "poses": poses}
        with open(out, "w") as f:
            json.dump(cache, f)
        print(f"[{i+1}/{len(rows)}] {r['pano_name'][:8]} true={r['room']:16s} -> "
              f"min-loss={room:16s} ({'OK' if room == r['room'] else 'X'})  "
              f"({(time.time()-t0)/60:.1f} min)", flush=True)
    hits = sum(1 for r in rows if cache[r["pano_name"]]["room"] == r["room"])
    print(f"\nraw min-loss recall = {hits}/{len(rows)} (vs {len(clouds)} rooms)  wrote {out}  "
          f"({(time.time()-t0)/60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
