"""Whole-area validation (D28 follow-up, user-approved): localize the SAME 12 panos vs
ALL 23 Area_3 candidate room clouds -> per_room CPO loss + per-room pose. Tests whether
the raw-mean gate beats CPO's match_color+weighted loss on the whole-area candidate set
where CPO-loss collapsed (v1 D21: whole-area recall 17-33%). ~90 min GPU (12 x 23
localizations). Saves the cache incrementally (after each pano) so a crash is recoverable.
Run in panopin-gpu. -> work/seeds/wholearea_cache.json (same format as cpo_cache.json)."""
import os, sys, json, time
_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, "..", "..", "src"))
sys.path.insert(0, os.path.join(_HERE, "..", ".."))
from experiments.fgpl_seed import subset, paths
from eval import s3dis_gt
from panopin.determinism import pin
from panopin.cpo_config import load_cfg
from panopin.cpo_adapter import localize_pair


def all_candidate_clouds():
    cfg = s3dis_gt.load_config(None)
    rd = cfg["s3dis"]["Area_3"]["rooms_dir"]
    rooms = sorted(d for d in os.listdir(rd)
                   if os.path.isdir(os.path.join(rd, d)) and os.path.exists(os.path.join(rd, d, d + ".txt")))
    return {room: os.path.join(rd, room, room + ".txt") for room in rooms}


def main(sample_rate=30):
    pin()
    cfg = load_cfg(sample_rate=sample_rate)
    rows = subset.build_subset()
    clouds = all_candidate_clouds()
    print(f"{len(rows)} panos x {len(clouds)} candidate rooms", flush=True)
    out = paths.subdir("seeds") / "wholearea_cache.json"
    cache, t_start = {}, time.time()
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
        with open(out, "w") as f:                       # incremental save (crash-safe)
            json.dump(cache, f)
        print(f"[{i+1}/{len(rows)}] {r['pano_name']} true={r['room']:12s} -> min-loss={room:12s} "
              f"({'OK' if room == r['room'] else 'X'})  ({(time.time()-t_start)/60:.1f} min)", flush=True)
    raw_hits = sum(1 for r in rows if cache[r["pano_name"]]["room"] == r["room"])
    print(f"\nraw min-loss recall@1 = {raw_hits}/{len(rows)} (vs 23 rooms)   wrote {out}   "
          f"({(time.time()-t_start)/60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
