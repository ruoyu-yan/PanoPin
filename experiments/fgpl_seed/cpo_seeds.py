"""Per pano: CPO localize_pair vs each of the 6 candidate rooms. Cache EVERY room's
(t,R,loss) -- so a downstream seed can use ANY room's pose, not just the min-loss winner
-- plus the min-loss winner (backward compat) and a v1 minmax-calibrated room assignment.

The raw min-loss room is degenerate (a fixed 'loss-sink' corridor wins for many panos, D21).
v1 calibration (src/panopin/calibrate.py, D24) normalizes each room's loss to its own
leave-one-out [min,max] over the other panos -- fair, no ground truth (D5) -- and lifts
room recall. cpo_seeds now emits BOTH assignments so run_all can compare the raw seed (p1)
against the calibrated seed (p1_cal). Run in panopin-gpu."""
import os, sys, json, time
# panopin lives at src/ (not pip-installed); mirror the smoke-script path setup.
_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, "..", "..", "src"))   # panopin.*
sys.path.insert(0, os.path.join(_HERE, "..", ".."))          # experiments.* (repo root)
from experiments.fgpl_seed import subset, paths
from panopin.determinism import pin
from panopin.cpo_config import load_cfg
from panopin.cpo_adapter import localize_pair
from panopin import calibrate


def main(sample_rate=30):
    pin()
    cfg = load_cfg(sample_rate=sample_rate)
    rows = subset.build_subset()
    clouds = {r["room"]: r["cloud_txt"] for r in rows}   # room -> its cloud .txt (6 unique)
    cache = {}
    t_start = time.time()
    for r in rows:
        per_room_loss, poses, best = {}, {}, None
        for room, cloud in clouds.items():
            t, R, loss = localize_pair(cfg, r["pano_jpg"], cloud)
            tl = [float(x) for x in t]
            Rl = [[float(x) for x in row] for row in R]
            per_room_loss[room] = float(loss)
            poses[room] = {"t": tl, "R": Rl}             # keep EVERY room's pose
            if best is None or loss < best[2]:
                best = (tl, Rl, float(loss), room)
        t, R, loss, room = best
        cache[r["pano_name"]] = {"t": t, "R": R, "loss": loss, "room": room,
                                 "per_room": per_room_loss, "poses": poses}
        print(f"{r['pano_name']} true={r['room']:11s} -> assigned={room:11s} "
              f"loss={loss:.3f}  {'OK' if room == r['room'] else 'X'}", flush=True)

    # v1 minmax calibration over the per-room losses (fair, no GT -- D24). Adds a
    # calibrated room + confidence per pano; the p1_cal seed uses poses[room_cal].
    loss_matrix = {p: cache[p]["per_room"] for p in cache}
    cal = calibrate.assign(loss_matrix)
    for p, a in cal.items():
        cache[p]["room_cal"] = a.room
        cache[p]["conf"] = float(a.confidence)
        cache[p]["is_confident"] = bool(a.is_confident)

    out = paths.subdir("seeds") / "cpo_cache.json"
    with open(out, "w") as f:
        json.dump(cache, f, indent=2)
    raw_hits = sum(1 for r in rows if cache[r["pano_name"]]["room"] == r["room"])
    cal_hits = sum(1 for r in rows if cache[r["pano_name"]]["room_cal"] == r["room"])
    print(f"\nroom recall@1: raw min-loss = {raw_hits}/{len(rows)}   "
          f"minmax calibrated = {cal_hits}/{len(rows)}   wrote {out}   "
          f"({(time.time()-t_start)/60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
