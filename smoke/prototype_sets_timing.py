"""Measure wall-clock to assign a small prototype's panos, where the candidate set is
the SET's OWN 3-5 rooms (the prototype scene) — not the whole area. Each pano in the
set's rooms is scored against just those rooms via the full localize_pair on GPU.
Reports per-set wall-clock + recall@1. GT dev-only. Cap panos/room to bound compute."""
import argparse, glob, os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "eval"))
from panopin.determinism import pin
from panopin.cpo_config import load_cfg
from panopin.cpo_adapter import localize_pair
import s3dis_gt

SETS = {
    "A_offices":  ["office_1", "office_4", "office_6"],       # same-shape offices (hard)
    "B_mixed":    ["office_3", "WC_1", "hallway_3"],          # office / WC / corridor
    "C_varied":   ["office_5", "office_7", "conferenceRoom_1"],
}

def resolve(uuid, d):
    for p in glob.glob(os.path.join(d, "*.png")):
        pt = os.path.basename(p).split("_")
        if len(pt) >= 2 and pt[1] == uuid: return p
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-panos-per-room", type=int, default=2)
    ap.add_argument("--sample-rate", type=int, default=30)
    args = ap.parse_args()
    pin(0)
    cfg = load_cfg(sample_rate=args.sample_rate)   # full pipeline (defaults)
    g = s3dis_gt.load_config(None); a = g["s3dis"]["Area_3"]
    gt = s3dis_gt.load_gt("Area_3", g)
    by_room = {}
    for u, v in gt.items():
        by_room.setdefault(v["room"], []).append(u)

    for set_name, rooms in SETS.items():
        set_clouds = {r: os.path.join(a["rooms_dir"], r, r + ".txt") for r in rooms}
        panos = []
        for r in rooms:
            for u in by_room.get(r, [])[:args.max_panos_per_room]:
                p = resolve(u, a["pano_rgb_dir"])
                if p: panos.append((r, u, p))
        print(f"\n=== SET {set_name} rooms={rooms} (candidates={len(rooms)})  {len(panos)} panos ===", flush=True)
        t_set = time.time(); hits = 0
        for r, u, pano in panos:
            t0 = time.time()
            scored = sorted((localize_pair(cfg, pano, c)[2], cand) for cand, c in set_clouds.items())
            dt = time.time() - t0
            rank = [c for _, c in scored].index(r) + 1
            hits += rank == 1
            print(f"  {r:16s} -> {scored[0][1]:16s} rank={rank} {'OK' if rank==1 else 'X'}  {dt:5.1f}s", flush=True)
        elapsed = time.time() - t_set
        print(f"SET {set_name}: {elapsed/60:.1f} min for {len(panos)} panos "
              f"({elapsed/max(1,len(panos)):.0f}s/pano)  recall@1={hits}/{len(panos)}", flush=True)

if __name__ == "__main__":
    main()
