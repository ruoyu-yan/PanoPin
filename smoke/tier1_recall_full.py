"""Decisive diagnostic: does the FULL localize_pair (full inlier detection, default
CPO cfg) recall the correct room against ALL candidate rooms? The +75% discrimination
wins earlier were vs 6-room subsets that excluded the tiny/corridor clouds; this checks
the full 21-room set. GT is dev-only."""
import argparse, glob, os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "eval"))
from panopin.determinism import pin
from panopin.cpo_config import load_cfg
from panopin.cpo_adapter import localize_pair
import s3dis_gt


def resolve_pano(uuid, rgb_dir):
    for p in glob.glob(os.path.join(rgb_dir, "*.png")):
        parts = os.path.basename(p).split("_")
        if len(parts) >= 2 and parts[1] == uuid:
            return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rooms", default="office_3")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--sample-rate", type=int, default=30)
    args = ap.parse_args()
    pin(0)
    cfg = load_cfg(sample_rate=args.sample_rate)   # full inlier + defaults (top_k=6, num_iter=100)
    g = s3dis_gt.load_config(None); a = g["s3dis"]["Area_3"]
    gt = s3dis_gt.load_gt("Area_3", g)
    all_rooms = s3dis_gt.candidate_rooms("Area_3", g)
    clouds = {r: os.path.join(a["rooms_dir"], r, r + ".txt") for r in all_rooms}
    hits = 0; total = 0; secs = []
    for room in [r.strip() for r in args.rooms.split(",")]:
        uuid = next((u for u, v in gt.items() if v["room"] == room), None)
        pano = resolve_pano(uuid, a["pano_rgb_dir"]) if uuid else None
        if not pano:
            print(f"{room}: no pano"); continue
        scored = []
        for cand, cloud in clouds.items():
            t0 = time.time(); _, _, loss = localize_pair(cfg, pano, cloud)
            secs.append(time.time() - t0); scored.append((loss, cand))
        scored.sort()
        rank = [c for _, c in scored].index(room) + 1
        ok = rank <= args.top_k; hits += ok; total += 1
        print(f"{room:18s} rank={rank:2d} top{args.top_k}={'HIT' if ok else 'MISS'}  "
              f"true_loss={dict((c,l) for l,c in scored)[room]:.4f}")
        print("  full ranking: " + ", ".join(f"{c}:{l:.3f}" for l, c in scored[:8]))
    print(f"\nfull-pipeline recall@{args.top_k}: {hits}/{total}")
    print(f"mean sec/room: {np.mean(secs):.2f}")
    return 0 if hits == total else 1


if __name__ == "__main__":
    sys.exit(main())
