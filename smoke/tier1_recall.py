"""G1/G4: on several in-frame Area_3 panos, does cheap Tier-1 keep the correct room
in its top-k, and how fast is it per room? GT is dev-only (never in src/panopin)."""
import argparse, glob, os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "eval"))
from panopin.determinism import pin
from panopin.cpo_config import load_cfg
from panopin.cpo_adapter import score_room_cheap
import s3dis_gt


def resolve_pano(uuid, rgb_dir):
    for p in glob.glob(os.path.join(rgb_dir, "*.png")):
        parts = os.path.basename(p).split("_")
        if len(parts) >= 2 and parts[1] == uuid:
            return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rooms", default="office_3,office_5,office_7,conferenceRoom_1,WC_1")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--sample-rate", type=int, default=30)
    args = ap.parse_args()
    pin(0)
    cfg = load_cfg(sample_rate=args.sample_rate, num_yaw=4, num_pitch=4, num_roll=4, num_trans=10)
    g = s3dis_gt.load_config(None); a = g["s3dis"]["Area_3"]
    gt = s3dis_gt.load_gt("Area_3", g)
    all_rooms = s3dis_gt.candidate_rooms("Area_3", g)
    clouds = {r: os.path.join(a["rooms_dir"], r, r + ".txt") for r in all_rooms}
    hits = 0; total = 0; per_room_sec = []
    for room in [r.strip() for r in args.rooms.split(",")]:
        uuid = next((u for u, v in gt.items() if v["room"] == room), None)
        pano = resolve_pano(uuid, a["pano_rgb_dir"]) if uuid else None
        if not pano:
            print(f"{room}: no pano"); continue
        scored = []
        for cand, cloud in clouds.items():
            t0 = time.time(); _, _, loss = score_room_cheap(cfg, pano, cloud)
            per_room_sec.append(time.time() - t0); scored.append((loss, cand))
        scored.sort()
        topk = [c for _, c in scored[:args.top_k]]
        ok = room in topk; hits += ok; total += 1
        rank = [c for _, c in scored].index(room) + 1
        print(f"{room:18s} rank={rank:2d} top{args.top_k}={'HIT' if ok else 'MISS'}  best={scored[0][1]}")
    print(f"\nTier-1 recall@{args.top_k}: {hits}/{total}")
    print(f"mean sec/room: {np.mean(per_room_sec):.2f}  (G4: want < ~2 s)")
    return 0 if hits == total else 1


if __name__ == "__main__":
    sys.exit(main())
