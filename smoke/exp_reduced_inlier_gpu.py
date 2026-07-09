"""Test: reduce ONLY the inlier pose pool (the Python-loop bottleneck) while keeping
the full main pose search + Adam refine. Does recall survive at much lower cost?"""
import argparse, glob, os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "eval"))
from panopin.determinism import pin
from panopin.cpo_config import load_cfg
from panopin.cpo_adapter import localize_pair
import s3dis_gt

def resolve(uuid, d):
    for p in glob.glob(os.path.join(d, "*.png")):
        pt = os.path.basename(p).split("_")
        if len(pt) >= 2 and pt[1] == uuid: return p
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rooms", default="office_3,office_5,office_7")
    ap.add_argument("--sample-rate", type=int, default=30)
    ap.add_argument("--recall-k", type=int, default=5)
    args = ap.parse_args()
    pin(0)
    # reduce ONLY the inlier pool; keep full main search (8/8/8, 50 trans) + full refine (top_k=6, num_iter=100)
    cfg = load_cfg(sample_rate=args.sample_rate,
                   inlier_num_yaw=4, inlier_num_pitch=4, inlier_num_roll=4, inlier_num_trans=10,
                   inlier_num_split_h=8, inlier_num_split_w=16)
    g = s3dis_gt.load_config(None); a = g["s3dis"]["Area_3"]
    gt = s3dis_gt.load_gt("Area_3", g)
    clouds = {r: os.path.join(a["rooms_dir"], r, r + ".txt") for r in s3dis_gt.candidate_rooms("Area_3", g)}
    hits=0; total=0; secs=[]
    for room in [r.strip() for r in args.rooms.split(",")]:
        uuid = next((u for u,v in gt.items() if v["room"]==room), None)
        pano = resolve(uuid, a["pano_rgb_dir"]) if uuid else None
        if not pano: print(f"{room}: no pano"); continue
        scored=[]
        for cand, cloud in clouds.items():
            t0=time.time(); _,_,loss=localize_pair(cfg, pano, cloud); secs.append(time.time()-t0); scored.append((loss,cand))
        scored.sort(); rank=[c for _,c in scored].index(room)+1; ok=rank<=args.recall_k; hits+=ok; total+=1
        print(f"{room:16s} rank={rank:2d} top{args.recall_k}={'HIT' if ok else 'MISS'} true={dict((c,l) for l,c in scored)[room]:.4f} top3={[f'{c}:{l:.3f}' for l,c in scored[:3]]}")
    print(f"\nreduced-inlier-only recall@{args.recall_k}: {hits}/{total}")
    print(f"mean sec/room: {np.mean(secs):.2f}")
    return 0 if hits==total else 1

if __name__=="__main__":
    sys.exit(main())
