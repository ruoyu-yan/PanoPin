"""Research experiment (D18 fix): does an Adam-refined scorer WITHOUT the ~20s inlier
detection recall the correct room? Isolates whether the Adam pose-refine (not the
inlier weighting) is the load-bearing ingredient the cheap scorer was missing.

score_room_adam = histogram_pose_search (full pool, cheap) -> top_k candidate poses ->
refine_pose_sampling_loss with NO weights (img_weight=pcd_weight=None -> unweighted
mean residual, Adam-refined) -> min loss. NO make_score_map_2d/3d. GT dev-only.
"""
import argparse, glob, os, sys, time
import numpy as np, torch, cv2
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "eval"))
from panopin.determinism import pin
from panopin.cpo_config import load_cfg
from panopin import _cpo_path  # noqa
import data_utils
from dict_utils import get_init_dict_cpo
from color_utils import color_match
from utils import generate_trans_points, generate_rot_points, histogram_pose_search
from cpo.sampling_loss import refine_pose_sampling_loss
import s3dis_gt


def score_room_adam(cfg, pano_path, cloud_path):
    device = torch.device('cpu')
    sr = getattr(cfg, 'sample_rate', 1)
    top_k = getattr(cfg, 'top_k_candidate', 6)
    xyz_np, rgb_np = data_utils.read_txt_pcd(cloud_path, sample_rate=sr)
    xyz = torch.from_numpy(xyz_np).float().to(device)
    rgb = torch.from_numpy(rgb_np).float().to(device)
    orig = cv2.cvtColor(cv2.imread(pano_path), cv2.COLOR_BGR2RGB)
    orig = cv2.resize(orig, (2048, 1024))
    if getattr(cfg, 'match_color', False):
        mod = (torch.from_numpy(orig).float() / 255.).to(device)
        orig = (255 * color_match(mod, rgb).cpu().numpy()).astype(np.uint8)
    init = get_init_dict_cpo(cfg)
    rot = generate_rot_points(init, device=device)
    trans = generate_trans_points(xyz, init, device=device)
    idh = getattr(cfg, 'init_downsample_h', 1); idw = getattr(cfg, 'init_downsample_w', 1)
    img_s = cv2.resize(orig, (orig.shape[1] // idw, orig.shape[0] // idh))
    img_s = (torch.from_numpy(img_s).float() / 255.).to(device)
    in_trans, in_rot = histogram_pose_search(img_s, xyz, rgb, trans, rot, top_k,
                                             init['num_split_h'], init['num_split_w'], None, init['sin_hist'])
    img = (torch.from_numpy(orig).float() / 255.).to(device)
    best = None
    for i in range(top_k):
        r = refine_pose_sampling_loss(img, xyz, rgb, in_trans, in_rot, i, cfg)  # no weights
        loss = float(r[2].detach())
        if best is None or loss < best:
            best = loss
    return best


def resolve(uuid, d):
    for p in glob.glob(os.path.join(d, "*.png")):
        pt = os.path.basename(p).split("_")
        if len(pt) >= 2 and pt[1] == uuid:
            return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rooms", default="office_3,office_5,office_7")
    ap.add_argument("--top-k", type=int, default=6)
    ap.add_argument("--num-iter", type=int, default=100)
    ap.add_argument("--sample-rate", type=int, default=30)
    ap.add_argument("--recall-k", type=int, default=5)
    args = ap.parse_args()
    pin(0)
    cfg = load_cfg(sample_rate=args.sample_rate, top_k_candidate=args.top_k, num_iter=args.num_iter)
    g = s3dis_gt.load_config(None); a = g["s3dis"]["Area_3"]
    gt = s3dis_gt.load_gt("Area_3", g)
    clouds = {r: os.path.join(a["rooms_dir"], r, r + ".txt") for r in s3dis_gt.candidate_rooms("Area_3", g)}
    hits = 0; total = 0; secs = []
    for room in [r.strip() for r in args.rooms.split(",")]:
        uuid = next((u for u, v in gt.items() if v["room"] == room), None)
        pano = resolve(uuid, a["pano_rgb_dir"]) if uuid else None
        if not pano:
            print(f"{room}: no pano"); continue
        scored = []
        for cand, cloud in clouds.items():
            t0 = time.time(); loss = score_room_adam(cfg, pano, cloud)
            secs.append(time.time() - t0); scored.append((loss, cand))
        scored.sort()
        rank = [c for _, c in scored].index(room) + 1
        ok = rank <= args.recall_k; hits += ok; total += 1
        print(f"{room:18s} rank={rank:2d} top{args.recall_k}={'HIT' if ok else 'MISS'}  "
              f"true={dict((c,l) for l,c in scored)[room]:.4f}  top3={[f'{c}:{l:.3f}' for l,c in scored[:3]]}")
    print(f"\nAdam-no-inlier recall@{args.recall_k}: {hits}/{total}")
    print(f"mean sec/room: {np.mean(secs):.2f}")
    return 0 if hits == total else 1


if __name__ == "__main__":
    sys.exit(main())
