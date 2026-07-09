"""Research diagnostic (D18 fix): is the hallway/small-cloud degeneracy a COVERAGE bias?
For a query pano, at each room's histogram-search best pose, record the unweighted
mean residual (loss) AND coverage = fraction of cloud points that project into the
pano (non-black). If the degenerate winners (hallways/WC) have LOW coverage, a
coverage-normalized score can de-bias them cheaply (single forward, no inlier, no Adam).
GT dev-only.
"""
import argparse, glob, os, sys
import numpy as np, torch, cv2
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "eval"))
from panopin.determinism import pin
from panopin.cpo_config import load_cfg
from panopin import _cpo_path  # noqa
import data_utils
from dict_utils import get_init_dict_cpo
from color_utils import color_match
from utils import generate_trans_points, generate_rot_points, histogram_pose_search, cloud2idx, sample_from_img
import s3dis_gt


def loss_and_coverage(cfg, pano_path, cloud_path):
    device = torch.device('cpu')
    sr = getattr(cfg, 'sample_rate', 1)
    xyz_np, rgb_np = data_utils.read_txt_pcd(cloud_path, sample_rate=sr)
    xyz = torch.from_numpy(xyz_np).float(); rgb = torch.from_numpy(rgb_np).float()
    N = xyz.shape[0]
    orig = cv2.cvtColor(cv2.imread(pano_path), cv2.COLOR_BGR2RGB)
    orig = cv2.resize(orig, (2048, 1024))
    if getattr(cfg, 'match_color', False):
        mod = (torch.from_numpy(orig).float() / 255.)
        orig = (255 * color_match(mod, rgb).cpu().numpy()).astype(np.uint8)
    init = get_init_dict_cpo(cfg)
    rot = generate_rot_points(init, device=device)
    trans = generate_trans_points(xyz, init, device=device)
    idh = getattr(cfg, 'init_downsample_h', 1); idw = getattr(cfg, 'init_downsample_w', 1)
    img_s = cv2.resize(orig, (orig.shape[1] // idw, orig.shape[0] // idh))
    img_s = (torch.from_numpy(img_s).float() / 255.)
    in_trans, in_rot = histogram_pose_search(img_s, xyz, rgb, trans, rot, 1,
                                             init['num_split_h'], init['num_split_w'], None, init['sin_hist'])
    # project at the best pose
    from utils import rot_from_ypr
    R = rot_from_ypr(in_rot[0])
    new_xyz = (torch.matmul(R, (xyz.t() - in_trans[0].reshape(3, -1)))).t()
    img = (torch.from_numpy(orig).float() / 255.)
    coord = cloud2idx(new_xyz)
    sample_rgb = sample_from_img(img, coord)
    mask = torch.sum(sample_rgb == 0, dim=1) != 3
    nm = int(mask.sum())
    loss = float(torch.norm(sample_rgb[mask] - rgb[mask], dim=-1).mean()) if nm else float('nan')
    return loss, nm, N


def resolve(uuid, d):
    for p in glob.glob(os.path.join(d, "*.png")):
        pt = os.path.basename(p).split("_")
        if len(pt) >= 2 and pt[1] == uuid:
            return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rooms", default="office_5,office_7")
    ap.add_argument("--sample-rate", type=int, default=30)
    args = ap.parse_args()
    pin(0)
    cfg = load_cfg(sample_rate=args.sample_rate)
    g = s3dis_gt.load_config(None); a = g["s3dis"]["Area_3"]
    gt = s3dis_gt.load_gt("Area_3", g)
    clouds = {r: os.path.join(a["rooms_dir"], r, r + ".txt") for r in s3dis_gt.candidate_rooms("Area_3", g)}
    for room in [r.strip() for r in args.rooms.split(",")]:
        uuid = next((u for u, v in gt.items() if v["room"] == room), None)
        pano = resolve(uuid, a["pano_rgb_dir"]) if uuid else None
        if not pano:
            print(f"{room}: no pano"); continue
        rows = []
        for cand, cloud in clouds.items():
            loss, nm, N = loss_and_coverage(cfg, pano, cloud)
            cov = nm / N if N else 0
            rows.append((loss, cand, cov, nm, N))
        rows.sort()  # by raw loss
        print(f"\n=== query {room}: rooms ranked by RAW loss (degenerate) ===")
        print(f"{'room':16s} {'loss':>7s} {'cov':>6s} {'n_match':>8s} {'N':>8s} {'loss/cov':>9s}")
        for loss, cand, cov, nm, N in rows:
            mark = " <-TRUE" if cand == room else ""
            print(f"{cand:16s} {loss:7.3f} {cov:6.3f} {nm:8d} {N:8d} {loss/max(cov,1e-6):9.3f}{mark}")
        # where does TRUE rank under raw loss vs loss/coverage?
        raw_rank = [c for _, c, *_ in rows].index(room) + 1
        by_adj = sorted(rows, key=lambda r: r[0] / max(r[2], 1e-6))
        adj_rank = [c for _, c, *_ in by_adj].index(room) + 1
        print(f"TRUE '{room}': raw-loss rank={raw_rank}, loss/coverage rank={adj_rank}")


if __name__ == "__main__":
    main()
