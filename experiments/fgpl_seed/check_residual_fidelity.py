"""Fidelity gate for residuals_at_pose: mean(residuals) == cpo.sampling_loss scalar at
the SAME pose, to 1e-4. Deterministic (fixed pose, no Adam). Run in panopin-gpu."""
import os, sys, numpy as np, torch, cv2
_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, "..", "..", "src"))
sys.path.insert(0, os.path.join(_HERE, "..", ".."))
from experiments.fgpl_seed import subset
from panopin.determinism import pin
from panopin.cpo_config import load_cfg
from panopin import cpo_adapter
import data_utils
from dict_utils import get_init_dict_cpo
from utils import generate_trans_points, generate_rot_points, histogram_pose_search
from cpo.sampling_loss import sampling_loss

def main():
    pin()
    cfg = load_cfg(sample_rate=30)
    row = subset.build_subset()[0]
    pano, cloud = row["pano_jpg"], row["cloud_txt"]
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    xyz_np, rgb_np = data_utils.read_txt_pcd(cloud, sample_rate=getattr(cfg, 'sample_rate', 1))
    xyz = torch.from_numpy(xyz_np).float().to(device)
    rgb = torch.from_numpy(rgb_np).float().to(device)
    orig = cv2.resize(cv2.cvtColor(cv2.imread(pano), cv2.COLOR_BGR2RGB), (2048, 1024))
    img = (torch.from_numpy(orig).float() / 255.).to(device)

    init_dict = get_init_dict_cpo(cfg)
    rot = generate_rot_points(init_dict, device=device)
    trans = generate_trans_points(xyz, init_dict, device=device)
    input_trans, input_rot = histogram_pose_search(
        img, xyz, rgb, trans, rot, 1, init_dict['num_split_h'], init_dict['num_split_w'],
        None, init_dict['sin_hist'])

    t_used, R_used, L = sampling_loss(img, xyz, rgb, input_trans, input_rot, 0, cfg, return_list=True)
    t_used = t_used.detach().numpy().reshape(3); R_used = R_used.detach().numpy().reshape(3, 3)
    L = float(L.detach())

    resid = cpo_adapter.residuals_at_pose(cfg, pano, cloud, t_used, R_used)
    m = float(np.mean(resid))
    print(f"sampling_loss L = {L:.6f}   mean(residuals) = {m:.6f}   |diff| = {abs(m - L):.2e}   n_resid = {len(resid)}")
    assert abs(m - L) < 1e-4, f"FIDELITY FAIL: {abs(m - L):.2e} >= 1e-4"
    print("FIDELITY OK")

if __name__ == "__main__":
    main()
