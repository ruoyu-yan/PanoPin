"""Return-valued CPO single-pair localizer (composes CPO primitives).
Body mirrors third_party/cpo/cpo/localize_single.py (lines 38-150), minus the
result.png visualization, plus a return of (t, R, loss)."""
import numpy as np, torch, cv2
from panopin import _cpo_path  # noqa: F401
import data_utils
from dict_utils import get_init_dict_cpo
from color_utils import color_match, color_mod
from utils import (out_of_room, generate_trans_points, generate_rot_points,
                   make_score_map_2d, process_score_map_2d, make_score_map_3d,
                   histogram_pose_search)
from cpo.sampling_loss import refine_pose_sampling_loss, sampling_loss


def localize_pair(cfg, pano_path, cloud_path):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    sample_rate = getattr(cfg, 'sample_rate', 1)
    top_k_candidate = getattr(cfg, 'top_k_candidate', 5)
    init_downsample_h = getattr(cfg, 'init_downsample_h', 1)
    init_downsample_w = getattr(cfg, 'init_downsample_w', 1)
    main_downsample_h = getattr(cfg, 'main_downsample_h', 1)
    main_downsample_w = getattr(cfg, 'main_downsample_w', 1)

    xyz_np, rgb_np = data_utils.read_txt_pcd(cloud_path, sample_rate=sample_rate)
    xyz = torch.from_numpy(xyz_np).float().to(device)
    rgb = torch.from_numpy(rgb_np).float().to(device)

    orig_img = cv2.cvtColor(cv2.imread(pano_path), cv2.COLOR_BGR2RGB)
    orig_img = cv2.resize(orig_img, (2048, 1024))

    sharpen_color = getattr(cfg, 'sharpen_color', False)
    match_color = getattr(cfg, 'match_color', False)
    num_bins = getattr(cfg, 'num_bins', 256)
    mod_img = (torch.from_numpy(orig_img).float() / 255.).to(device)
    if sharpen_color or match_color:
        if match_color:
            new_img = color_match(mod_img, rgb); orig_img = (255 * new_img.cpu().numpy()).astype(np.uint8)
        if sharpen_color:
            new_img, rgb = color_mod(mod_img, rgb, num_bins); orig_img = (255 * new_img.cpu().numpy()).astype(np.uint8)

    img = cv2.resize(orig_img, (orig_img.shape[1] // init_downsample_w, orig_img.shape[0] // init_downsample_h))
    img = (torch.from_numpy(img).float() / 255.).to(device)
    input_xyz = xyz
    init_dict = get_init_dict_cpo(cfg)

    inlier_init_dict = dict(init_dict); inlier_init_dict['is_inlier_dict'] = True
    inlier_init_dict['num_trans'] = getattr(cfg, 'inlier_num_trans', init_dict['num_trans'])
    inlier_init_dict['num_yaw'] = getattr(cfg, 'inlier_num_yaw', 4)
    inlier_init_dict['num_pitch'] = getattr(cfg, 'inlier_num_pitch', 4)
    inlier_init_dict['num_roll'] = getattr(cfg, 'inlier_num_roll', 4)
    inlier_init_dict['trans_init_mode'] = getattr(cfg, 'inlier_trans_init_mode', 'quantile')
    inlier_test_trans = generate_trans_points(input_xyz, inlier_init_dict, device=input_xyz.device)
    inlier_test_rot = generate_rot_points(inlier_init_dict, device=input_xyz.device)
    inlier_num_split_h = getattr(cfg, 'inlier_num_split_h', 8)
    inlier_num_split_w = getattr(cfg, 'inlier_num_split_w', 16)
    margin = inlier_num_split_h // 8

    score_map_2d = make_score_map_2d(img, input_xyz, rgb, inlier_test_trans, inlier_test_rot,
                                     inlier_num_split_h, inlier_num_split_w, margin)
    score_map_2d_search = process_score_map_2d(torch.zeros(cfg.num_split_h, cfg.num_split_w, device=xyz.device),
                                               score_map_2d, 'preserve', 0.0)
    score_map_2d_refine = process_score_map_2d(torch.from_numpy(orig_img).to(xyz.device),
                                               score_map_2d, 'preserve', 0.0).unsqueeze(-1)
    score_map_3d = make_score_map_3d(img, xyz, rgb, inlier_test_trans, inlier_test_rot,
                                     inlier_num_split_h, inlier_num_split_w, margin, match_rgb=match_color)
    pcd_weight = score_map_3d

    rot = generate_rot_points(init_dict, device=img.device)
    trans = generate_trans_points(xyz, init_dict, device=img.device)
    input_trans, input_rot = histogram_pose_search(img, input_xyz, rgb, trans, rot, top_k_candidate,
                                                    init_dict['num_split_h'], init_dict['num_split_w'],
                                                    score_map_2d_search, init_dict['sin_hist'])

    img = cv2.resize(orig_img, (orig_img.shape[1] // main_downsample_w, orig_img.shape[0] // main_downsample_h))
    img = (torch.from_numpy(img).float() / 255.).to(device)

    result = []
    for i in range(top_k_candidate):
        result.append(refine_pose_sampling_loss(img, input_xyz, rgb, input_trans, input_rot, i, cfg,
                                                 img_weight=score_map_2d_refine, pcd_weight=pcd_weight))
    with torch.no_grad():
        result = np.asarray(result, dtype=object)
        min_ind = int(result[:, 2].argmin())
        t = result[min_ind, 0].detach().numpy().reshape(3)
        R = result[min_ind, 1].detach().numpy().reshape(3, 3)
        loss = float(result[min_ind, 2].detach())
    return t, R, loss


def score_room_cheap(cfg, pano_path, cloud_path):
    """Cheap Tier-1 room score: small-pool histogram_pose_search (no inlier score
    maps) -> single-forward sampling_loss. Returns (t, R, loss); no Adam. CPU-only."""
    device = torch.device('cpu')
    sample_rate = getattr(cfg, 'sample_rate', 1)

    xyz_np, rgb_np = data_utils.read_txt_pcd(cloud_path, sample_rate=sample_rate)
    xyz = torch.from_numpy(xyz_np).float().to(device)
    rgb = torch.from_numpy(rgb_np).float().to(device)

    orig_img = cv2.cvtColor(cv2.imread(pano_path), cv2.COLOR_BGR2RGB)
    orig_img = cv2.resize(orig_img, (2048, 1024))
    if getattr(cfg, 'match_color', False):
        mod_img = (torch.from_numpy(orig_img).float() / 255.).to(device)
        new_img = color_match(mod_img, rgb)
        orig_img = (255 * new_img.cpu().numpy()).astype(np.uint8)

    init_dict = get_init_dict_cpo(cfg)
    rot = generate_rot_points(init_dict, device=device)
    trans = generate_trans_points(xyz, init_dict, device=device)

    idh = getattr(cfg, 'init_downsample_h', 1); idw = getattr(cfg, 'init_downsample_w', 1)
    img_search = cv2.resize(orig_img, (orig_img.shape[1] // idw, orig_img.shape[0] // idh))
    img_search = (torch.from_numpy(img_search).float() / 255.).to(device)
    input_trans, input_rot = histogram_pose_search(
        img_search, xyz, rgb, trans, rot, 1,
        init_dict['num_split_h'], init_dict['num_split_w'], None, init_dict['sin_hist'])

    mdh = getattr(cfg, 'main_downsample_h', 1); mdw = getattr(cfg, 'main_downsample_w', 1)
    img_score = cv2.resize(orig_img, (orig_img.shape[1] // mdw, orig_img.shape[0] // mdh))
    img_score = (torch.from_numpy(img_score).float() / 255.).to(device)
    t_c, R_c, loss_c = sampling_loss(img_score, xyz, rgb, input_trans, input_rot, 0, cfg,
                                     return_list=True)
    t = t_c.detach().numpy().reshape(3)
    R = R_c.detach().numpy().reshape(3, 3)
    return t, R, float(loss_c.detach())
