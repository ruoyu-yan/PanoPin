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


def residuals_at_pose(cfg, pano_path, cloud_path, t, R, match_color=False, seed=0):
    """Per-point color residuals ||sample_rgb - cloud_rgb|| at a FIXED given pose (t,R),
    replicating ONLY cpo.sampling_loss's sampling geometry (sampling_loss.py:189-203)
    WITHOUT the mean.

    Driver-isolation params (D28), both defaulting to the D27 raw path so existing callers
    and the fidelity gate are unchanged:
      - match_color=True  applies CPO's color_match(pano, cloud_rgb) to the pano image before
        sampling (mirrors localize_pair; toggles the match_color factor of the a-vs-a' gap).
      - seed  reseeds np.random before read_txt_pcd's subsample draw (toggles the subsample
        factor). Default 0 == pin()'s seed == the D27 residuals.

    Deliberately excludes (by default) match_color/sharpen_color: those are upstream
    preprocessing steps applied by CALLERS (e.g. localize_pair, score_room_cheap) to `img`
    /`rgb` BEFORE handing them to sampling_loss/refine_pose_sampling_loss -- sampling_loss.py
    itself (lines 189-203) only consumes img/rgb as given, with no color adjustment. Composes
    CPO primitives only (no third_party edit, D9). Returns a 1-D numpy array.

    Fidelity note: an earlier version of this function additionally replicated the
    match_color/sharpen_color preprocessing block (mirroring score_room_cheap's pattern).
    That version FAILED the fidelity gate (|diff|=1.39e-02) against cpo.sampling_loss's own
    scalar, because the gate's reference L is computed from the raw (non-color-matched) img.
    A diagnostic isolating the two code paths confirmed: raw img -> |diff|=0.00e+00 exact
    match; match_color-preprocessed img -> |diff|~1.1e-2. This version was rewritten from
    scratch (per project error policy) to match only lines 189-203, which resolved most of it
    but left a residual |diff|=1.80e-03: data_utils.read_txt_pcd draws a np.random permutation
    whenever sample_rate>1 (see panopin.determinism's own docstring), so reloading the SAME
    cloud_path a second time (here) after the caller already loaded it once (to compute the
    reference pose/loss) draws a DIFFERENT random point subset, unless np.random is reseeded
    to the same state first. Reseeding to determinism.pin()'s default seed immediately before
    the read reproduces the caller's exact subsample (verified: array_equal == True) and
    closed the gap to |diff|=0.00e+00."""
    img, xyz, rgb, device = _load_for_residuals(cfg, pano_path, cloud_path, match_color, seed)
    return _residuals_at(cfg, img, xyz, rgb, device, t, R)


def _load_for_residuals(cfg, pano_path, cloud_path, match_color, seed):
    """The load half of residuals_at_pose (:160-176 before 2026-10-07): reseed, read the cloud
    (identical subsample to the caller's own read), read + resize the panorama."""
    from utils import cloud2idx, refine_sampling_coords, sample_from_img  # noqa: F401 (import check)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    sample_rate = getattr(cfg, 'sample_rate', 1)

    np.random.seed(seed)  # default 0 == pin()'s seed so a second, independent read_txt_pcd
    # reload of the same cloud draws the identical random subsample (sample_rate>1 branch) as
    # the caller's own first load; a different seed toggles the subsample factor (D28).
    xyz_np, rgb_np = data_utils.read_txt_pcd(cloud_path, sample_rate=sample_rate)
    xyz = torch.from_numpy(xyz_np).float().to(device)
    rgb = torch.from_numpy(rgb_np).float().to(device)

    orig_img = cv2.cvtColor(cv2.imread(pano_path), cv2.COLOR_BGR2RGB)
    orig_img = cv2.resize(orig_img, (2048, 1024))
    if match_color:                                  # D28: replicate localize_pair's color_match
        mod_img = (torch.from_numpy(orig_img).float() / 255.).to(device)
        new_img = color_match(mod_img, rgb)
        orig_img = (255 * new_img.detach().cpu().numpy()).astype(np.uint8)

    mdh = getattr(cfg, 'main_downsample_h', 1); mdw = getattr(cfg, 'main_downsample_w', 1)
    img = cv2.resize(orig_img, (orig_img.shape[1] // mdw, orig_img.shape[0] // mdh))
    img = (torch.from_numpy(img).float() / 255.).to(device)
    return img, xyz, rgb, device


def _residuals_at(cfg, img, xyz, rgb, device, t, R):
    """The per-pose half (:178-197 before 2026-10-07): sampling_loss.py:189-203 without the mean.
    Draws no randomness, so one reseed + load in _load_for_residuals serves every pose."""
    from utils import cloud2idx, refine_sampling_coords, sample_from_img
    t_col = torch.as_tensor(np.asarray(t, dtype=np.float32), device=device).reshape(3, 1)
    R_t = torch.as_tensor(np.asarray(R, dtype=np.float32), device=device).reshape(3, 3)

    new_xyz = torch.transpose(xyz, 0, 1) - t_col
    new_xyz = torch.transpose(torch.matmul(R_t, new_xyz), 0, 1)
    coord_arr = cloud2idx(new_xyz)
    filter_factor = getattr(cfg, 'filter_factor', 1)
    filtered_idx = refine_sampling_coords(
        coord_arr, torch.norm(new_xyz, dim=-1), rgb,
        quantization=(img.shape[0] // filter_factor, img.shape[1] // filter_factor))
    coord_arr = coord_arr[filtered_idx]
    refined_rgb = rgb[filtered_idx]
    sample_rgb = sample_from_img(img, coord_arr)
    mask = torch.sum(sample_rgb == 0, dim=1) != 3
    residuals = torch.norm(sample_rgb[mask] - refined_rgb[mask], dim=-1)
    return residuals.detach().cpu().numpy()


def residuals_at_poses(cfg, pano_path, cloud_path, poses, match_color=False, seed=0):
    """residuals_at_pose for several (t, R) at once: the cloud and the panorama are loaded ONCE
    (same reseed, same subsample, same resize), then the per-pose geometry runs per pose on
    the same tensors. Element i equals residuals_at_pose(cfg, pano_path, cloud_path, *poses[i],
    match_color=match_color, seed=seed) exactly (tests/test_cpo_adapter.py). The arbitration
    scores 10-20 candidates per panorama; reloading a 1 M-point cloud per candidate is what
    this avoids."""
    if not poses:
        return []
    img, xyz, rgb, device = _load_for_residuals(cfg, pano_path, cloud_path, match_color, seed)
    return [_residuals_at(cfg, img, xyz, rgb, device, t, R) for t, R in poses]
