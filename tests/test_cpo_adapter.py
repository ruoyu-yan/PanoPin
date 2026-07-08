import os, numpy as np, tests.synthetic as S

def _prep(tmp_path, walls, pose_trans, name="room"):
    xyz, rgb = S.box_room(walls=walls)
    cloud = str(tmp_path / f"{name}.txt"); S.write_cloud_txt(cloud, xyz, rgb)
    R = np.eye(3)
    pano = str(tmp_path / f"{name}_q.png"); S.render_pano_png(pano, xyz, rgb, pose_trans, R)
    return pano, cloud

def test_localize_pair_returns_shapes_and_finite_loss(tmp_path):
    from panopin.cpo_config import load_cfg, TIER2
    from panopin.cpo_adapter import localize_pair
    pano, cloud = _prep(tmp_path, walls={'x1': [220, 40, 40]}, pose_trans=[2.0, 2.0, 1.5])
    cfg = load_cfg(**TIER2, sample_rate=1)
    t, R, loss = localize_pair(cfg, pano, cloud)
    assert t.shape == (3,) and R.shape == (3, 3)
    assert np.isfinite(loss)

def test_matching_cloud_scores_lower_than_mismatched(tmp_path):
    """Room A and room B share the SAME global color palette (one red wall, one
    blue wall) but the colors sit on SWAPPED walls -- a purely spatial difference.

    This matters because CPO's match_color=True (stanford_cpo.ini default)
    CDF-normalizes the query pano toward each candidate cloud's GLOBAL color
    distribution before scoring, which partly erases a global-color difference
    between candidates. A same-palette / different-layout pair isolates the
    spatial signal that match_color cannot erase -- exactly PanoPin's real claim
    that same-shape, same-palette rooms are disambiguated by where content sits.
    """
    from panopin.cpo_config import load_cfg, TIER2
    from panopin.cpo_adapter import localize_pair
    walls_a = {'x1': [220, 40, 40], 'y0': [40, 40, 220]}   # x1 red, y0 blue
    walls_b = {'x1': [40, 40, 220], 'y0': [220, 40, 40]}   # swapped: x1 blue, y0 red
    # query pano is rendered from room A's own geometry/colors
    pano, match_cloud = _prep(tmp_path, walls=walls_a, pose_trans=[2.0, 2.0, 1.5], name="a")
    xyz_b, rgb_b = S.box_room(walls=walls_b)
    mis_cloud = str(tmp_path / "b.txt"); S.write_cloud_txt(mis_cloud, xyz_b, rgb_b)
    cfg = load_cfg(**TIER2, sample_rate=1)
    _, _, loss_match = localize_pair(cfg, pano, match_cloud)
    _, _, loss_mis = localize_pair(cfg, pano, mis_cloud)
    assert loss_match < loss_mis   # spatial layout discriminates even under global color matching
