import numpy as np, tests.synthetic as S


def _prep(tmp_path):
    xyz, rgb = S.box_room(walls={'x1': [220, 40, 40]})
    cloud = str(tmp_path / "room.txt"); S.write_cloud_txt(cloud, xyz, rgb)
    pano = str(tmp_path / "q.png"); S.render_pano_png(pano, xyz, rgb, [2.0, 2.0, 1.5], np.eye(3))
    return pano, cloud


def test_score_room_cheap_shapes_and_finite(tmp_path):
    from panopin.cpo_config import load_cfg
    from panopin.cpo_adapter import score_room_cheap
    pano, cloud = _prep(tmp_path)
    cfg = load_cfg(sample_rate=1, num_yaw=4, num_pitch=4, num_roll=4, num_trans=10)
    t, R, loss = score_room_cheap(cfg, pano, cloud)
    assert t.shape == (3,) and R.shape == (3, 3)
    assert np.isfinite(loss)


def test_cheap_scorer_skips_inlier_detection(tmp_path, monkeypatch):
    # score_room_cheap must NOT call the expensive make_score_map_* functions.
    from panopin.cpo_config import load_cfg
    from panopin.cpo_adapter import score_room_cheap
    def boom(*a, **k):
        raise AssertionError("inlier detection must not run in Tier-1")
    monkeypatch.setattr("panopin.cpo_adapter.make_score_map_2d", boom)
    monkeypatch.setattr("panopin.cpo_adapter.make_score_map_3d", boom)
    pano, cloud = _prep(tmp_path)
    cfg = load_cfg(sample_rate=1, num_yaw=4, num_pitch=4, num_roll=4, num_trans=10)
    score_room_cheap(cfg, pano, cloud)   # must not raise
