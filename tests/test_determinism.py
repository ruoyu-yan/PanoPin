def test_pin_sets_single_thread_and_seeds():
    import torch, numpy as np
    from panopin.determinism import pin
    torch.set_num_threads(4)
    pin(0)
    assert torch.get_num_threads() == 1
    a = np.random.rand(3)
    pin(0)
    b = np.random.rand(3)
    assert (a == b).all()   # same seed -> same numpy draw


def test_localize_and_score_is_reproducible_across_runs(tmp_path):
    """The shipped library entry point (panopin.seed.localize_and_score -- reached
    both from panopin.cli.seed_from_clouds and panopin.seed.seed_rooms) must give
    IDENTICAL scores/poses for identical inputs on two independent invocations.

    Root cause of the regression this guards: data_utils.read_txt_pcd draws an
    np.random permutation whenever sample_rate>1 (the deployed sample_rate=30
    config always hits this branch). Nothing under src/panopin/ pinned np.random
    before this test was written, so each call to localize_and_score consumed
    whatever global RNG state its predecessor left behind -- two back-to-back
    calls with the SAME inputs drew DIFFERENT subsamples for at least the first
    cloud read, and so could disagree on scores/poses. This test deliberately
    does NOT call determinism.pin() itself: pinning must happen inside the
    library so every caller (not just callers that remember to pin) is protected.
    sample_rate=2 is enough to hit read_txt_pcd's permutation branch on these
    tiny synthetic clouds; two panos x two rooms is enough to exercise the loop.
    """
    import numpy as np
    import tests.synthetic as S
    from panopin.cpo_config import load_cfg, TIER1
    from panopin import seed as seedmod

    def make(name, wall_rgb):
        xyz, rgb = S.box_room(n=40000, walls={'x1': wall_rgb})
        cloud = str(tmp_path / f"{name}.txt")
        S.write_cloud_txt(cloud, xyz, rgb)
        pano = str(tmp_path / f"{name}_q.png")
        S.render_pano_png(pano, xyz, rgb, [2.0, 2.0, 1.5], np.eye(3))
        return pano, cloud

    pano_a, cloud_a = make("a", [220, 40, 40])
    pano_b, cloud_b = make("b", [40, 200, 60])
    panos = {"p1": pano_a, "p2": pano_b}
    clouds = {"roomA": cloud_a, "roomB": cloud_b}

    cfg = load_cfg(**TIER1, sample_rate=2, num_yaw=4, num_pitch=4, num_roll=4,
                   num_trans=10, inlier_num_trans=10, inlier_num_yaw=4,
                   inlier_num_pitch=4, inlier_num_roll=4)

    score_matrix_1, poses_1 = seedmod.localize_and_score(panos, clouds, cfg)
    score_matrix_2, poses_2 = seedmod.localize_and_score(panos, clouds, cfg)

    assert score_matrix_1 == score_matrix_2, (score_matrix_1, score_matrix_2)
    for pano_id in poses_1:
        for room in poses_1[pano_id]:
            t1, R1 = poses_1[pano_id][room]
            t2, R2 = poses_2[pano_id][room]
            assert np.array_equal(t1, t2), (pano_id, room, "t diverged")
            assert np.array_equal(R1, R2), (pano_id, room, "R diverged")
