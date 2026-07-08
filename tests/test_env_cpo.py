def test_cpo_primitives_import():
    import _cpo_path  # noqa: F401  (adds third_party/cpo to sys.path)
    import color_utils, utils, data_utils, dict_utils
    from cpo.sampling_loss import refine_pose_sampling_loss
    for name in ("read_txt_pcd",):
        assert hasattr(data_utils, name)
    for name in ("histogram_pose_search", "make_pano", "make_score_map_2d", "make_score_map_3d"):
        assert hasattr(utils, name)
    assert callable(refine_pose_sampling_loss)

def test_torch_cpu_op():
    import torch
    x = torch.ones(3) + torch.ones(3)
    assert x.sum().item() == 6.0
