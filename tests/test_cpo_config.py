"""Tests for panopin.cpo_config: builds CPO's cfg namedtuple with immutable overrides."""


def test_load_cfg_defaults():
    from panopin.cpo_config import load_cfg
    cfg = load_cfg()
    assert cfg.dataset == "stanford"
    assert hasattr(cfg, "num_split_h") and hasattr(cfg, "num_iter")


def test_load_cfg_override_is_immutable_replace():
    from panopin.cpo_config import load_cfg, TIER1
    cfg = load_cfg(**TIER1)
    assert cfg.num_iter == 1
    assert cfg.top_k_candidate == 1


def test_tier2_overrides_num_iter_only():
    from panopin.cpo_config import load_cfg, TIER2
    cfg = load_cfg(**TIER2)
    assert cfg.num_iter == 100


def test_load_cfg_no_overrides_does_not_mutate_defaults():
    # Calling load_cfg with overrides must not leak into a later default call.
    from panopin.cpo_config import load_cfg, TIER1
    load_cfg(**TIER1)
    cfg = load_cfg()
    assert cfg.num_iter == 100  # stanford_cpo.ini default, unaffected by prior override
    assert cfg.top_k_candidate == 6  # stanford_cpo.ini default
