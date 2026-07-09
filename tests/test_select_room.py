"""Unit tests for the two-tier funnel LOGIC.

We mock localize_pair (real CPO discrimination on flat synthetic rooms doesn't
work — DECISIONS D13/D15; it is validated on real data via
smoke/discriminate_one_pano.py). These tests pin the funnel's control flow
deterministically and always run (no CPO, no data).

Tier-1 calls carry cfg.num_iter == 1 (TIER1); Tier-2 calls carry cfg.num_iter == 100
(TIER2). load_cfg runs for real (fast); only localize_pair is faked.
"""
import numpy as np

import panopin.select_room as SR


def _fake_localizer(tier1_loss, calls):
    """Return a localize_pair stub keyed by cloud path; records (num_iter, cloud)."""
    def fake(cfg, pano, cloud):
        calls.append((cfg.num_iter, cloud))
        # Tier-2 refine improves loss a touch but preserves the ranking.
        loss = tier1_loss[cloud] if cfg.num_iter == 1 else tier1_loss[cloud] - 0.05
        return np.zeros(3), np.eye(3), float(loss)
    return fake


def test_funnel_picks_min_loss_room(monkeypatch):
    calls = []
    tier1_loss = {"a.txt": 0.5, "b.txt": 0.1, "c.txt": 0.9, "d.txt": 0.2}
    monkeypatch.setattr(SR, "localize_pair", _fake_localizer(tier1_loss, calls))
    rooms = {"A": "a.txt", "B": "b.txt", "C": "c.txt", "D": "d.txt"}
    res = SR.select_room("q.png", rooms, top_k=2, sample_rate=1)
    assert res.room == "B"
    assert res.t.shape == (3,) and res.R.shape == (3, 3)
    assert np.isfinite(res.loss)


def test_funnel_tiers_all_then_refines_top_k(monkeypatch):
    calls = []
    tier1_loss = {"a.txt": 0.5, "b.txt": 0.1, "c.txt": 0.9, "d.txt": 0.2}
    monkeypatch.setattr(SR, "localize_pair", _fake_localizer(tier1_loss, calls))
    rooms = {"A": "a.txt", "B": "b.txt", "C": "c.txt", "D": "d.txt"}
    SR.select_room("q.png", rooms, top_k=2, sample_rate=1)
    tier1_calls = [c for c in calls if c[0] == 1]
    tier2_calls = [c for c in calls if c[0] == 100]
    # Tier-1 scores every candidate; Tier-2 refines only the top-k=2 survivors (B, D).
    assert len(tier1_calls) == 4
    assert sorted(c[1] for c in tier2_calls) == ["b.txt", "d.txt"]


def test_top_k_larger_than_candidates_is_clamped(monkeypatch):
    calls = []
    tier1_loss = {"a.txt": 0.3, "b.txt": 0.1}
    monkeypatch.setattr(SR, "localize_pair", _fake_localizer(tier1_loss, calls))
    res = SR.select_room("q.png", {"A": "a.txt", "B": "b.txt"}, top_k=9, sample_rate=1)
    assert res.room == "B"
    assert len([c for c in calls if c[0] == 100]) == 2  # both refined, no crash


def test_empty_candidates_returns_none(monkeypatch):
    monkeypatch.setattr(SR, "localize_pair", _fake_localizer({}, []))
    assert SR.select_room("q.png", {}, top_k=5) is None
