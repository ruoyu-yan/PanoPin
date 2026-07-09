"""Two-tier CPO room selector (D11).

Tier-1: cheap CPO cost (num_iter=1, top_k_candidate=1) for every candidate room ->
keep the top-k lowest-loss rooms. Tier-2: full Adam refine (num_iter=100) on those
survivors -> return the global min-loss room + its coarse pose.

The ranking signal is CPO's colour-consistency `loss`; validated on real Area_3 data
(the correct room breaks well below the ~0.22 no-match floor — DECISIONS D14).

Reads only pano + candidate cloud paths (fairness, D5): no GT.

COST NOTE (D16): with stock CPO cfg, Tier-1 is NOT much cheaper than Tier-2 — the
fixed inlier score-map detection (~20 s/room on CPU) dominates both. The funnel is
correct but its speed-up is small until Tier-1 gets a lighter scorer; see DECISIONS D16
and the Task-7 cost work.
"""
from collections import namedtuple

from panopin.cpo_config import load_cfg, TIER1, TIER2
from panopin.cpo_adapter import localize_pair

RoomResult = namedtuple("RoomResult", "room t R loss")


def select_room(pano_path, candidate_rooms, top_k=5, sample_rate=10):
    """Pick the best room for `pano_path` among `candidate_rooms` {room: cloud_path}.

    Returns RoomResult(room, t, R, loss) for the min-loss room, or None if no
    candidates were given.
    """
    if not candidate_rooms:
        return None
    cfg1 = load_cfg(**TIER1, sample_rate=sample_rate)
    cfg2 = load_cfg(**TIER2, sample_rate=sample_rate)

    # Tier 1: cheap cost for every candidate room.
    tier1 = []
    for room, cloud in candidate_rooms.items():
        _, _, loss = localize_pair(cfg1, pano_path, cloud)
        tier1.append((loss, room, cloud))
    tier1.sort(key=lambda x: x[0])
    survivors = tier1[:max(1, min(top_k, len(tier1)))]

    # Tier 2: full refine on survivors; pick the global min loss.
    best = None
    for _, room, cloud in survivors:
        t, R, loss = localize_pair(cfg2, pano_path, cloud)
        if best is None or loss < best.loss:
            best = RoomResult(room, t, R, loss)
    return best
