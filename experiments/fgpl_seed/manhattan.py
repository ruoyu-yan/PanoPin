"""Strictly-Manhattan demo pool (2026-07-16).

The Scan2BIM pipeline assumes a Manhattan-world building — Stage 3's clusterer hard-codes
`vote_principal_directions(n_directions=3)` (exactly 3 orthogonal wall directions) — so rooms with
a genuine diagonal wall are OUT OF SCOPE. Point_360 `roadmap.md` §5 classified all 23 Area_3 rooms
with a ceiling-band wall-normal test: 4 rooms are non-Manhattan (39-47% off-grid), the rest <=9%.

This pool confines BOTH halves of the experiment to Manhattan: the candidate rooms PanoPin scores
against, and (via `pool_clouds`) the room clouds the FGPL map is built from.

NOTE on room choice: the D33 largeval pool contains 3 of the 4 non-Manhattan rooms in Area_3
(office_3/7/8) and the D25/D35 subset contains 2 (office_4/7) — so NEITHER validated pool is
strictly Manhattan, and their headline numbers cannot be quoted as Manhattan results. This pool =
largeval INTERSECT Manhattan, which keeps the cached `largeval_residuals.json` grids usable: each
(pano, room) localization is independent (`localize_pair` per room), so restricting the candidate
set offline is equivalent to having only ever searched these rooms (the D30 rescoring argument).
It stays diverse (office/hallway/lounge/conference/WC) and retains the real loss-sink (hallway_1).
"""
import os
import sys

_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, "..", ".."))

from experiments.fgpl_seed import largeval

# Point_360 roadmap.md §5 "MANHATTAN CLASSIFICATION" (2026-07-15), <=9% off-grid.
MANHATTAN_ROOMS = frozenset({
    "WC_1", "WC_2", "conferenceRoom_1",
    "hallway_1", "hallway_2", "hallway_3", "hallway_4", "hallway_5", "hallway_6",
    "lounge_1", "lounge_2",
    "office_1", "office_2", "office_5", "office_6", "office_9", "office_10",
})

# Real diagonal wall, 39-47% off-grid -> excluded by the Manhattan assumption.
NON_MANHATTAN_ROOMS = frozenset({"office_3", "office_4", "office_7", "office_8"})

# Sparse/indeterminate scans the roadmap says to avoid (not trusted either way).
AVOID_ROOMS = frozenset({"storage_1", "storage_2"})

# largeval INTERSECT Manhattan -> cached grids apply, no GPU needed for room assignment.
POOL_ROOMS = [r for r in largeval.POOL_ROOMS if r in MANHATTAN_ROOMS]

# Isolated FGPL namespace: the Manhattan map must NOT clobber work/linemap/ + work/clouds/
# (the 6-room ablation map the cached oracle/p1/D35 arms were produced against).
SCENE = "area3_manhattan"
LINEMAP_SUBDIR = "linemap_manhattan"
ARM = "manhattan_export"


def build_pool():
    """Manhattan-only rows, reusing largeval's validated in-frame pano filter (D17)."""
    return [r for r in largeval.build_pool() if r["room"] in MANHATTAN_ROOMS]


def pool_clouds():
    """{room: per-room S3DIS cloud .txt} restricted to Manhattan — the FGPL map source."""
    return {r: c for r, c in largeval.pool_clouds().items() if r in MANHATTAN_ROOMS}


if __name__ == "__main__":
    from collections import Counter
    rows = build_pool()
    c = Counter(r["room"] for r in rows)
    excluded = [r for r in largeval.POOL_ROOMS if r not in MANHATTAN_ROOMS]
    print(f"Manhattan pool: {len(rows)} in-frame panos, {len(POOL_ROOMS)} rooms")
    for room in POOL_ROOMS:
        print(f"  {room:18s} {c[room]}")
    print(f"excluded as non-Manhattan: {', '.join(excluded)}")
