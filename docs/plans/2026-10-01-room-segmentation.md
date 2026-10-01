# Automatic Room Segmentation (T11) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** a new `panopin.cli segment` turns ONE merged multi-room cloud into the `{room: cloud}`
candidates that `panopin.cli seed` already takes. A harness proves bar (b) against the S3DIS partition.

**Architecture:**
- Rasterise a band just below the ceiling at 5 cm. Its occupied cells are walls, including the lintels
  that close open doors.
- Free-space connected components inside the building footprint are the rooms. Thin or tiny ones are
  dropped, and every remaining footprint cell joins its nearest room.
- Every 3D point takes its cell's room.
- A separate numpy/scipy scorer compares the result against S3DIS room membership. A two-arm Stage-0
  run compares PanoPin + FGPL on predicted vs ground-truth room clouds.

**Tech Stack:** Python in the `panopin` conda env (numpy 1.23.5, scipy 1.10.1, OpenCV 5.0.0,
open3d 0.19, pandas). pytest. **No new packages.**

**Spec:** `docs/specs/2026-10-01-room-segmentation-design.md`. Read it first. Background:
`docs/room-segmentation-literature-review.md` §10.1.

## Global Constraints

**Environment and imports**
- Every Python command runs as `cd /home/ruoyu/PanoPin && conda run -n panopin python …`. The
  exceptions are the two Point_360 scripts in Task 9, which name their own env.
- Import the new code only as `panopin.roomseg…`. `tests/conftest.py` also puts `src/panopin` on
  `sys.path`, and a bare `import roomseg` would create a second module copy.

**Geometry and determinism**
- Units are metres. Gravity is +z. One storey per cloud. Output points stay in the **input
  coordinates**; no rotation, no recentring.
- Deterministic: no random numbers except the fixed `RandomState(7)` palette of `rooms.png`.
- Segment files use the format `%.6f %.6f %.6f %d %d %d` (X Y Z R G B, colours 0–255). This is what
  `third_party/cpo/data_utils.read_txt_pcd` reads.

**Do not touch**
- `panopin.cli seed`'s behaviour.
- `eval/score.py`, `eval/metrics.py`, and any S3DIS ground-truth file.

**Fairness (D5)**
- The segmenter reads only the merged cloud: never room `.txt` files, `Area_3/1.e57`, pose files or
  room names.
- Ground truth is used only by `eval/roomseg_*.py` and `experiments/roomseg/two_arm.py`.

**Tuning**
- Parameter values change only on the dev scenes (`Area_3_manhattan4`, `Area_3_manhattan6`).
- Each change is logged in `docs/roomseg-results.md` with dev metrics before and after.
- Holdout (`Area_2_manhattan4`, `Area_2_manhattan7`) is scored **once**, after freezing at a recorded
  commit.
- Any change to the *algorithm* (not a value) goes back to the user first (spec §10).

**Errors (user's global rule)**
- After writing or editing any Python file, run it or its tests.
- An *unexpected* error means: undo that change completely and rewrite that section a different way.
  Never stack patches.
- If the rewrite also fails, stop and explain to the user.
- The deliberate red step of TDD is not an error.

**Git**
- Commit on branch `feat/room-segmentation`, with messages ending
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- **Never push.**
- Point_360 files touched in Task 9 are untracked data scripts; do not commit anything in Point_360.

## Review Focus

1. **Cloud in millimetres** (the thesis front-end used to accept mm): `SegmentationError` whose message
   says "metres". Not a silent nonsense segmentation. Test in Task 1.
2. **Stray z outliers** far above the ceiling or below the floor (reflections, noise): the floor and
   ceiling heights are unchanged. Test in Task 1.
3. **A stray outlier point hundreds of metres away in XY:** `SegmentationError` naming the extent.
   Not a `MemoryError` from a gigantic grid. Test in Task 2.
4. **Re-running `segment` into an out-dir that holds more `seg_*.txt` from an earlier run:** stale
   files are removed, so the directory matches `clouds.json`. Test in Task 5.
5. **A `.ply` without colours, or any refused cloud:** non-zero exit with a clear message, and
   **nothing** written to the out-dir. Test in Task 5.

---

## File map

| File | Responsibility |
|---|---|
| `src/panopin/roomseg/__init__.py` | re-exports `segment`, `SegmentationError`, `Params`, `DEFAULTS`, `HOVSG` |
| `src/panopin/roomseg/errors.py` | `SegmentationError` |
| `src/panopin/roomseg/params.py` | frozen `Params` dataclass, `DEFAULTS`, `HOVSG` (baseline B1) |
| `src/panopin/roomseg/levels.py` | `storey_heights(z, p)` |
| `src/panopin/roomseg/raster.py` | `Grid`, `count_image`, `band_image`, `footprint`, `disk`, `close` |
| `src/panopin/roomseg/partition.py` | `wall_mask`, `rooms`, `fill` |
| `src/panopin/roomseg/lift.py` | `lift` |
| `src/panopin/roomseg/core.py` | `segment(xyz, p, n_panos)` — orchestration, canonical order, report, warnings |
| `src/panopin/roomseg/files.py` | `read_cloud`, `write_outputs`, `segment_cloud` |
| `src/panopin/cli.py` (modify) | new `segment` subcommand |
| `tests/roomseg_synth.py` | synthetic indoor clouds |
| `tests/test_roomseg_{levels,raster,partition,core,cli}.py` | unit tests |
| `config/roomseg_scenes.json` | the four scenes |
| `eval/roomseg_gt.py` | per-point GT room labels |
| `eval/roomseg_score.py` | metrics + containment + report |
| `tests/test_roomseg_score.py` | scorer unit tests |
| `experiments/roomseg/area3_stage0_selection.py` | Area_3 dev scenes' one-pano-per-room selection for Stage 0 |
| `experiments/roomseg/two_arm.py` | bar (b) end-to-end, two arms |
| `docs/roomseg-results.md` | dev / holdout / end-to-end results and every tuning change |

---

### Task 1: Scaffold, parameters, synthetic scenes, storey heights

**Files:**
- Create: `src/panopin/roomseg/__init__.py`, `src/panopin/roomseg/errors.py`,
  `src/panopin/roomseg/params.py`, `src/panopin/roomseg/levels.py`
- Create: `tests/roomseg_synth.py`, `tests/test_roomseg_levels.py`
- Modify: `docs/specs/2026-10-01-room-segmentation-design.md` (§4 step 1, §5) — the measured amendment

**Interfaces:**
- Produces:
  - `Params` (frozen dataclass, fields below), `DEFAULTS = Params()`,
    `HOVSG = Params(band_from_floor=1.5, band_top=0.30)`;
  - `SegmentationError(RuntimeError)`;
  - `storey_heights(z: np.ndarray, p: Params = DEFAULTS) -> tuple[float, float, list[tuple[float, int]]]`,
    returning `(floor_z, ceiling_z, strong_peaks)`;
  - synthetic builders `shell`, `wall`, `plane_z`, `box`, `thick_partition_x`, `two_rooms`, `colour`,
    and constants `SP = 0.03`, `H = 2.8`.

- [ ] **Step 1: Create the package skeleton and parameters**

`src/panopin/roomseg/errors.py`:
```python
"""The one exception the room segmenter raises."""


class SegmentationError(RuntimeError):
    """The cloud cannot be segmented as asked; the message says why and what to check."""
```

`src/panopin/roomseg/params.py`:
```python
"""Every constant of the enclosure room segmenter, in metres (spec 2026-10-01 §4).

Values change ONLY on the dev scenes, and every change is logged in docs/roomseg-results.md.
"""
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Params:
    # storey heights (levels.py)
    z_bin: float = 0.02               # z-histogram bin
    peak_rel: float = 0.30            # a local z maximum is "strong" at >= this x the tallest bin
    peak_cluster_gap: float = 0.30    # strong peaks this close belong to one slab
    min_storey: float = 2.0
    max_storey: float = 6.0
    mid_slab_above_floor: float = 1.8     # a strong peak in (floor + 1.8, ceiling - 0.5)
    mid_slab_below_ceiling: float = 0.5   #   means a second storey -> refuse
    # grid and band (raster.py)
    cell: float = 0.05
    band_bottom: float = 0.60         # band = [ceiling - band_bottom, ceiling - band_top]
    band_top: float = 0.10
    band_from_floor: Optional[float] = None   # if set, band lower edge = floor + this (B1)
    max_grid_cells: int = 40_000_000  # ~316 m square at 5 cm; beyond this the extent is outliers
    footprint_close: float = 0.30
    # rooms (partition.py)
    wall_min_pts: int = 3
    gap_close_r: float = 0.15         # seals scan gaps up to ~0.3 m; must stay < half a door width
    min_room_area: float = 1.0
    min_room_halfwidth: float = 0.30
    max_fill_dist: float = 1.0
    # report warnings (core.py)
    warn_unassigned_frac: float = 0.02
    warn_small_room_area: float = 2.0


DEFAULTS = Params()
# Baseline B1 (spec §6.3 as amended in Task 7): the HOV-SG band [floor + 1.5, ceiling - 0.3].
HOVSG = Params(band_from_floor=1.5, band_top=0.30)
```

`src/panopin/roomseg/__init__.py` (Task 4 extends it):
```python
"""Automatic room segmentation of one merged, z-up, single-storey cloud (spec 2026-10-01)."""
from .errors import SegmentationError
from .params import DEFAULTS, HOVSG, Params

__all__ = ["SegmentationError", "Params", "DEFAULTS", "HOVSG"]
```

- [ ] **Step 2: Write the synthetic scene builders**

`tests/roomseg_synth.py`:
```python
"""Synthetic indoor clouds for the room-segmentation tests: planes sampled on a regular 3 cm grid."""
import numpy as np

SP = 0.03   # sample spacing (m)
H = 2.8     # storey height (m)


def _grid(a0, a1, b0, b1):
    a, b = np.meshgrid(np.arange(a0, a1, SP), np.arange(b0, b1, SP))
    return a.ravel(), b.ravel()


def plane_z(z, x0, x1, y0, y1):
    x, y = _grid(x0, x1, y0, y1)
    return np.column_stack([x, y, np.full(x.size, z)])


def wall(axis, at, a0, a1, h=H, openings=()):
    """Vertical wall on x=at (axis='x') or y=at (axis='y'), spanning a0..a1 along the other axis.

    openings: (b0, b1, top) gaps along the wall. top=None is open floor to ceiling; otherwise it is a
    door whose lintel fills top..h.
    """
    cuts = sorted(openings)
    edges = [a0] + [v for b0, b1, _ in cuts for v in (b0, b1)] + [a1]
    pieces = [_grid(s0, s1, 0.0, h) for s0, s1 in zip(edges[0::2], edges[1::2]) if s1 > s0]
    pieces += [_grid(b0, b1, top, h) for b0, b1, top in cuts if top is not None]
    if not pieces:                       # the opening spans the whole wall
        return np.zeros((0, 3))
    s = np.concatenate([p[0] for p in pieces])
    z = np.concatenate([p[1] for p in pieces])
    if axis == "x":
        return np.column_stack([np.full(s.size, at), s, z])
    return np.column_stack([s, np.full(s.size, at), z])


def box(x0, x1, y0, y1, top):
    """Furniture: top face and four sides of a box standing on the floor."""
    return np.concatenate([plane_z(top, x0, x1, y0, y1),
                           wall("x", x0, y0, y1, h=top), wall("x", x1, y0, y1, h=top),
                           wall("y", y0, x0, x1, h=top), wall("y", y1, x0, x1, h=top)])


def shell(x1, y1, h=H, ceiling=True):
    """Floor, (ceiling,) and four outer walls of the rectangle [0, x1] x [0, y1]."""
    parts = [plane_z(0.0, 0, x1, 0, y1),
             wall("x", 0.0, 0, y1, h), wall("x", x1, 0, y1, h),
             wall("y", 0.0, 0, x1, h), wall("y", y1, 0, x1, h)]
    if ceiling:
        parts.append(plane_z(h, 0, x1, 0, y1))
    return np.concatenate(parts)


def thick_partition_x(at, y1, thick=0.2, openings=(), h=H):
    """The two faces of an interior wall centred on x=at, sharing the same openings."""
    return np.concatenate([wall("x", at - thick / 2, 0, y1, h, openings),
                           wall("x", at + thick / 2, 0, y1, h, openings)])


def two_rooms(door_top=2.1, gap=(1.5, 2.5), split=4.0, ceiling=True):
    """8 x 4 m split at x=split by a 0.2 m wall with one opening (door with lintel, or open)."""
    return np.concatenate([shell(8.0, 4.0, ceiling=ceiling),
                           thick_partition_x(split, 4.0, openings=[(gap[0], gap[1], door_top)])])


def colour(xyz):
    """Deterministic RGB (uint8) for a synthetic cloud."""
    return (np.abs(np.sin(np.asarray(xyz) * 7.0)) * 255).astype(np.uint8)
```

- [ ] **Step 3: Write the failing tests**

`tests/test_roomseg_levels.py`:
```python
import numpy as np
import pytest

from panopin.roomseg import DEFAULTS, SegmentationError
from panopin.roomseg.levels import storey_heights
from tests.roomseg_synth import H, two_rooms


def test_floor_and_ceiling_of_a_single_storey():
    floor, ceiling, peaks = storey_heights(two_rooms()[:, 2])
    assert abs(floor - 0.0) <= DEFAULTS.z_bin
    assert abs(ceiling - H) <= DEFAULTS.z_bin
    assert len(peaks) >= 2


def test_ceiling_is_the_lowest_peak_of_the_top_cluster():
    """Area_2 ceilings sit at 2.58-2.81 m: the band must stay below the LOWEST of them."""
    xyz = two_rooms()
    lower = xyz[xyz[:, 0] < 4.0].copy()
    lower = lower[np.abs(lower[:, 2] - H) < 1e-9]
    lower[:, 2] = H - 0.2                      # a second, lower ceiling level 0.2 m down
    _, ceiling, _ = storey_heights(np.concatenate([xyz[:, 2], lower[:, 2]]))
    assert abs(ceiling - (H - 0.2)) <= DEFAULTS.z_bin


def test_z_outliers_do_not_move_the_heights():
    z = two_rooms()[:, 2]
    noisy = np.concatenate([z, np.full(20, -8.0), np.full(20, 12.0)])
    # bin edges start at z.min(), so the outliers may shift a peak by one bin -- no more
    assert storey_heights(noisy)[:2] == pytest.approx(storey_heights(z)[:2], abs=DEFAULTS.z_bin)


def test_millimetre_cloud_is_refused_and_says_metres():
    with pytest.raises(SegmentationError, match="metres"):
        storey_heights(two_rooms()[:, 2] * 1000.0)


def test_two_storeys_are_refused():
    xyz = two_rooms()
    upper = xyz + np.array([0.0, 0.0, H + 0.2])
    with pytest.raises(SegmentationError, match="storey"):
        storey_heights(np.concatenate([xyz, upper])[:, 2])


def test_missing_ceiling_is_refused():
    with pytest.raises(SegmentationError, match="floor and ceiling"):
        storey_heights(two_rooms(ceiling=False)[:, 2])


def test_empty_cloud_is_refused():
    with pytest.raises(SegmentationError, match="empty"):
        storey_heights(np.zeros(0))
```

- [ ] **Step 4: Run the tests to see them fail**

Run: `cd /home/ruoyu/PanoPin && conda run -n panopin python -m pytest tests/test_roomseg_levels.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'panopin.roomseg.levels'`.

- [ ] **Step 5: Implement `levels.py`**

`src/panopin/roomseg/levels.py`:
```python
"""Storey heights from the z histogram (spec §4 step 1, as amended 2026-10-01).

floor   = strongest peak of the LOWEST cluster of strong peaks.
ceiling = LOWEST peak of the HIGHEST cluster. Area_2's ceilings sit at 2.58-2.81 m, and the band
          must stay below every one of them, or a lower ceiling fills the band.
A strong peak between floor + 1.8 m and ceiling - 0.5 m is another storey's slab, so refuse.
Desk tops (~0.75 m, up to 0.38 x the tallest bin on Area_2) and lintels (~2.1 m, ~0.15 x) stay
outside that window or below the strength threshold.
"""
import numpy as np

from .errors import SegmentationError
from .params import DEFAULTS


def _strong_peaks(z, p):
    edges = np.arange(z.min(), z.max() + 2 * p.z_bin, p.z_bin)
    h, edges = np.histogram(z, bins=edges)
    centres = edges[:-1] + p.z_bin / 2
    padded = np.concatenate([[-1], h, [-1]])
    is_max = (padded[1:-1] >= padded[:-2]) & (padded[1:-1] >= padded[2:])
    strong = is_max & (h >= p.peak_rel * h.max())
    return centres[strong], h[strong]


def _clusters(zs, gap):
    groups = []
    for i, z in enumerate(zs):
        if groups and z - zs[groups[-1][-1]] <= gap:
            groups[-1].append(i)
        else:
            groups.append([i])
    return groups


def storey_heights(z, p=DEFAULTS):
    """(floor_z, ceiling_z, strong_peaks) of a single-storey, z-up cloud in metres."""
    z = np.asarray(z, dtype=float)
    if z.size == 0:
        raise SegmentationError("empty cloud")
    zs, hs = _strong_peaks(z, p)
    peaks = [(round(float(a), 3), int(b)) for a, b in zip(zs, hs)]
    groups = _clusters(zs, p.peak_cluster_gap)
    if len(groups) < 2:
        raise SegmentationError(
            f"no clear floor and ceiling in the z histogram (strong peaks {peaks}); "
            "is the cloud z-up and does it include the ceiling?")
    lo, hi = groups[0], groups[-1]
    floor_z = float(zs[lo][int(np.argmax(hs[lo]))])
    ceiling_z = float(zs[hi].min())
    storey = ceiling_z - floor_z
    if not p.min_storey <= storey <= p.max_storey:
        raise SegmentationError(
            f"storey height {storey:.2f} outside [{p.min_storey}, {p.max_storey}]: "
            "is the cloud in metres, z-up and a single storey?")
    mid = [round(float(v), 2) for v in zs
           if floor_z + p.mid_slab_above_floor < v < ceiling_z - p.mid_slab_below_ceiling]
    if mid:
        raise SegmentationError(
            f"strong horizontal slab at z={mid} between floor {floor_z:.2f} and ceiling "
            f"{ceiling_z:.2f}: more than one storey? Split the cloud by storey first")
    return floor_z, ceiling_z, peaks
```

- [ ] **Step 6: Run the tests to see them pass**

Run: `cd /home/ruoyu/PanoPin && conda run -n panopin python -m pytest tests/test_roomseg_levels.py -q`
Expected: `7 passed`.

- [ ] **Step 7: Amend the spec with the measured rule**

In `docs/specs/2026-10-01-room-segmentation-design.md` §4, replace the row
`| 1 Heights | z histogram, 2 cm bins. Floor = strongest bin in the lower half of the z range; ceiling = strongest bin in the upper half | `z_bin` 0.02 m |`
with:
```
| 1 Heights | z histogram, 2 cm bins; strong peaks = local maxima ≥ 0.3 × the tallest bin, clustered when ≤ 0.3 m apart. Floor = strongest peak of the lowest cluster; ceiling = **lowest** peak of the highest cluster. *Amended 2026-10-01 (plan Task 1): Area_2's ceilings sit at 2.58–2.81 m, so the band must sit below the lowest of them; Area_2_manhattan4's desk peak (0.77 m) reaches 0.38 × the tallest bin.* | `z_bin` 0.02, `peak_rel` 0.30, `peak_cluster_gap` 0.30 |
```
In §5's fail-loudly list, add the bullet:
`- a strong horizontal slab between floor + 1.8 m and ceiling − 0.5 m (another storey).`

- [ ] **Step 8: Commit**

```bash
cd /home/ruoyu/PanoPin && git add src/panopin/roomseg tests/roomseg_synth.py tests/test_roomseg_levels.py docs/specs/2026-10-01-room-segmentation-design.md
git commit -m "feat(roomseg): parameters, synthetic scenes and storey-height detection

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Metric grid and images

**Files:**
- Create: `src/panopin/roomseg/raster.py`
- Test: `tests/test_roomseg_raster.py`

**Interfaces:**
- Consumes: `SegmentationError`, `DEFAULTS` (Task 1).
- Produces:
  - `Grid(x0, y0, cell, ny, nx)`, a frozen dataclass:
    - `Grid.fit(xy, cell, max_cells) -> Grid`;
    - `.cells(xy) -> (row: int64[N], col: int64[N])`;
    - `.world(row, col) -> float[N, 2]` (cell centres);
  - `count_image(grid, xy) -> int32[ny, nx]`;
  - `band_image(grid, xyz, z0, z1) -> int32[ny, nx]`;
  - `footprint(grid, xyz, close_m) -> bool[ny, nx]`;
  - `disk(r_cells) -> bool[2r+1, 2r+1]`;
  - `close(mask, r_cells) -> bool` (border-exact binary closing).

- [ ] **Step 1: Write the failing tests**

`tests/test_roomseg_raster.py`:
```python
import numpy as np
import pytest

from panopin.roomseg import DEFAULTS, SegmentationError
from panopin.roomseg.raster import Grid, band_image, close, count_image, footprint
from tests.roomseg_synth import two_rooms


def test_cells_and_world_round_trip():
    xyz = two_rooms()
    g = Grid.fit(xyz[:, :2], 0.05, DEFAULTS.max_grid_cells)
    row, col = g.cells(xyz[:, :2])
    back = g.world(row, col)
    assert np.abs(back - xyz[:, :2]).max() <= 0.05 * np.sqrt(2) / 2 + 1e-9
    assert row.min() >= 1 and col.min() >= 1          # PAD cells on every side (float-safe)


def test_count_image_counts_every_point():
    xyz = two_rooms()
    g = Grid.fit(xyz[:, :2], 0.05, DEFAULTS.max_grid_cells)
    assert count_image(g, xyz[:, :2]).sum() == len(xyz)


def test_band_image_keeps_only_the_band():
    xyz = two_rooms()
    g = Grid.fit(xyz[:, :2], 0.05, DEFAULTS.max_grid_cells)
    inside = (xyz[:, 2] >= 2.2) & (xyz[:, 2] <= 2.7)
    assert band_image(g, xyz, 2.2, 2.7).sum() == inside.sum()


def test_far_outlier_is_refused_with_the_extent_not_a_memory_error():
    xyz = np.concatenate([two_rooms(), [[500.0, 500.0, 1.0]]])
    with pytest.raises(SegmentationError, match="extent"):
        Grid.fit(xyz[:, :2], 0.05, DEFAULTS.max_grid_cells)


def test_close_seals_narrow_gaps_between_blobs_only():
    """close() is for the footprint (thick blobs). It cannot bridge a gap in a 1-cell line,
    which is why walls are grown instead (Task 3)."""
    m = np.zeros((40, 60), bool)
    m[10:30, 5:20] = True; m[10:30, 24:40] = True    # 4-cell gap
    m[10:30, 49:60] = True                            # 9-cell gap; touches the border
    c = close(m, 3)
    assert c[20, 20:24].all()                         # 4-cell gap sealed (<= 2r)
    assert not c[20, 41:48].any()                     # 9-cell gap kept
    assert c[10:30, 49:60].all()                      # border blob intact


def test_footprint_covers_the_building_and_not_outside():
    xyz = two_rooms()
    g = Grid.fit(xyz[:, :2], 0.05, DEFAULTS.max_grid_cells)
    fp = footprint(g, xyz, 0.30)
    r, c = g.cells(np.array([[2.0, 2.0], [6.0, 2.0]]))
    assert fp[r, c].all()
    assert not fp[0, 0]
```

- [ ] **Step 2: Run to see them fail**

Run: `cd /home/ruoyu/PanoPin && conda run -n panopin python -m pytest tests/test_roomseg_raster.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'panopin.roomseg.raster'`.

- [ ] **Step 3: Implement `raster.py`**

`src/panopin/roomseg/raster.py`:
```python
"""Metric 2D grid over the cloud's XY extent, and the images built on it (spec §3, §4)."""
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from .errors import SegmentationError

PAD = 2   # empty cells around the extent


@dataclass(frozen=True)
class Grid:
    x0: float
    y0: float
    cell: float
    ny: int
    nx: int

    @classmethod
    def fit(cls, xy, cell, max_cells):
        lo = xy.min(axis=0) - PAD * cell
        hi = xy.max(axis=0) + PAD * cell
        nx = int(np.floor((hi[0] - lo[0]) / cell)) + 1
        ny = int(np.floor((hi[1] - lo[1]) / cell)) + 1
        if nx * ny > max_cells:
            raise SegmentationError(
                f"XY extent {hi[0] - lo[0]:.0f} x {hi[1] - lo[1]:.0f} m needs {nx * ny} cells at "
                f"{cell} m (> {max_cells}); remove far outlier points first")
        return cls(float(lo[0]), float(lo[1]), float(cell), ny, nx)

    def cells(self, xy):
        """(row, col) of the cells holding each xy, clipped to the grid."""
        col = np.clip(np.floor((xy[:, 0] - self.x0) / self.cell).astype(np.int64), 0, self.nx - 1)
        row = np.clip(np.floor((xy[:, 1] - self.y0) / self.cell).astype(np.int64), 0, self.ny - 1)
        return row, col

    def world(self, row, col):
        """XY of the cell centres."""
        return np.stack([self.x0 + (np.asarray(col) + 0.5) * self.cell,
                         self.y0 + (np.asarray(row) + 0.5) * self.cell], axis=1)


def count_image(grid, xy):
    row, col = grid.cells(xy)
    flat = np.bincount(row * grid.nx + col, minlength=grid.ny * grid.nx)
    return flat.reshape(grid.ny, grid.nx).astype(np.int32)


def band_image(grid, xyz, z0, z1):
    """Points per cell with z0 <= z <= z1."""
    keep = (xyz[:, 2] >= z0) & (xyz[:, 2] <= z1)
    return count_image(grid, xyz[keep, :2])


def disk(r_cells):
    r = int(r_cells)
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    return (xx * xx + yy * yy) <= r * r


def close(mask, r_cells):
    """Binary closing that is exact at the grid border (pad, close, crop)."""
    r = int(r_cells)
    if r < 1:
        return mask.copy()
    k = disk(r)
    big = np.pad(mask, r + 1)
    big = ndimage.binary_erosion(ndimage.binary_dilation(big, k), k, border_value=1)
    return big[r + 1:-(r + 1), r + 1:-(r + 1)]


def footprint(grid, xyz, close_m):
    """Cells inside the building: anything scanned at any height, holes filled, closed."""
    occ = ndimage.binary_fill_holes(count_image(grid, xyz[:, :2]) > 0)
    occ = close(occ, round(close_m / grid.cell))
    return ndimage.binary_fill_holes(occ)
```

- [ ] **Step 4: Run to see them pass**

Run: `cd /home/ruoyu/PanoPin && conda run -n panopin python -m pytest tests/test_roomseg_raster.py -q`
Expected: `6 passed`.

- [ ] **Step 5: Commit**

```bash
cd /home/ruoyu/PanoPin && git add src/panopin/roomseg/raster.py tests/test_roomseg_raster.py
git commit -m "feat(roomseg): metric grid, band image, footprint, border-exact closing

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Walls → rooms → fill

**Files:**
- Create: `src/panopin/roomseg/partition.py`
- Test: `tests/test_roomseg_partition.py`

**Interfaces:**
- Consumes: `Grid`, `band_image`, `footprint`, `disk` (Task 2); `Params`, `DEFAULTS` (Task 1).
- Produces:
  - `wall_mask(band_counts: int32[ny,nx], cell: float, p: Params) -> bool[ny,nx]` (band walls
    dilated by `gap_close_r`);
  - `rooms(walls: bool, footprint: bool, cell: float, p: Params) -> int32[ny,nx]` (0 = none, 1..K in
    label-scan order);
  - `fill(labels: int32, footprint: bool, cell: float, p: Params) -> int32[ny,nx]` (every footprint cell
    within `max_fill_dist` of a room gets that room; others 0).

- [ ] **Step 1: Write the failing tests**

`tests/test_roomseg_partition.py`:
```python
import numpy as np

from panopin.roomseg import DEFAULTS
from panopin.roomseg.partition import fill, rooms, wall_mask
from panopin.roomseg.raster import Grid, band_image, footprint
from tests.roomseg_synth import box, shell, thick_partition_x, two_rooms

BAND = (2.2, 2.7)   # ceiling 2.8 - 0.6 .. - 0.1


def _rooms_of(xyz, p=DEFAULTS):
    g = Grid.fit(xyz[:, :2], p.cell, p.max_grid_cells)
    fp = footprint(g, xyz, p.footprint_close)
    walls = wall_mask(band_image(g, xyz, *BAND), g.cell, p)
    return g, fp, rooms(walls, fp, g.cell, p)


def test_door_under_a_lintel_separates_two_rooms():
    _, _, lab = _rooms_of(two_rooms(door_top=2.1))
    assert lab.max() == 2


def test_opening_up_to_the_ceiling_joins_them():
    """Documents the open-plan limit: no lintel, no wall in the band."""
    _, _, lab = _rooms_of(two_rooms(door_top=None, gap=(1.0, 3.0)))
    assert lab.max() == 1


def test_a_narrow_scan_hole_is_sealed():
    xyz = np.concatenate([shell(8.0, 4.0), thick_partition_x(4.0, 4.0, openings=[(2.0, 2.2, None)])])
    _, _, lab = _rooms_of(xyz)
    assert lab.max() == 2


def test_a_door_across_a_corridor_splits_it():
    xyz = np.concatenate([shell(10.0, 2.0), thick_partition_x(5.0, 2.0, openings=[(0.2, 1.8, 2.1)])])
    _, _, lab = _rooms_of(xyz)
    assert lab.max() == 2


def test_furniture_below_the_band_changes_nothing():
    plain = _rooms_of(two_rooms())[2]
    furnished = _rooms_of(np.concatenate([two_rooms(), box(0.5, 1.5, 0.5, 1.5, 0.8),
                                          box(5.0, 7.0, 3.0, 3.6, 2.0)]))[2]
    assert np.array_equal(plain, furnished)


def test_thin_slivers_are_dropped():
    walls = np.zeros((60, 60), bool)
    walls[:, 30] = True; walls[:, 39] = True           # 8-cell (0.40 m) channel between two walls
    grown = wall_mask(walls.astype(np.int32) * 10, 0.05, DEFAULTS)   # 2 cells (0.10 m) remain free
    lab = rooms(grown, np.ones_like(walls), 0.05, DEFAULTS)
    assert lab.max() == 2                               # left and right rooms only
    assert lab[:, 31:39].max() == 0                     # the sliver is dropped


def test_fill_gives_wall_cells_their_nearest_room_and_caps_distance():
    g, fp, lab = _rooms_of(two_rooms())
    filled = fill(lab, fp, g.cell, DEFAULTS)
    r, c = g.cells(np.array([[3.9, 3.0], [4.1, 3.0], [2.0, 2.0], [6.0, 2.0]]))
    left_face, right_face, left, right = filled[r, c]
    assert left_face == left and right_face == right and left != right
    far = np.zeros_like(lab); far[5, 5] = 1
    capped = fill(far, np.ones_like(fp), g.cell, DEFAULTS)
    assert capped[5, 5 + 19] == 1 and capped[5, 5 + 21] == 0   # 0.95 m in, 1.05 m out
```

- [ ] **Step 2: Run to see them fail**

Run: `cd /home/ruoyu/PanoPin && conda run -n panopin python -m pytest tests/test_roomseg_partition.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'panopin.roomseg.partition'`.

- [ ] **Step 3: Implement `partition.py`**

`src/panopin/roomseg/partition.py`:
```python
"""Band image + footprint -> room label image (spec §4 steps 2, 4-7, as amended 2026-10-01)."""
import numpy as np
from scipy import ndimage

from .raster import disk

FOUR = ndimage.generate_binary_structure(2, 1)   # 4-connectivity: no leaks through wall corners


def wall_mask(band_counts, cell, p):
    """Cells with >= wall_min_pts band points, GROWN by gap_close_r: seals scan gaps up to ~2r.

    A morphological closing cannot bridge a gap in a 1-cell-thin wall (a disk beside the gap
    misses the wall), so the walls are dilated instead. fill() hands the grown cells back to
    the nearest room, and min_room_halfwidth is measured on the free space that remains.
    """
    walls = band_counts >= p.wall_min_pts
    r = round(p.gap_close_r / cell)
    return ndimage.binary_dilation(walls, disk(r)) if r >= 1 else walls


def rooms(walls, footprint, cell, p):
    """Free-space components that are big and wide enough: 0 = none, 1..K."""
    free = footprint & ~walls
    lab, n = ndimage.label(free, structure=FOUR)
    if n == 0:
        return lab.astype(np.int32)
    idx = np.arange(1, n + 1)
    area = np.asarray(ndimage.sum(free, lab, idx)) * cell * cell
    halfw = np.asarray(ndimage.maximum(ndimage.distance_transform_edt(free), lab, idx)) * cell
    keep = idx[(area >= p.min_room_area) & (halfw >= p.min_room_halfwidth)]
    lut = np.zeros(n + 1, np.int32)
    lut[keep] = np.arange(1, len(keep) + 1)
    return lut[lab]


def fill(labels, footprint, cell, p):
    """Every footprint cell joins its nearest room if that room is within max_fill_dist."""
    if labels.max() == 0:
        return labels.astype(np.int32)
    dist, (r, c) = ndimage.distance_transform_edt(labels == 0, return_indices=True)
    out = labels[r, c].astype(np.int32)
    out[(dist * cell > p.max_fill_dist) | ~footprint] = 0
    return out
```

- [ ] **Step 4: Run to see them pass**

Run: `cd /home/ruoyu/PanoPin && conda run -n panopin python -m pytest tests/test_roomseg_partition.py -q`
Expected: `7 passed`.

- [ ] **Step 5: Amend the spec's gap-closing step**

In `docs/specs/2026-10-01-room-segmentation-design.md` §4, replace the row beginning `| 4 Gap closing |`
with:
```
| 4 Gap closing | grow the wall mask by `gap_close_r` (dilation): seals scan holes up to ~2r; door openings (≥ 0.8 m) stay open and are sealed only by lintels; step 7 hands the grown cells back to the nearest room. *Amended 2026-10-01 (plan review): a morphological closing cannot bridge a gap in a 1-cell-thin wall.* Step 6's half-width is measured on the free space left after growing. | `gap_close_r` 0.15 m |
```

- [ ] **Step 6: Commit**

```bash
cd /home/ruoyu/PanoPin && git add src/panopin/roomseg/partition.py tests/test_roomseg_partition.py docs/specs/2026-10-01-room-segmentation-design.md
git commit -m "feat(roomseg): walls from the band, free-space rooms, nearest-room fill

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Lift to points and the `segment()` pipeline

**Files:**
- Create: `src/panopin/roomseg/lift.py`, `src/panopin/roomseg/core.py`
- Modify: `src/panopin/roomseg/__init__.py`
- Test: `tests/test_roomseg_core.py`

**Interfaces:**
- Consumes: everything from Tasks 1–3.
- Produces:
  - `lift(filled: int32[ny,nx], grid: Grid, xy: float[N,2]) -> int32[N]` (0..K-1, or −1);
  - `segment(xyz: float[N,3], p: Params = DEFAULTS, n_panos: int | None = None) -> (labels: int32[N], image: int32[ny,nx], report: dict)`.
- Room order is canonical: points descending, ties by centroid x then y. Rooms with zero points are dropped.
- `report` keys: `floor_z, ceiling_z, z_peaks, band, params, grid{x0,y0,cell,ny,nx}, segments[{name,n_points,area_m2,centroid_xy}], K, N, n_points, n_unassigned, unassigned_frac, warnings`.
- `__init__` re-exports `segment`.

- [ ] **Step 1: Write the failing tests**

`tests/test_roomseg_core.py`:
```python
import numpy as np
import pytest

from panopin.roomseg import DEFAULTS, SegmentationError, segment
from panopin.roomseg.lift import lift
from panopin.roomseg.raster import Grid
from tests.roomseg_synth import H, box, two_rooms


def _floor(xyz):
    return np.abs(xyz[:, 2]) < 1e-9


def test_two_rooms_with_their_own_wall_faces():
    xyz = two_rooms()
    labels, image, rep = segment(xyz)
    assert rep["K"] == 2 and image.max() == 2
    left = _floor(xyz) & (xyz[:, 0] < 3.8)
    right = _floor(xyz) & (xyz[:, 0] > 4.2)
    assert len(set(labels[left])) == 1 and len(set(labels[right])) == 1
    assert labels[left][0] != labels[right][0]
    lface = np.abs(xyz[:, 0] - 3.9) < 1e-9
    rface = np.abs(xyz[:, 0] - 4.1) < 1e-9
    assert set(labels[lface]) == {labels[left][0]}
    assert set(labels[rface]) == {labels[right][0]}


def test_rooms_are_named_by_size():
    xyz = two_rooms(split=3.0)                      # right room is the bigger one
    labels, _, rep = segment(xyz)
    big = _floor(xyz) & (xyz[:, 0] > 3.2)
    assert set(labels[big]) == {0}
    assert [s["name"] for s in rep["segments"]] == ["seg_00", "seg_01"]
    assert rep["segments"][0]["n_points"] > rep["segments"][1]["n_points"]


def test_furniture_points_join_the_room_they_stand_in():
    xyz = np.concatenate([two_rooms(), box(5.0, 7.0, 3.0, 3.6, 2.0)])
    labels, _, _ = segment(xyz)
    cab = (xyz[:, 0] >= 5.0) & (xyz[:, 0] <= 7.0) & (xyz[:, 1] >= 3.0) & (xyz[:, 2] > 0.1) & (xyz[:, 2] < 2.05)
    right = _floor(xyz) & (xyz[:, 0] > 4.2)
    assert set(labels[cab]) == {labels[right][0]}


def test_a_point_far_outside_is_unassigned():
    xyz = np.concatenate([two_rooms(), [[20.0, 2.0, 1.0]]])
    labels, _, rep = segment(xyz)
    assert labels[-1] == -1 and rep["n_unassigned"] == 1
    assert rep["warnings"] == []


def test_more_segments_than_panos_warns():
    _, _, rep = segment(two_rooms(), n_panos=1)
    assert any("K=2 > N=1" in w for w in rep["warnings"])


def test_report_records_heights_band_and_params():
    _, _, rep = segment(two_rooms())
    assert abs(rep["ceiling_z"] - H) <= DEFAULTS.z_bin
    assert rep["band"][1] - rep["band"][0] == pytest.approx(0.5)
    assert rep["params"]["cell"] == 0.05


def test_no_room_raises():
    xyz = two_rooms(door_top=None, gap=(0.0, 4.0))     # wall removed entirely -> still 1 room
    assert segment(xyz)[2]["K"] == 1
    tiny = two_rooms() * np.array([0.1, 0.1, 1.0])     # 0.8 x 0.4 m: below min_room_area
    with pytest.raises(SegmentationError, match="no room"):
        segment(tiny)


def test_segment_is_deterministic():
    a = segment(two_rooms())
    b = segment(two_rooms())
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]) and a[2] == b[2]


def test_lift_maps_cells_to_zero_based_labels():
    g = Grid(0.0, 0.0, 1.0, 2, 2)
    filled = np.array([[1, 0], [2, 2]], np.int32)
    out = lift(filled, g, np.array([[0.5, 0.5], [1.5, 0.5], [1.5, 1.5]]))
    assert out.tolist() == [0, -1, 1]
```

- [ ] **Step 2: Run to see them fail**

Run: `cd /home/ruoyu/PanoPin && conda run -n panopin python -m pytest tests/test_roomseg_core.py -q`
Expected: FAIL with `ImportError: cannot import name 'segment' from 'panopin.roomseg'`.

- [ ] **Step 3: Implement `lift.py`, `core.py`, and update `__init__.py`**

`src/panopin/roomseg/lift.py`:
```python
"""Room label image -> per-point labels (spec §4 step 8)."""
import numpy as np


def lift(filled, grid, xy):
    """Per-point room index 0..K-1, or -1 where the point's cell holds no room."""
    row, col = grid.cells(xy)
    return filled[row, col].astype(np.int32) - 1
```

`src/panopin/roomseg/core.py`:
```python
"""segment(): one merged cloud -> per-point room labels + label image + report (spec §3-§5)."""
from dataclasses import asdict

import numpy as np

from .errors import SegmentationError
from .levels import storey_heights
from .lift import lift
from .params import DEFAULTS
from .partition import fill, rooms, wall_mask
from .raster import Grid, band_image, footprint


def segment(xyz, p=DEFAULTS, n_panos=None):
    """xyz (N,3) in metres, z-up, one storey -> (labels int32[N] in 0..K-1 or -1,
    image int32[ny,nx] in 0..K, report dict)."""
    xyz = np.asarray(xyz, dtype=float)
    floor_z, ceiling_z, peaks = storey_heights(xyz[:, 2], p)
    if p.band_from_floor is not None:
        z0 = floor_z + p.band_from_floor
    else:
        z0 = ceiling_z - p.band_bottom
    z1 = ceiling_z - p.band_top
    grid = Grid.fit(xyz[:, :2], p.cell, p.max_grid_cells)
    fp = footprint(grid, xyz, p.footprint_close)
    walls = wall_mask(band_image(grid, xyz, z0, z1), grid.cell, p)
    kept = rooms(walls, fp, grid.cell, p)
    if kept.max() == 0:
        raise SegmentationError(
            f"no room survived cleanup (band {z0:.2f}-{z1:.2f} m); check the band heights")
    filled = fill(kept, fp, grid.cell, p)
    labels, image, segments = _canonical_order(lift(filled, grid, xyz[:, :2]), filled, xyz, grid)
    n_un = int((labels < 0).sum())
    report = {
        "floor_z": floor_z, "ceiling_z": ceiling_z, "z_peaks": peaks, "band": [z0, z1],
        "params": asdict(p),
        "grid": {"x0": grid.x0, "y0": grid.y0, "cell": grid.cell, "ny": grid.ny, "nx": grid.nx},
        "segments": segments, "K": len(segments), "N": n_panos,
        "n_points": int(len(labels)), "n_unassigned": n_un,
        "unassigned_frac": n_un / max(len(labels), 1),
    }
    report["warnings"] = _warnings(report, p)
    return labels, image, report


def _canonical_order(raw, filled, xyz, grid):
    """Rename rooms by point count (desc), ties by centroid x then y; drop rooms with no points."""
    info = []
    for k in range(int(filled.max())):
        m = raw == k
        n = int(m.sum())
        if n == 0:
            continue
        c = xyz[m, :2].mean(axis=0)
        area = float((filled == k + 1).sum()) * grid.cell ** 2
        info.append((-n, float(c[0]), float(c[1]), k, area))
    info.sort()
    new_of_old = np.full(int(filled.max()) + 1, -1, np.int32)   # index = old label + 1
    segments = []
    for new, (neg_n, cx, cy, old, area) in enumerate(info):
        new_of_old[old + 1] = new
        segments.append({"name": f"seg_{new:02d}", "n_points": -neg_n, "area_m2": round(area, 3),
                         "centroid_xy": [round(cx, 3), round(cy, 3)]})
    labels = new_of_old[raw + 1]
    image = (new_of_old[filled] + 1).astype(np.int32)
    return labels, image, segments


def _warnings(r, p):
    out = []
    if r["N"] is not None and r["K"] > r["N"]:
        out.append(f"K={r['K']} > N={r['N']}: more segments than panoramas "
                   "(over-segmentation, or a space without a panorama)")
    if r["unassigned_frac"] > p.warn_unassigned_frac:
        out.append(f"{r['unassigned_frac']:.1%} of points unassigned "
                   f"(> {p.warn_unassigned_frac:.0%})")
    for s in r["segments"]:
        if s["area_m2"] < p.warn_small_room_area:
            out.append(f"{s['name']} is only {s['area_m2']} m2 (< {p.warn_small_room_area} m2)")
    return out
```

Replace `src/panopin/roomseg/__init__.py` with:
```python
"""Automatic room segmentation of one merged, z-up, single-storey cloud (spec 2026-10-01)."""
from .core import segment
from .errors import SegmentationError
from .params import DEFAULTS, HOVSG, Params

__all__ = ["segment", "SegmentationError", "Params", "DEFAULTS", "HOVSG"]
```

- [ ] **Step 4: Run to see them pass, plus the earlier suites**

Run: `cd /home/ruoyu/PanoPin && conda run -n panopin python -m pytest tests/test_roomseg_levels.py tests/test_roomseg_raster.py tests/test_roomseg_partition.py tests/test_roomseg_core.py -q`
Expected: `29 passed`.

- [ ] **Step 5: Commit**

```bash
cd /home/ruoyu/PanoPin && git add src/panopin/roomseg tests/test_roomseg_core.py
git commit -m "feat(roomseg): segment() pipeline with canonical room order and report

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Files and the `segment` CLI subcommand

**Files:**
- Create: `src/panopin/roomseg/files.py`
- Modify: `src/panopin/cli.py` (add the subparser; route `segment` before the existing `seed` body in `main`)
- Test: `tests/test_roomseg_cli.py`

**Interfaces:**
- Consumes: `segment`, `SegmentationError`, `DEFAULTS`, `HOVSG` (Task 4).
- Produces:
  - `read_cloud(path) -> (xyz float64[N,3], rgb uint8[N,3])`, for `.ply` (open3d) or `.txt`, in file order;
  - `write_outputs(out_dir, xyz, rgb, labels, image, report) -> dict[str, str]`;
  - `segment_cloud(cloud, out_dir, n_panos=None, p=DEFAULTS) -> report`;
  - CLI: `python -m panopin.cli segment --cloud C --out-dir D [--n-panos N] [--baseline-hovsg]`. Exit 0
    on success, 2 on `SegmentationError`, with nothing written to `D`.
- Outputs in `D`: `seg_XX.txt`, `clouds.json`, `labels.npy`, `segmentation.json`, `rooms.png`.

- [ ] **Step 1: Write the failing tests**

`tests/test_roomseg_cli.py`:
```python
import json

import numpy as np
import open3d as o3d

from data_utils import read_txt_pcd   # vendored CPO reader (tests/conftest.py puts it on sys.path)
from panopin import cli
from tests.roomseg_synth import H, colour, two_rooms


def _ply(path, xyz, with_colour=True):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    if with_colour:
        pcd.colors = o3d.utility.Vector3dVector(colour(xyz) / 255.0)
    o3d.io.write_point_cloud(str(path), pcd)
    return path


def test_segment_writes_every_output_and_cpo_can_read_it(tmp_path):
    xyz = two_rooms()
    out = tmp_path / "seg"
    rc = cli.main(["segment", "--cloud", str(_ply(tmp_path / "m.ply", xyz)), "--out-dir", str(out),
                   "--n-panos", "2"])
    assert rc == 0
    clouds = json.loads((out / "clouds.json").read_text())
    assert sorted(clouds) == ["seg_00", "seg_01"]
    labels = np.load(out / "labels.npy")
    assert labels.shape == (len(xyz),)
    total = 0
    for name, path in clouds.items():
        pts, rgb = read_txt_pcd(path)
        assert pts.shape[1] == 3 and rgb.shape[1] == 3 and 0.0 <= rgb.min() and rgb.max() <= 1.0
        k = int(name.split("_")[1])
        assert len(pts) == int((labels == k).sum())
        total += len(pts)
    assert total == int((labels >= 0).sum())
    rep = json.loads((out / "segmentation.json").read_text())
    assert rep["K"] == 2 and rep["N"] == 2 and rep["warnings"] == []
    assert (out / "rooms.png").exists()


def test_output_points_keep_input_coordinates(tmp_path):
    xyz = two_rooms() + np.array([100.0, -50.0, 3.0])
    out = tmp_path / "seg"
    assert cli.main(["segment", "--cloud", str(_ply(tmp_path / "m.ply", xyz)), "--out-dir", str(out)]) == 0
    pts, _ = read_txt_pcd(str(out / "seg_00.txt"))
    assert pts[:, 0].min() > 99.0 and pts[:, 1].max() < -45.0 and pts[:, 2].max() > H + 2.9


def test_k_greater_than_n_warns_on_stderr(tmp_path, capsys):
    out = tmp_path / "seg"
    cli.main(["segment", "--cloud", str(_ply(tmp_path / "m.ply", two_rooms())), "--out-dir", str(out),
              "--n-panos", "1"])
    assert "K=2 > N=1" in capsys.readouterr().err


def test_stale_segment_files_are_removed(tmp_path):
    out = tmp_path / "seg"; out.mkdir()
    (out / "seg_09.txt").write_text("0 0 0 0 0 0\n")
    cli.main(["segment", "--cloud", str(_ply(tmp_path / "m.ply", two_rooms())), "--out-dir", str(out)])
    assert sorted(p.name for p in out.glob("seg_*.txt")) == ["seg_00.txt", "seg_01.txt"]


def test_cloud_without_colours_is_refused_and_nothing_written(tmp_path, capsys):
    out = tmp_path / "seg"
    rc = cli.main(["segment", "--cloud", str(_ply(tmp_path / "m.ply", two_rooms(), with_colour=False)),
                   "--out-dir", str(out)])
    assert rc == 2 and "colour" in capsys.readouterr().err
    assert not out.exists()


def test_refused_geometry_writes_nothing(tmp_path, capsys):
    xyz = two_rooms()
    two_storey = np.concatenate([xyz, xyz + np.array([0.0, 0.0, H + 0.2])])
    out = tmp_path / "seg"
    rc = cli.main(["segment", "--cloud", str(_ply(tmp_path / "m.ply", two_storey)), "--out-dir", str(out)])
    assert rc == 2 and "storey" in capsys.readouterr().err
    assert not out.exists()


def test_txt_input_and_byte_identical_reruns(tmp_path):
    xyz = two_rooms()
    txt = tmp_path / "m.txt"
    np.savetxt(txt, np.column_stack([xyz, colour(xyz)]), fmt="%.6f %.6f %.6f %d %d %d")
    a, b = tmp_path / "a", tmp_path / "b"
    assert cli.main(["segment", "--cloud", str(txt), "--out-dir", str(a)]) == 0
    assert cli.main(["segment", "--cloud", str(txt), "--out-dir", str(b)]) == 0
    for name in ["seg_00.txt", "seg_01.txt", "labels.npy", "segmentation.json", "rooms.png"]:
        assert (a / name).read_bytes() == (b / name).read_bytes(), name


def test_baseline_flag_uses_the_hovsg_band(tmp_path):
    out = tmp_path / "seg"
    cli.main(["segment", "--cloud", str(_ply(tmp_path / "m.ply", two_rooms())), "--out-dir", str(out),
              "--baseline-hovsg"])
    rep = json.loads((out / "segmentation.json").read_text())
    assert rep["params"]["band_from_floor"] == 1.5
```

Note: `segmentation.json` records `"input"` as an absolute path, so files written by runs on different
input paths differ. The rerun test uses the same input path for both runs.

- [ ] **Step 2: Run to see them fail**

Run: `cd /home/ruoyu/PanoPin && conda run -n panopin python -m pytest tests/test_roomseg_cli.py -q`
Expected: FAIL. argparse exits with `invalid choice: 'segment'` (pytest reports `SystemExit: 2`).

- [ ] **Step 3: Implement `files.py`**

`src/panopin/roomseg/files.py`:
```python
"""Read a merged cloud; write the `segment` command's outputs (spec §3.1)."""
import json
from pathlib import Path

import cv2
import numpy as np

from .core import segment
from .errors import SegmentationError
from .params import DEFAULTS


def read_cloud(path):
    """(xyz float64[N,3], rgb uint8[N,3]) in file order, from .ply or an S3DIS-style .txt."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".ply":
        import open3d as o3d
        pcd = o3d.io.read_point_cloud(str(path))
        if not pcd.has_colors():
            raise SegmentationError(f"{path} has no colours; PanoPin's colour registration needs RGB")
        xyz = np.asarray(pcd.points, dtype=np.float64)
        rgb = np.rint(np.asarray(pcd.colors) * 255.0).astype(np.uint8)
    elif suffix == ".txt":
        import pandas as pd
        a = pd.read_csv(path, sep=r"\s+", header=None).values
        if a.ndim != 2 or a.shape[1] < 6:
            raise SegmentationError(f"{path}: expected columns X Y Z R G B")
        xyz = a[:, :3].astype(np.float64)
        rgb = np.clip(np.rint(a[:, 3:6]), 0, 255).astype(np.uint8)
    else:
        raise SegmentationError(f"unsupported cloud format '{suffix}' (use .ply or .txt)")
    if len(xyz) == 0:
        raise SegmentationError(f"{path} holds no points")
    return xyz, rgb


def write_outputs(out_dir, xyz, rgb, labels, image, report):
    """Write seg_XX.txt, clouds.json, labels.npy, segmentation.json, rooms.png; return clouds."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for stale in out.glob("seg_*.txt"):
        stale.unlink()
    clouds = {}
    for k, s in enumerate(report["segments"]):
        f = out / f"{s['name']}.txt"
        m = labels == k
        np.savetxt(f, np.column_stack([xyz[m], rgb[m]]), fmt="%.6f %.6f %.6f %d %d %d")
        clouds[s["name"]] = str(f.resolve())
    (out / "clouds.json").write_text(json.dumps(clouds, indent=1))
    np.save(out / "labels.npy", labels.astype(np.int32))
    (out / "segmentation.json").write_text(json.dumps(report, indent=1))
    cv2.imwrite(str(out / "rooms.png"), _colour(image))
    return clouds


def segment_cloud(cloud, out_dir, n_panos=None, p=DEFAULTS):
    """Read, segment, then write. Nothing is written if reading or segmenting raises."""
    xyz, rgb = read_cloud(cloud)
    labels, image, report = segment(xyz, p, n_panos)
    report["input"] = str(Path(cloud).resolve())
    write_outputs(out_dir, xyz, rgb, labels, image, report)
    return report


def _colour(image):
    lut = np.random.RandomState(7).randint(60, 256, size=(int(image.max()) + 1, 3)).astype(np.uint8)
    lut[0] = 0
    return np.flipud(lut[image])   # north up
```

- [ ] **Step 4: Wire the subcommand into `cli.py`**

In `src/panopin/cli.py` `main()`, after the `s.add_argument("--tau", …)` block and before
`args = parser.parse_args(argv)`, add:
```python
    g = sub.add_parser("segment", help="split one merged cloud into per-room candidate clouds")
    g.add_argument("--cloud", required=True, type=Path, help="merged cloud, .ply (with colours) or .txt")
    g.add_argument("--out-dir", required=True, type=Path,
                   help="writes seg_XX.txt, clouds.json (for `seed --clouds`), labels.npy, ...")
    g.add_argument("--n-panos", type=int, default=None,
                   help="number of panoramas; warns when segments outnumber them")
    g.add_argument("--baseline-hovsg", action="store_true",
                   help="baseline B1: band [floor + 1.5, ceiling - 0.3] instead of the ceiling band")
```
Directly after `args = parser.parse_args(argv)`, add:
```python
    if args.command == "segment":
        return _segment(args)
```
Then add this function above `main`:
```python
def _segment(args):
    """`segment` subcommand: exit 0 on success, 2 when the cloud is refused (nothing written)."""
    from panopin.roomseg import DEFAULTS, HOVSG, SegmentationError
    from panopin.roomseg.files import segment_cloud

    try:
        report = segment_cloud(args.cloud, args.out_dir, args.n_panos,
                               HOVSG if args.baseline_hovsg else DEFAULTS)
    except SegmentationError as e:
        print(f"segment: {e}", file=sys.stderr)
        return 2
    for w in report["warnings"]:
        print(f"segment: warning: {w}", file=sys.stderr)
    print(f"{report['K']} segments -> {args.out_dir / 'clouds.json'}")
    return 0
```

- [ ] **Step 5: Run the new tests and the existing CLI tests**

Run: `cd /home/ruoyu/PanoPin && conda run -n panopin python -m pytest tests/test_roomseg_cli.py tests/test_cli.py -q`
Expected: all pass. `test_roomseg_cli.py` contributes 8; `test_cli.py` keeps its own count unchanged.

- [ ] **Step 6: Commit**

```bash
cd /home/ruoyu/PanoPin && git add src/panopin/roomseg/files.py src/panopin/cli.py tests/test_roomseg_cli.py
git commit -m "feat(roomseg): panopin.cli segment writes the clouds.json that seed consumes

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Scenes config, ground-truth labels, scorer

**Files:**
- Create: `config/roomseg_scenes.json`, `eval/roomseg_gt.py`, `eval/roomseg_score.py`
- Test: `tests/test_roomseg_score.py`

**Interfaces:**
- Consumes: `read_cloud` (Task 5); `disk` (Task 2).
- Produces:
  - `roomseg_gt.load_scenes() -> dict`;
  - `roomseg_gt.gt_labels_for(xyz, rooms_dir, rooms, tol=1e-3) -> int32[N]` (index into `rooms`);
  - `roomseg_gt.cached_gt(scene) -> (xyz, gt)`;
  - `roomseg_score.voxels(xyz, gt, pred, v) -> (vkeys int64[M,3], vgt, vpred)`;
  - `roomseg_score.near_other_room(vkeys, vgt, radius_cells) -> bool[M]`;
  - `roomseg_score.metrics(vgt, vpred, rooms, seg_names) -> dict` with keys
    `matching{room: {segment, iou}}, f1_05, f1_07, pq, sq, rq, splits[room], merges{segment: [rooms]}, unassigned_frac, K, G`;
  - `roomseg_score.containment(xyz, gt, pred, panos, rooms, seg_names, matching, r=0.15) -> dict[key, {...}]`;
  - CLI `python eval/roomseg_score.py --scene S --seg-dir D --out O`, which writes `O/score.json` and `O/score.md`.

- [ ] **Step 1: Write the scenes config**

`config/roomseg_scenes.json`:
```json
{
  "_comment": "Room-segmentation scenes (spec 2026-10-01 §6.1). cloud = the ONLY solver input; rooms_dir/rooms/pose_json are GT for scoring only. e2e_stage0 = Point_360 Stage-0 inputs for the two-arm run (Task 9).",
  "scenes": {
    "Area_3_manhattan4": {
      "split": "dev",
      "cloud": "/home/ruoyu/Point_360/data/pointcloud/stanford/Area_3/Area_3_manhattan4.ply",
      "rooms_dir": "/home/ruoyu/Point_360/data/s3dis/Stanford3dDataset_v1.2/Area_3",
      "rooms": ["hallway_1", "office_3", "office_5", "office_7"],
      "pose_json": "/home/ruoyu/Point_360/data/s3dis/_multiroom4/pose.json",
      "e2e_stage0": "/home/ruoyu/Point_360/data/full_runs/Area_3_manhattan4/stage0_localization"
    },
    "Area_3_manhattan6": {
      "split": "dev",
      "cloud": "/home/ruoyu/Point_360/data/pointcloud/stanford/Area_3/Area_3_manhattan6.ply",
      "rooms_dir": "/home/ruoyu/Point_360/data/s3dis/Stanford3dDataset_v1.2/Area_3",
      "rooms": ["hallway_1", "office_3", "office_4", "office_5", "office_6", "office_7"],
      "pose_json": "/home/ruoyu/Point_360/data/s3dis/_multiroom6/pose.json",
      "e2e_stage0": "/home/ruoyu/Point_360/data/full_runs/Area_3_manhattan6/stage0_localization"
    },
    "Area_2_manhattan4": {
      "split": "holdout",
      "cloud": "/home/ruoyu/Point_360/data/pointcloud/stanford/Area_2_manhattan4.ply",
      "rooms_dir": "/home/ruoyu/Point_360/data/s3dis/Stanford3dDataset_v1.2/Area_2",
      "rooms": ["hallway_2", "office_6", "office_7", "office_8"],
      "pose_json": "/home/ruoyu/Point_360/data/s3dis/_area_2_manhattan4/pose.json",
      "e2e_stage0": "/home/ruoyu/Point_360/data/full_runs/Area_2_manhattan4/stage0_localization"
    },
    "Area_2_manhattan7": {
      "split": "holdout",
      "cloud": "/home/ruoyu/Point_360/data/pointcloud/stanford/Area_2_manhattan7.ply",
      "rooms_dir": "/home/ruoyu/Point_360/data/s3dis/Stanford3dDataset_v1.2/Area_2",
      "rooms": ["hallway_2", "hallway_3", "office_4", "office_5", "office_6", "office_7", "office_8"],
      "pose_json": "/home/ruoyu/Point_360/data/s3dis/_area_2_manhattan7/pose.json",
      "e2e_stage0": "/home/ruoyu/Point_360/data/full_runs/Area_2_manhattan7/stage0_localization"
    }
  }
}
```

- [ ] **Step 2: Write the failing tests**

`tests/test_roomseg_score.py`:
```python
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
import roomseg_gt      # noqa: E402
import roomseg_score   # noqa: E402

ROOMS = ["a", "b"]


def _two_blocks():
    """Room a: x in [0, 2); room b: x in [2, 4); 5 cm voxels, one point per voxel."""
    xs, ys = np.meshgrid(np.arange(0, 4, 0.05) + 0.025, np.arange(0, 1, 0.05) + 0.025)
    xyz = np.column_stack([xs.ravel(), ys.ravel(), np.full(xs.size, 0.025)])
    gt = (xyz[:, 0] >= 2.0).astype(np.int64)
    return xyz, gt


def test_perfect_partition_scores_one():
    xyz, gt = _two_blocks()
    vk, vg, vp = roomseg_score.voxels(xyz, gt, gt.copy(), 0.05)
    m = roomseg_score.metrics(vg, vp, ROOMS, ["seg_00", "seg_01"])
    assert m["f1_07"] == 1.0 and m["pq"] == pytest.approx(1.0)
    assert m["splits"] == [] and m["merges"] == {}
    assert m["matching"]["a"]["segment"] == "seg_00"


def test_merge_and_split_are_named():
    xyz, gt = _two_blocks()
    merged = np.zeros_like(gt)
    m = roomseg_score.metrics(*roomseg_score.voxels(xyz, gt, merged, 0.05)[1:], ROOMS, ["seg_00"])
    assert m["merges"] == {"seg_00": ["a", "b"]}
    split = np.where(xyz[:, 0] < 1.0, 0, np.where(xyz[:, 0] < 2.0, 1, 2))
    m = roomseg_score.metrics(*roomseg_score.voxels(xyz, gt, split, 0.05)[1:], ROOMS,
                              ["seg_00", "seg_01", "seg_02"])
    assert m["splits"] == ["a"]


def test_majority_label_per_voxel():
    xyz = np.array([[0.01, 0.01, 0.01]] * 3 + [[0.07, 0.01, 0.01]])
    vk, vg, vp = roomseg_score.voxels(xyz, np.array([0, 0, 1, 1]), np.array([1, 1, 0, -1]), 0.05)
    assert vg.tolist() == [0, 1] and vp.tolist() == [1, -1]


def test_boundary_band_flags_only_voxels_near_another_room():
    xyz, gt = _two_blocks()
    vk, vg, _ = roomseg_score.voxels(xyz, gt, gt, 0.05)
    near = roomseg_score.near_other_room(vk, vg, 4)
    x = (vk[:, 0] + 0.5) * 0.05
    assert near[np.abs(x - 2.0) < 0.15].all()
    assert not near[np.abs(x - 2.0) > 0.25].any()


def test_containment_uses_points_around_the_camera():
    xyz, gt = _two_blocks()
    pred = gt.copy()
    panos = {"camera_u1_a": [0.5, 0.5, 1.4], "camera_u2_b": [3.5, 0.5, 1.4]}
    matching = {"a": {"segment": "seg_00", "iou": 1.0}, "b": {"segment": "seg_01", "iou": 1.0}}
    rows = roomseg_score.containment(xyz, gt, pred, panos, ROOMS, ["seg_00", "seg_01"], matching)
    assert rows["camera_u1_a"]["gt_inside"] and rows["camera_u1_a"]["pred_inside"]
    rows = roomseg_score.containment(xyz, gt, np.zeros_like(gt), panos, ROOMS, ["seg_00"],
                                     {"a": {"segment": "seg_00", "iou": 0.5}, "b": None})
    assert rows["camera_u2_b"]["pred_inside"] is False


def test_gt_labels_match_room_files_in_any_order(tmp_path):
    xyz, gt = _two_blocks()
    for i, r in enumerate(ROOMS):
        (tmp_path / r).mkdir()
        sel = xyz[gt == i]
        np.savetxt(tmp_path / r / f"{r}.txt", np.column_stack([sel, np.zeros((len(sel), 3))]),
                   fmt="%.6f")
    perm = np.random.RandomState(0).permutation(len(xyz))
    labels = roomseg_gt.gt_labels_for(xyz[perm], tmp_path, ROOMS)
    assert np.array_equal(labels, gt[perm])
    with pytest.raises(ValueError, match="points"):
        roomseg_gt.gt_labels_for(xyz[:-1], tmp_path, ROOMS)
```

- [ ] **Step 3: Run to see them fail**

Run: `cd /home/ruoyu/PanoPin && conda run -n panopin python -m pytest tests/test_roomseg_score.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'roomseg_gt'`.

- [ ] **Step 4: Implement `eval/roomseg_gt.py`**

```python
"""Per-point ground-truth room labels for a merged S3DIS scene (HARNESS ONLY; spec §6.2).

The merged scene .ply is the exact union of its rooms' .txt files (measured 2026-10-01), so every
merged point is matched to a room point within 1 mm, or this module refuses.
Run: conda run -n panopin python eval/roomseg_gt.py --scene Area_3_manhattan4
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from panopin.roomseg.files import read_cloud  # noqa: E402

CACHE = REPO / "runs" / "roomseg_gt"


def load_scenes(path=REPO / "config" / "roomseg_scenes.json"):
    return json.loads(Path(path).read_text())["scenes"]


def gt_labels_for(xyz, rooms_dir, rooms, tol=1e-3):
    """Index into `rooms` of every xyz point, by nearest match to the rooms' .txt files."""
    parts, ids = [], []
    for i, r in enumerate(rooms):
        a = pd.read_csv(Path(rooms_dir) / r / f"{r}.txt", sep=r"\s+", header=None,
                        usecols=[0, 1, 2]).values
        parts.append(a)
        ids.append(np.full(len(a), i, np.int32))
    ref, rid = np.concatenate(parts), np.concatenate(ids)
    if len(ref) != len(xyz):
        raise ValueError(f"merged cloud has {len(xyz)} points, the room files hold {len(ref)} points")
    d, j = cKDTree(ref).query(xyz, k=1)
    if d.max() > tol:
        raise ValueError(f"{int((d > tol).sum())} points have no room point within {tol} m "
                         f"(max {d.max():.4f} m)")
    return rid[j]


def cached_gt(scene):
    """(xyz, gt) for a configured scene; GT labels cached in runs/roomseg_gt/<scene>.npy."""
    cfg = load_scenes()[scene]
    xyz, _ = read_cloud(cfg["cloud"])
    f = CACHE / f"{scene}.npy"
    if f.exists():
        gt = np.load(f)
        if len(gt) == len(xyz):
            return xyz, gt
    gt = gt_labels_for(xyz, cfg["rooms_dir"], cfg["rooms"])
    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(f, gt)
    return xyz, gt


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    a = ap.parse_args()
    xyz, gt = cached_gt(a.scene)
    rooms = load_scenes()[a.scene]["rooms"]
    print(a.scene, len(xyz), "points", {r: int((gt == i).sum()) for i, r in enumerate(rooms)})
```

- [ ] **Step 5: Implement `eval/roomseg_score.py`**

```python
"""Score a room segmentation against the S3DIS partition (HARNESS ONLY; spec §6.2).

5 cm voxels (Matterport density varies with range), majority labels per voxel, voxels within
0.2 m (XY) of another GT room ignored. Then one-to-one Hungarian matching on IoU, F1 at 0.5 / 0.7,
PQ, named splits / merges, unassigned share, and pano containment.
Run: conda run -n panopin python eval/roomseg_score.py --scene S --seg-dir runs/roomseg/S --out runs/roomseg_eval/S
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "eval"))
from panopin.roomseg.raster import disk  # noqa: E402
import roomseg_gt  # noqa: E402

VOXEL = 0.05
BAND_CELLS = 4          # 0.2 m ignore band
SPLIT_SHARE = 0.80      # a GT room is split if no segment holds >= 80 % of it
MERGE_SHARE = 0.10      # a segment merges rooms if it holds >= 10 % of each of >= 2 rooms


def _majority(vox, lab):
    lab = lab.astype(np.int64)
    base = lab.min()
    off = lab - base
    span = int(off.max()) + 1
    u, c = np.unique(vox.astype(np.int64) * span + off, return_counts=True)
    pv, pl = u // span, u % span
    order = np.lexsort((pl, -c, pv))            # by voxel, then count desc, then label asc
    _, first = np.unique(pv[order], return_index=True)
    return pl[order][first] + base


def voxels(xyz, gt, pred, v=VOXEL):
    keys = np.floor(np.asarray(xyz) / v).astype(np.int64)
    vkeys, vox = np.unique(keys, axis=0, return_inverse=True)
    vox = vox.ravel()
    return vkeys, _majority(vox, gt), _majority(vox, pred)


def near_other_room(vkeys, vgt, radius_cells=BAND_CELLS):
    xy = vkeys[:, :2] - vkeys[:, :2].min(axis=0)
    shape = tuple(xy.max(axis=0) + 1)
    k = disk(radius_cells)
    occ = {}
    for r in np.unique(vgt):
        m = np.zeros(shape, bool)
        m[xy[vgt == r, 0], xy[vgt == r, 1]] = True
        occ[r] = m
    near = np.zeros(len(vgt), bool)
    for r in occ:
        others = np.zeros(shape, bool)
        for s, m in occ.items():
            if s != r:
                others |= m
        if not others.any():
            continue
        dil = ndimage.binary_dilation(others, k)
        sel = vgt == r
        near[sel] = dil[xy[sel, 0], xy[sel, 1]]
    return near


def metrics(vgt, vpred, rooms, seg_names):
    G, S = len(rooms), len(seg_names)
    inter = np.zeros((G, S))
    ok = vpred >= 0
    np.add.at(inter, (vgt[ok], vpred[ok]), 1)
    gsize = np.bincount(vgt, minlength=G).astype(float)
    ssize = np.bincount(vpred[ok], minlength=S).astype(float)
    union = gsize[:, None] + ssize[None, :] - inter
    iou = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
    rr, cc = linear_sum_assignment(-iou)
    matching = {r: None for r in rooms}
    tp_ious = []
    for g, s in zip(rr, cc):
        if iou[g, s] > 0:
            matching[rooms[g]] = {"segment": seg_names[s], "iou": round(float(iou[g, s]), 4)}
            tp_ious.append(float(iou[g, s]))

    def f1(t):
        tp = sum(i > t for i in tp_ious)
        return 0.0 if tp == 0 else 2 * tp / (G + S)

    tp5 = [i for i in tp_ious if i > 0.5]
    rq = 0.0 if not tp5 else len(tp5) / (len(tp5) + 0.5 * (S - len(tp5)) + 0.5 * (G - len(tp5)))
    sq = float(np.mean(tp5)) if tp5 else 0.0
    share = np.divide(inter, gsize[:, None], out=np.zeros_like(inter), where=gsize[:, None] > 0)
    splits = [rooms[g] for g in range(G) if share[g].max() < SPLIT_SHARE]
    merges = {seg_names[s]: [rooms[g] for g in range(G) if share[g, s] >= MERGE_SHARE]
              for s in range(S) if (share[:, s] >= MERGE_SHARE).sum() >= 2}
    return {"matching": matching, "f1_05": f1(0.5), "f1_07": f1(0.7), "pq": sq * rq, "sq": sq,
            "rq": rq, "splits": splits, "merges": merges,
            "unassigned_frac": float((~ok).mean()), "K": S, "G": G}


def containment(xyz, gt, pred, panos, rooms, seg_names, matching, r=0.15):
    """Is each pano's GT camera XY inside its room's matched segment? (majority of points within r)"""
    tree = cKDTree(np.asarray(xyz)[:, :2])
    rows = {}
    for key, t in panos.items():
        room = key.split("_", 2)[2]
        if room not in rooms:
            continue
        idx = tree.query_ball_point(np.asarray(t[:2], float), r)
        row = {"room": room, "n_points": len(idx), "gt_inside": False, "pred_segment": None,
               "pred_inside": False}
        if idx:
            row["gt_inside"] = rooms[int(np.bincount(gt[idx]).argmax())] == room
            pv = pred[idx]
            pv = pv[pv >= 0]
            if len(pv):
                row["pred_segment"] = seg_names[int(np.bincount(pv).argmax())]
            m = matching.get(room)
            row["pred_inside"] = bool(m) and row["pred_segment"] == m["segment"]
        rows[key] = row
    return rows


def score_scene(scene, seg_dir):
    cfg = roomseg_gt.load_scenes()[scene]
    xyz, gt = roomseg_gt.cached_gt(scene)
    pred = np.load(Path(seg_dir) / "labels.npy")
    seg_names = list(json.loads((Path(seg_dir) / "clouds.json").read_text()))
    if len(pred) != len(xyz):
        raise ValueError(f"labels.npy has {len(pred)} rows, the scene cloud {len(xyz)} points")
    vkeys, vgt, vpred = voxels(xyz, gt, pred)
    keep = ~near_other_room(vkeys, vgt)
    m = metrics(vgt[keep], vpred[keep], cfg["rooms"], seg_names)
    panos = {k: v["t"] for k, v in json.loads(Path(cfg["pose_json"]).read_text()).items()}
    m["containment"] = containment(xyz, gt, pred, panos, cfg["rooms"], seg_names, m["matching"])
    m["scene"], m["split"], m["seg_dir"] = scene, cfg["split"], str(seg_dir)
    m["voxels_scored"], m["voxels_ignored"] = int(keep.sum()), int((~keep).sum())
    return m


def to_markdown(m):
    lines = [f"### {m['scene']} ({m['split']}) — {m['seg_dir']}", "",
             f"K={m['K']} vs G={m['G']} · F1@0.5 {m['f1_05']:.2f} · F1@0.7 {m['f1_07']:.2f} · "
             f"PQ {m['pq']:.3f} · unassigned {m['unassigned_frac']:.2%} · "
             f"splits {m['splits'] or 'none'} · merges {m['merges'] or 'none'}", "",
             "| GT room | segment | IoU |", "|---|---|---|"]
    for room, x in m["matching"].items():
        lines.append(f"| {room} | {x['segment'] if x else '—'} | {x['iou'] if x else 0:.3f} |")
    c = m["containment"]
    lines += ["", f"containment: pred {sum(r['pred_inside'] for r in c.values())}/{len(c)}, "
              f"GT partition itself {sum(r['gt_inside'] for r in c.values())}/{len(c)}"]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--seg-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    m = score_scene(a.scene, a.seg_dir)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "score.json").write_text(json.dumps(m, indent=1))
    md = to_markdown(m)
    (a.out / "score.md").write_text(md)
    print(md)
```

Note: `f1(t)` uses `F1 = 2·TP / (G + K)`, which equals `2PR/(P+R)` with `P = TP/K` and `R = TP/G`.

- [ ] **Step 6: Run to see them pass**

Run: `cd /home/ruoyu/PanoPin && conda run -n panopin python -m pytest tests/test_roomseg_score.py -q`
Expected: `6 passed`.

- [ ] **Step 7: Check GT labels on all four real scenes**

Run: `cd /home/ruoyu/PanoPin && for s in Area_3_manhattan4 Area_3_manhattan6 Area_2_manhattan4 Area_2_manhattan7; do conda run -n panopin python eval/roomseg_gt.py --scene $s; done`
Expected: four lines with per-room point counts. For example, `Area_3_manhattan4 3403895 points {...}`
with `hallway_1` ≈ 973222. No `ValueError`.

- [ ] **Step 8: Commit**

```bash
cd /home/ruoyu/PanoPin && git add config/roomseg_scenes.json eval/roomseg_gt.py eval/roomseg_score.py tests/test_roomseg_score.py
git commit -m "feat(eval): room-segmentation scorer against the S3DIS partition

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: Dev scenes — score, tune values only, run baseline B1

**Files:**
- Create: `docs/roomseg-results.md`
- Modify: `src/panopin/roomseg/params.py` (values only, if and as logged);
  `docs/specs/2026-10-01-room-segmentation-design.md` §6.3 (B1 amendment)

**Interfaces:**
- Consumes: CLI `segment` (Task 5); scorer (Task 6).
- Produces: frozen `DEFAULTS` values; `docs/roomseg-results.md` with the dev tables.

- [ ] **Step 1: Amend the spec's baseline B1**

In `docs/specs/2026-10-01-room-segmentation-design.md` §6.3, replace
`- **B1:** the same pipeline with the full-height image (`band = full`). It measures what the band buys.`
with:
```
- **B1:** the same pipeline on the HOV-SG band [floor + 1.5, ceiling − 0.3] (`--baseline-hovsg`). *Amended 2026-10-01 (plan Task 7):* a full-height band is degenerate for the enclosure method, because every cell holds points, so there is no free space. The HOV-SG band is the published alternative that the probe showed leaking.
```

- [ ] **Step 2: Run the segmenter and the scorer on both dev scenes**

```bash
cd /home/ruoyu/PanoPin
for s in Area_3_manhattan4:4 Area_3_manhattan6:6; do n=${s%%:*}; k=${s##*:}
  (cd src && conda run -n panopin python -m panopin.cli segment \
     --cloud /home/ruoyu/Point_360/data/pointcloud/stanford/Area_3/$n.ply --out-dir ../runs/roomseg/$n --n-panos $k)
  conda run -n panopin python eval/roomseg_score.py --scene $n --seg-dir runs/roomseg/$n --out runs/roomseg_eval/$n
done
```
Expected: two markdown blocks.
- **Pass, per scene:** K = G, F1@0.7 = 1.00, no splits, no merges, every GT room matched with
  IoU ≥ 0.7, and pred containment equal to the panos for which the GT partition itself contains the
  camera.
- Open `runs/roomseg/<scene>/rooms.png` (Read tool) to see the partition.

- [ ] **Step 3: Run baseline B1 on both dev scenes**

```bash
cd /home/ruoyu/PanoPin
for s in Area_3_manhattan4:4 Area_3_manhattan6:6; do n=${s%%:*}; k=${s##*:}
  (cd src && conda run -n panopin python -m panopin.cli segment --baseline-hovsg \
     --cloud /home/ruoyu/Point_360/data/pointcloud/stanford/Area_3/$n.ply --out-dir ../runs/roomseg_b1/$n --n-panos $k)
  conda run -n panopin python eval/roomseg_score.py --scene $n --seg-dir runs/roomseg_b1/$n --out runs/roomseg_eval_b1/$n
done
```
Expected: B1 scores no better than the ceiling band. The probe showed the HOV-SG band leaking on
`Area_3_manhattan6`. Record either way.

- [ ] **Step 4: If a dev scene fails, diagnose and tune values only**

1. Read the failing scene's `score.md` (which rooms split or merge) and `rooms.png`.
2. Map the failure to the one parameter that governs it:

| Symptom | Parameter |
|---|---|
| two rooms merged through a wall gap | `gap_close_r` (≤ 0.35 m; must stay below half a door width) or `wall_min_pts` |
| a room merged through a lowered ceiling or soffit band | `band_bottom` / `band_top` |
| a corridor split by a duct | `band_top` (move the band down) |
| a real room dropped | `min_room_area` / `min_room_halfwidth` |

3. Change **one** value in `params.py`, re-run Step 2 for **both** dev scenes, and log the old and new
   value plus both scenes' before/after F1@0.7 / splits / merges in `docs/roomseg-results.md`.
4. Re-run `conda run -n panopin python -m pytest tests/test_roomseg_*.py -q` after every change. Synthetic
   tests must stay green.
5. **Stop and report to the user if:**
   - three value changes have not passed both dev scenes;
   - the fix needs an algorithm change (spec §10, e.g. a per-region ceiling);
   - a value would leave its stated bound.

- [ ] **Step 5: Write `docs/roomseg-results.md`**

Use this structure, with the actual numbers from Steps 2–4:
```markdown
# Room segmentation (T11) — results

Spec `docs/specs/2026-10-01-room-segmentation-design.md` · plan `docs/plans/2026-10-01-room-segmentation.md`.
All numbers from `eval/roomseg_score.py` (5 cm voxels, 0.2 m boundary band ignored).

## Dev (Area_3) — method
<paste both score.md blocks>

## Dev — baseline B1 (HOV-SG band)
<paste both score.md blocks>

## Tuning log
| # | parameter | old → new | why (symptom) | manhattan4 F1@0.7 / splits / merges before → after | manhattan6 … |
(or: "No changes: defaults passed both dev scenes.")

## Frozen parameters
<the final Params values> — frozen at commit <filled in Task 8 step 1>.
```

- [ ] **Step 6: Commit**

```bash
cd /home/ruoyu/PanoPin && git add docs/roomseg-results.md docs/specs/2026-10-01-room-segmentation-design.md src/panopin/roomseg/params.py
git commit -m "eval(roomseg): dev scenes scored, baseline B1, tuning log

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: Freeze and score the holdout once

**Files:**
- Modify: `docs/roomseg-results.md`

- [ ] **Step 1: Record the freeze commit**

Run: `cd /home/ruoyu/PanoPin && git rev-parse --short HEAD`.
In `docs/roomseg-results.md` §Frozen parameters, write `frozen at commit <hash>`, then commit. From
here on `params.py` must not change.
```bash
git add docs/roomseg-results.md && git commit -m "docs(roomseg): freeze parameters before holdout

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

- [ ] **Step 2: Segment and score both holdout scenes — once**

```bash
cd /home/ruoyu/PanoPin
for s in Area_2_manhattan4:4 Area_2_manhattan7:7; do n=${s%%:*}; k=${s##*:}
  (cd src && conda run -n panopin python -m panopin.cli segment \
     --cloud /home/ruoyu/Point_360/data/pointcloud/stanford/$n.ply --out-dir ../runs/roomseg/$n --n-panos $k)
  conda run -n panopin python eval/roomseg_score.py --scene $n --seg-dir runs/roomseg/$n --out runs/roomseg_eval/$n
done
```
Expected: two markdown blocks.
- Same pass rule as Task 7 Step 2.
- `Area_2_manhattan7` contains the `hallway_2`|`hallway_3` corridor door (x = 7.77); convention A
  expects two corridor segments there.

- [ ] **Step 3: Record the results — whatever they are**

Add a `## Holdout (Area_2) — scored once at <hash>` section to `docs/roomseg-results.md` with both
blocks. **Do not tune on holdout.** If either scene fails, record which rooms split or merged, say so,
and continue to Task 9 only if the user agrees.

- [ ] **Step 4: Commit**

```bash
cd /home/ruoyu/PanoPin && git add docs/roomseg-results.md
git commit -m "eval(roomseg): holdout Area_2 scored once with frozen parameters

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 9: End-to-end, two arms (bar b, criteria 2 and 3)

**Files:**
- Create: `experiments/roomseg/area3_stage0_selection.py`, `experiments/roomseg/two_arm.py`
- Modify (Point_360, **untracked data script, do not commit there**):
  `/home/ruoyu/Point_360/data/full_runs/_scripts/loc_inputs.py:31`
- Modify: `docs/roomseg-results.md`

**Interfaces:**
- Consumes:
  - `runs/roomseg/<scene>/clouds.json` (Tasks 7–8);
  - `runs/roomseg_eval/<scene>/score.json` keys `matching`, `containment` (Task 6);
  - `config/roomseg_scenes.json` keys `pose_json`, `e2e_stage0`.
- From Point_360:
  - `run_localization.py` writes `<work-dir>/<scene>/demo6_alignment.json`, whose `matches[]` carry
    `pano_name` (bare uuid) and `room_label`;
  - the contract JSON is keyed by bare uuid, with `t`;
  - `stage0_localization/key_map.json` maps uuid → `camera_<uuid>_<room>`.

- [ ] **Step 1: Write the Area_3 pano-selection script**

`experiments/roomseg/area3_stage0_selection.py`:
```python
"""Write Point_360/data/full_runs/<scene>/{pano_selection.json, panos/} for an Area_3 dev scene from
its scene.json manifest (s3dis/choose_stations.py: one station per room), so Point_360's
data/full_runs/_scripts/loc_inputs.py can build its Stage-0 inputs exactly as for Area_2.

Run: python3 experiments/roomseg/area3_stage0_selection.py Area_3_manhattan4
"""
import json
import shutil
import sys
from pathlib import Path

P360 = Path("/home/ruoyu/Point_360")
MANIFEST = {"Area_3_manhattan4": "_multiroom4", "Area_3_manhattan6": "_multiroom6"}
PANO_RGB = P360 / "data/s3dis/area_3_no_xyz/area_3/pano/rgb"

scene = sys.argv[1]
stations = json.loads((P360 / "data/s3dis" / MANIFEST[scene] / "scene.json").read_text())["stations"]
run = P360 / "data/full_runs" / scene
(run / "panos").mkdir(parents=True, exist_ok=True)
rooms = {}
for st in stations:
    png = st["pano_key"] + "_frame_equirectangular_domain_rgb.png"
    shutil.copy(PANO_RGB / png, run / "panos" / png)
    rooms[st["room"]] = {"pano": png, "t": st["t"], "clearance_m": st["clearance_m"],
                         "to_centroid_m": st["to_centroid_m"], "basis": st["basis"]}
(run / "pano_selection.json").write_text(json.dumps(
    {"scene": scene, "rule": "the scene.json station (Point_360 s3dis/choose_stations.py)",
     "rooms": rooms}, indent=1))
print(scene, "->", {r: v["pano"][:16] for r, v in rooms.items()})
```
Run: `cd /home/ruoyu/PanoPin && for s in Area_3_manhattan4 Area_3_manhattan6; do python3 experiments/roomseg/area3_stage0_selection.py $s; done`
Expected: two lines listing one pano per room (4 and 6 rooms).

- [ ] **Step 2: Let Point_360's `loc_inputs.py` take the scene's area**

In `/home/ruoyu/Point_360/data/full_runs/_scripts/loc_inputs.py`, line 31 reads
`A2 = REPO / "data/s3dis/Stanford3dDataset_v1.2/Area_2"`. Replace it with:
```python
A2 = REPO / "data/s3dis/Stanford3dDataset_v1.2" / "_".join(sys.argv[1].split("_")[:2])  # the scene's area (was Area_2 only)
```
This file is untracked; do not commit in Point_360. Mention the edit in the final report to the user.

- [ ] **Step 3: Build Stage-0 inputs for the Area_3 dev scenes**

Run: `cd /home/ruoyu/Point_360 && for s in Area_3_manhattan4 Area_3_manhattan6; do conda run --no-capture-output -n point360 python data/full_runs/_scripts/loc_inputs.py $s; done`
Expected: ends with `STAGE0_INPUTS_DONE Area_3_manhattan4` and `… manhattan6`. This builds the line map
and FGPL features; expect minutes.

- [ ] **Step 4: Write the two-arm runner**

`experiments/roomseg/two_arm.py`:
```python
"""Bar (b) end-to-end (spec §8.3): Stage 0 = PanoPin seed + upright-prior FGPL on one scene, run FRESH
in two arms by the same command, differing ONLY in --clouds-json:
  gt   = the S3DIS room clouds (the scene's stage0_localization/clouds.json)
  pred = this branch's segments (runs/roomseg/<scene>/clouds.json)
Pass: pred room accuracy >= gt's, pred translation median <= gt's + 0.05 m, and every e2e pano
contained in its room's matched segment (runs/roomseg_eval/<scene>/score.json).

Run (system python3; run_localization.py shells into the conda envs):
  python3 experiments/roomseg/two_arm.py --scene Area_3_manhattan4
"""
import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
P360 = Path("/home/ruoyu/Point_360")
MARGIN_M = 0.05


def run_arm(scene, stage0, clouds_json, out):
    out.mkdir(parents=True, exist_ok=True)
    cmd = ["python3", str(P360 / "run_localization.py"), "--pose-source", "panopin", "--scene", scene,
           "--output", str(out / "pose_contract.json"), "--sidecar", str(out / "pose_contract.sidecar.json"),
           "--panos-json", str(stage0 / "panos.json"), "--clouds-json", str(clouds_json),
           "--metadata", str(stage0 / "metadata.json"),
           "--line-map", str(stage0 / "linemap" / "3d_line_map.pkl"),
           "--cloud-ply", str(stage0 / "cloud" / f"{scene}.ply"), "--density-png", str(stage0 / "density.png"),
           "--features-dir", str(stage0 / "features"), "--pano-dir", str(stage0 / "panos_fgpl"),
           "--work-dir", str(out / "_work"), "--tau", "0.10",
           "--fgpl-root", "/home/ruoyu/scan2measure-webframework", "--panopin-root", str(REPO)]
    with open(out / "run.log", "w") as log:
        subprocess.run(cmd, check=True, cwd=P360, stdout=log, stderr=subprocess.STDOUT)


def evaluate(scene, key_map, out, gt_pose, room_of_label):
    """Per pano: the room PanoPin chose (via room_of_label) and the FGPL translation error."""
    align = json.loads((out / "_work" / scene / "demo6_alignment.json").read_text())
    chosen = {m["pano_name"]: m["room_label"] for m in align["matches"]}
    poses = json.loads((out / "pose_contract.json").read_text())
    rows = {}
    for uuid, key in key_map.items():
        room = key.split("_", 2)[2]
        label = chosen.get(uuid)
        err = (float(np.linalg.norm(np.asarray(poses[uuid]["t"]) - np.asarray(gt_pose[key]["t"])))
               if uuid in poses else float("inf"))
        rows[uuid] = {"room": room, "label": label,
                      "room_ok": label is not None and room_of_label(label) == room,
                      "trans_err_m": round(err, 3)}
    errs = [r["trans_err_m"] for r in rows.values()]
    return {"per_pano": rows, "room_acc": sum(r["room_ok"] for r in rows.values()) / len(rows),
            "trans_median_m": float(np.median(errs))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    a = ap.parse_args()
    cfg = json.loads((REPO / "config" / "roomseg_scenes.json").read_text())["scenes"][a.scene]
    stage0 = Path(cfg["e2e_stage0"])
    key_map = json.loads((stage0 / "key_map.json").read_text())
    gt_pose = json.loads(Path(cfg["pose_json"]).read_text())
    score = json.loads((REPO / "runs" / "roomseg_eval" / a.scene / "score.json").read_text())
    seg_room = {m["segment"]: room for room, m in score["matching"].items() if m}
    out = REPO / "runs" / "roomseg_e2e" / a.scene
    run_arm(a.scene, stage0, stage0 / "clouds.json", out / "gt")
    run_arm(a.scene, stage0, REPO / "runs" / "roomseg" / a.scene / "clouds.json", out / "pred")
    gt = evaluate(a.scene, key_map, out / "gt", gt_pose, lambda lab: lab)
    pred = evaluate(a.scene, key_map, out / "pred", gt_pose, seg_room.get)
    contained = {u: bool(score["containment"].get(k, {}).get("pred_inside")) for u, k in key_map.items()}
    result = {"scene": a.scene, "gt": gt, "pred": pred, "contained": contained, "margin_m": MARGIN_M,
              "pass": {"room_acc": pred["room_acc"] >= gt["room_acc"],
                       "trans_median": pred["trans_median_m"] <= gt["trans_median_m"] + MARGIN_M,
                       "containment": all(contained.values())}}
    (out / "two_arm.json").write_text(json.dumps(result, indent=1))
    print(f"{a.scene}: room acc gt {gt['room_acc']:.2f} pred {pred['room_acc']:.2f} | "
          f"trans median gt {gt['trans_median_m']:.3f} pred {pred['trans_median_m']:.3f} m | "
          f"contained {sum(contained.values())}/{len(contained)} | pass {result['pass']}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Smoke-check the runner's plumbing on the smallest scene**

Run: `cd /home/ruoyu/PanoPin && python3 experiments/roomseg/two_arm.py --scene Area_2_manhattan4`
Expected: one summary line and `runs/roomseg_e2e/Area_2_manhattan4/two_arm.json`. If
`run_localization.py` fails, read `runs/roomseg_e2e/Area_2_manhattan4/<arm>/run.log` before changing
anything; the error rule applies.

Sanity check: the fresh gt arm's `trans_median_m` should be close to the 2026-10-01
`full_runs/Area_2_manhattan4` value (0.055 m). It is the same inputs, but it is **not** a pass criterion
(D38).

- [ ] **Step 6: Run the remaining scenes in the background**

Run with `run_in_background`:
`cd /home/ruoyu/PanoPin && for s in Area_2_manhattan7 Area_3_manhattan4 Area_3_manhattan6; do python3 experiments/roomseg/two_arm.py --scene $s; done`
Expected: one summary line per scene.

- [ ] **Step 7: Record the end-to-end results and commit**

Add a `## End-to-end (bar b, criteria 2–3)` section to `docs/roomseg-results.md`. Include:
- a table with one row per scene: room acc gt/pred, trans median gt/pred, contained n/N, pass flags;
- the per-pano rows of any failing scene;
- the verdict on bar (b) over all three criteria.

```bash
cd /home/ruoyu/PanoPin && git add experiments/roomseg docs/roomseg-results.md
git commit -m "eval(roomseg): two-arm end-to-end Stage-0 runs, predicted vs GT room clouds

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 10: Project docs

**Files:**
- Modify: `docs/DECISIONS.md` (append D39, D40), `docs/PROGRESS.md` (new top entry), `docs/tasks.json` (add T11)

- [ ] **Step 1: Append D39 and D40 to `docs/DECISIONS.md`**

```markdown
## D39 (2026-10-01) — Room segmentation = "enclosure" on a ceiling band (T11)

PanoPin now builds its own `{room: cloud}` candidates (`panopin.cli segment`). The method is a
5 cm band [ceiling − 0.60, ceiling − 0.10]: walls plus lintels, so open doors close and furniture
drops out. Rooms are the free-space connected components, slivers and specks are dropped, and every
footprint cell is filled to its nearest room within 1 m.

Why:
- The literature review (`docs/room-segmentation-literature-review.md`) found that the image, not
  the segmenter, decides the result on furnished scans with open doors.
- The 2026-10-01 probe closed every room on both dev scenes, where the HOV-SG band leaked.
- The approach is deterministic, CPU-only, has no Manhattan assumption, and needs no training.

Measured amendments (spec §4, §5, §6.3):
- ceiling = lowest peak of the top cluster (Area_2 ceilings at 2.58–2.81 m);
- a mid-height slab is refused as a second storey;
- B1 = the HOV-SG band, because full height is degenerate for the enclosure method.

Corridor convention A (S3DIS as shipped), with N panos as an upper bound on rooms (user decisions
2026-10-01). Results: `docs/roomseg-results.md`.

## D40 (2026-10-01) — The room-segmentation scorer uses numpy/scipy (deviation from D6)

`eval/roomseg_gt.py` / `eval/roomseg_score.py` score millions of points: KD-tree GT matching,
voxel majorities, Hungarian matching. Stdlib-only (D6) would be impractically slow. The original
harness (`eval/score.py`, `eval/metrics.py`) stays stdlib-only and untouched; the new scorer is a
separate module.
```

- [ ] **Step 2: Add T11 to `docs/tasks.json`**

Append to the `tasks` array, setting `"passes"` from the actual Task 7–9 results:
```json
{
 "id": "T11",
 "title": "Automatic room segmentation: one merged cloud -> {room: cloud} candidates (panopin.cli segment)",
 "status": "done",
 "passes": true,
 "verify": "conda run -n panopin python -m pytest tests/test_roomseg_*.py -q -> all pass; docs/roomseg-results.md: all 4 scenes K=G, F1@0.7=1.00, 0 splits/merges; two_arm.py pass on all 4 scenes",
 "notes": "Spec docs/specs/2026-10-01-room-segmentation-design.md, plan docs/plans/2026-10-01-room-segmentation.md, D39/D40. Next (separate spec): merge-to-K<=N and the segment<->PanoPin closed loop (review shortlist 3); wiring `segment` into Point_360 run_localization.py."
}
```
If any bar-(b) criterion failed, use `"status": "in_progress", "passes": false` and say which
criterion in `notes`.

- [ ] **Step 3: Add the PROGRESS entry**

At the top of `docs/PROGRESS.md` (under the header line), add a `## Current state (…)` block covering:
- what was built (`panopin.cli segment`, the scorer, `two_arm.py`);
- the dev, holdout and end-to-end verdicts, with the numbers copied from `docs/roomseg-results.md`;
- the Point_360 `loc_inputs.py:31` edit (untracked);
- **NEXT**:
  - merge-to-K ≤ N / closed loop (shortlist ③);
  - wiring into Point_360 Stage 0;
  - the whole-`Area_2.ply` stretch run.

- [ ] **Step 4: Run every roomseg test and the harness sanity check**

Run: `cd /home/ruoyu/PanoPin && conda run -n panopin python -m pytest tests/test_roomseg_*.py tests/test_cli.py -q && python tests/test_harness.py`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
cd /home/ruoyu/PanoPin && git add docs/DECISIONS.md docs/tasks.json docs/PROGRESS.md
git commit -m "docs: D39/D40, T11 and progress for automatic room segmentation

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```
