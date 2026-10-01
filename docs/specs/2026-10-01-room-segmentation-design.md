# Automatic room segmentation — design

**Date:** 2026-10-01
**Status:** design approved in conversation (3 sections + 2 decisions); written spec awaiting review; plan not yet written
**Task:** T11 (to be added to `docs/tasks.json`)
**Grounded in:** `docs/room-segmentation-literature-review.md`, including its §10.1 decisions

## 1. Context

Every PanoPin / Stage-0 result so far used the **S3DIS ground-truth room partition** as the candidate
list (`panopin.cli seed --clouds {room: GT room cloud}`). That covers the D36/D38 Area_3 acceptance
(0.086 m) and the 2026-10-01 Area_2 `full_runs`.

A real user hands the pipeline **one merged multi-room cloud** (TLS, 3–7 rooms, furnished, doors often
open) plus panoramas, and will not segment rooms by hand. PanoPin cannot run in that setting today.
This design makes PanoPin produce its own `{room: cloud}` candidates.

### 1.1 Findings that shape this design

- **What goes into the image matters more than the segmenter** (review §0.1). Projecting only a
  horizontal band just below the ceiling removes furniture. It also closes open doors, because the
  lintel above each door is continuous.
- **Probe, 2026-10-01** (throwaway; `scratchpad/probe/band_probe.py`):
  - band [ceiling − 0.60, ceiling − 0.10] m at 5 cm/px on `Area_2_manhattan7` and `Area_3_manhattan6`;
  - **every room is a closed outline**: offices, corridors, and the angled walls of
    `Area_3_manhattan6`;
  - the HOV-SG band [floor + 1.5, ceiling − 0.3] **leaks** (the top room joins the corridor);
  - [ceiling − 0.30, ceiling − 0.08] catches a lowered-ceiling patch in the Area_2 corridor;
  - ceiling fixtures appear as islands that do not touch walls.
- **Corridor convention A** (S3DIS as shipped; user decision): corridor pieces are separate rooms.
  The one such cut inside a test scene (Area_2 `hallway_2`|`hallway_3`, x = 7.77 m) is an annotated
  **door across the full corridor width** (`hallway_2/Annotations/door_1`, x 7.63–7.77). Its lintel
  shows in the band (z 2.06–2.51 m). So convention A needs no special rule there.
- **The panorama count N is an upper bound on the room count, not equal to it** (user decision): long
  corridors carry several panos.
- Measured on our data:
  - each merged scene `.ply` is the exact union of its room `.txt` files (identical point counts);
  - room clouds are disjoint, and walls are not duplicated across rooms;
  - each room holds only the inner face of its own walls.

## 2. Goals and non-goals

### Goals
1. `panopin.cli segment`: one merged cloud in → per-room `.txt` clouds + a `clouds.json` that
   `panopin.cli seed --clouds` consumes unchanged.
2. Deterministic: same cloud in, byte-identical files out.
3. One fixed parameter set in metres for every scene. Tuned on dev scenes only, frozen before the
   holdout is scored.
4. A scoring harness against the S3DIS partition, plus an end-to-end comparison against GT room clouds.

### Non-goals
- Merging down to K ≤ N, and the segment ↔ PanoPin closed loop (review ③). These are a later spec;
  this design only *reports* K vs N.
- The SAM approaches (review ②).
- Multi-storey clouds (refused, §5).
- Non-levelled clouds. Gravity is assumed to be +z, which TLS clouds are.
- Wiring `segment` into Point_360's `run_localization.py`. That is a small follow-up; acceptance runs
  call the two steps by hand.
- Changing `seed`, `eval/score.py` or `eval/metrics.py`.

## 3. Architecture

New subpackage `src/panopin/roomseg/`. Four units, each with one job and testable alone:

| Unit | Job | In → out |
|---|---|---|
| `levels.py` | storey heights | z values → `(floor_z, ceiling_z)`; raises `SegmentationError` (§5) |
| `raster.py` | metric grid | `Grid(origin, cell, shape)` with `cells(xy)` and `world(ij)`; `band_image(points, z0, z1)`; `footprint(points)` |
| `partition.py` | images → rooms | wall mask + footprint → label image (`int32`, 0 = none, 1..K) |
| `lift.py` | rooms → points | label image + grid + xyz → per-point label (`int32`, −1 = unassigned) |

- `roomseg/__init__.py` exposes `segment(xyz, params=DEFAULTS) -> (labels, report)`.
- `roomseg/params.py` holds one frozen dataclass `Params` with every constant in §4.
- Dependencies: numpy, scipy (`ndimage`), OpenCV. All are already in the `panopin` and `panopin-gpu`
  environments (numpy 1.23.5, scipy 1.10.1, cv2 5.0.0). **No new packages.**

### 3.1 CLI

```
python -m panopin.cli segment --cloud merged.{ply,txt} --out-dir D [--n-panos N]
```

Writes:
- `D/seg_00.txt … seg_{K-1}.txt`: `X Y Z R G B`, colours 0–255, **input coordinates**. This is
  CPO's `read_txt_pcd` format.
- `D/clouds.json`: `{"seg_00": "<abs path>", …}`, the input `seed --clouds` takes.
- `D/labels.npy`: per input point, in input order; −1 = unassigned.
- `D/segmentation.json`: heights, every `Params` value, per-segment area / point count / centroid,
  unassigned count, K, N, warnings.
- `D/rooms.png`: the label image, for a human.

**Input reading.**
- `.ply`: open3d; colours ×255.
- `.txt`: the S3DIS layout.
- Point order is preserved so that `labels.npy` aligns with the input.

**Segment naming.** Order is by point count, descending; ties are broken by centroid (x, then y). The
names (`seg_00`, …) carry no meaning, the same as anonymised pano ids.

## 4. Algorithm (Approach 1, "enclosure")

| Step | What | Parameter (start value) |
|---|---|---|
| 1 Heights | z histogram, 2 cm bins; strong peaks = local maxima ≥ 0.3 × the tallest bin, clustered when ≤ 0.3 m apart. Floor = strongest peak of the lowest cluster; ceiling = **lowest** peak of the highest cluster. *Amended 2026-10-01 (plan Task 1): Area_2's ceilings sit at 2.58–2.81 m, so the band must sit below the lowest of them; Area_2_manhattan4's desk peak (0.77 m) reaches 0.38 × the tallest bin.* | `z_bin` 0.02, `peak_rel` 0.30, `peak_cluster_gap` 0.30 |
| 2 Band image | per 5 cm cell, count points with z ∈ [ceiling − 0.60, ceiling − 0.10]; cells with ≥ `wall_min_pts` are **walls** (incl. lintels) | `cell` 0.05 m, `band` (0.60, 0.10) m, `wall_min_pts` 3 |
| 3 Footprint | cells with any point at any height; fill holes; morphological close | `footprint_close` 0.30 m |
| 4 Gap closing | morphological close of the wall mask (seals scan holes up to ~2r; must stay below half a door width, so door openings are sealed only by lintels) | `gap_close_r` 0.15 m |
| 5 Rooms | free = footprint ∧ ¬walls; 4-connected components | — |
| 6 Cleanup | drop a component if area < `min_room_area` or its max inscribed radius (EDT max) < `min_room_halfwidth` (slivers: door recesses, wall gaps) | 1.0 m², 0.30 m |
| 7 Fill | every footprint cell not in a kept room takes the nearest kept room (`distance_transform_edt(return_indices=True)`), **if that room is within `max_fill_dist`**. Otherwise the cell stays 0, e.g. outdoor points seen through a window. A shared wall splits down its middle, so each room keeps its inner wall face, as in S3DIS | `max_fill_dist` 1.0 m |
| 8 Lift | point label = label of its XY cell; points in label-0 cells = −1 (unassigned) | — |

- No Manhattan assumption anywhere. Connected components do not care about wall angles.
- **Tuning rule:** values may change **only on the dev scenes** (§6), and every change is logged in
  DECISIONS with the dev metrics before and after. They are frozen, with the commit hash recorded,
  before any holdout scene is scored.

## 5. Errors and warnings

**Fail loudly** (raise `SegmentationError`; the CLI exits non-zero and writes nothing) when:
- there is no clear floor or ceiling peak;
- the storey height falls outside 2–6 m (likely multi-storey or not levelled);
- a strong horizontal slab between floor + 1.8 m and ceiling − 0.5 m (another storey).
- zero rooms survive cleanup.

**Warn** (in `segmentation.json` `warnings` and on stderr; output still written) when:
- K > N, with `--n-panos` given (over-segmentation or a pano-less space);
- more than 2% of points are unassigned;
- any segment is under 2 m².

These warnings are the hooks the later merge / closed-loop step will act on.

**Known weak spots,** each stated in the report docs and expected to surface on the dev scenes:
- a wall gap wider than ~0.3 m with no wall above it (glass, a big scan hole) merges two rooms;
- a duct or soffit spanning a corridor's full width splits it;
- open-plan spaces;
- a room whose ceiling is lower than the scene's single ceiling height has its band inside the room
  volume, so it fills with points.

## 6. Evaluation

### 6.1 Scenes (`config/roomseg_scenes.json`)

| Split | Scene | Rooms | End-to-end panos (one per room) |
|---|---|---|---|
| dev | `Area_3_manhattan4` | hallway_1, office_3/5/7 | `Point_360/data/s3dis/_multiroom4/scene.json` stations |
| dev | `Area_3_manhattan6` | + office_4/6 | `_multiroom6/scene.json` stations |
| holdout | `Area_2_manhattan4` | hallway_2, office_6/7/8 | the panos of `data/full_runs/Area_2_manhattan4` |
| holdout | `Area_2_manhattan7` | + office_4/5, hallway_3 | the panos of `data/full_runs/Area_2_manhattan7` |

Merged clouds live in `Point_360/data/pointcloud/stanford/`. GT rooms are S3DIS v1.2 *non-aligned*
(the Aligned_Version rotates each room on its own).

**Fairness (as D5).** The segmenter reads only the merged cloud. Never the room `.txt` files, the
`Area_3/1.e57` per-room export, pose files or names.

### 6.2 Scorer (new; the existing harness is untouched)
- **`eval/roomseg_gt.py`.** Per-point GT label for a merged scene, by nearest-neighbour match to the
  room `.txt` files. It asserts point counts are equal and every match is within 1 mm. The labels are
  cached as `.npy`.
- **`eval/roomseg_score.py`.** Takes `labels.npy` + the GT labels. Computes on **5 cm voxels**, with
  each voxel's GT and predicted label taken by majority. Voxels within **0.2 m of a GT room boundary**
  are ignored. A voxel counts as near a boundary when any voxel of a *different* GT room lies within
  0.2 m of it in XY. Reports per scene:
  - one-to-one matching (Hungarian on IoU) with per-room IoU;
  - F1 at IoU 0.5 and 0.7;
  - PQ / SQ / RQ;
  - splits (no segment holds ≥ 80% of a GT room), naming the rooms;
  - merges (a segment holds ≥ 10% of each of ≥ 2 GT rooms), naming the rooms;
  - % unassigned;
  - **pano containment**: is each pano's GT camera XY inside the segment matched to its room? This
    is reported next to the GT partition's own containment (D17: some GT cameras sit outside their
    room cloud).
- Output: JSON + a markdown table.
- The scorer uses numpy/scipy, unlike the stdlib-only harness (D6). This is recorded as D40.

### 6.3 Baselines
- **B1:** the same pipeline with the full-height image (`band = full`). It measures what the band buys.
- **B0** (optional): the thesis route, i.e. 256² full-height image + SAM3 "floor plan", polygons
  lifted the same way. Scored once on all four scenes if the thesis environment still runs.

## 7. Testing (TDD, synthetic clouds generated in `tests/`)

1. Two rooms sharing a wall with a doorway under a lintel → 2 rooms. Each keeps its own wall face.
2. The same doorway open to the ceiling → 1 room (documents the open-plan limit).
3. A 0.2 m scan hole in a wall → still 2 rooms.
4. A corridor with a door (with lintel) across it → 2 rooms.
5. Furniture blocks below the band → result unchanged.
6. `levels` raises on a two-storey cloud and on a cloud with no ceiling.
7. `lift`:
   - a wall point goes to the nearest room;
   - a point farther than `max_fill_dist` from every room gets −1;
   - `labels.npy` stays aligned with input order.
8. Determinism: two runs → byte-identical `seg_*.txt`, `labels.npy`, `clouds.json`.
9. CLI: the files written parse with CPO's `read_txt_pcd`; `clouds.json` keys match the files;
   warnings fire for K > N.

## 8. Acceptance criteria (the user's bar "b")

1. **Room level, all four scenes:**
   - every GT room is matched one-to-one at IoU ≥ 0.7;
   - 0 splits and 0 merges;
   - K = number of GT rooms.

   Holdout is scored once, with parameters frozen at a recorded commit.
2. **Containment:** every end-to-end pano's GT camera lies inside its room's matched segment.
3. **End-to-end:** Stage-0 (PanoPin `seed` + upright-prior FGPL), one pano per room (§6.1), in two arms
   run **fresh by the same script**: predicted clouds vs GT room clouds. Historical numbers are not
   compared (D38: rebuilt geometry invalidates them). Pass per scene:
   - predicted-arm room accuracy ≥ GT-arm room accuracy;
   - predicted-arm translation median ≤ GT-arm median **+ 0.05 m**.

## 9. Task order (detailed in the plan)

1. Synthetic-scene helpers + `levels.py`.
2. `raster.py`.
3. `partition.py`.
4. `lift.py` + `segment()`.
5. CLI `segment`.
6. `roomseg_gt.py` + `roomseg_score.py`.
7. Dev scoring + tuning (B1 alongside).
8. Freeze + holdout scoring.
9. End-to-end two-arm runs.
10. Docs: DECISIONS D39 (method) / D40 (scorer dependencies), PROGRESS, `tasks.json` T11.

Optional: B0.

## 10. Risks

- **The band assumes one ceiling height per scene.**
  - Lowered ceilings or soffits inside a room can fill the band. They are visible as filled regions in
    `rooms.png` and as low K.
  - Mitigation, if dev shows it: estimate the ceiling per region of the footprint. This would be a
    design change and goes back to the user.
- **The S3DIS hallway convention is applied inconsistently elsewhere** (review §10.1). Only the Area_2
  cut was checked for a door; the Area_3 `hallway_1`|`hallway_2` cut was not, and it lies at the
  crop edge of the dev scenes.
- **Four scenes and 4–7 rooms each is a small test.**
  - One error moves F1 by 15–25 points, so tables are per scene.
  - The whole-`Area_2.ply` stretch run is a recommended follow-up, not acceptance.
- **The end-to-end medians rest on 4–7 panos.** The 0.05 m margin was chosen for that reason. The
  Area_2 GT-cloud runs gave 0.055 m and 0.459 m (an FGPL 180° flip).

## 11. Branching

Work on a new PanoPin branch `feat/room-segmentation`, off the current `docs/scan2bim-localization-spec`
HEAD. Nothing is committed or pushed without the user's go-ahead. Point_360's `external/PanoPin` pin is
**not** moved by this work.
