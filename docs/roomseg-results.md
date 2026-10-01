# Room segmentation (T11) — results

Spec `docs/specs/2026-10-01-room-segmentation-design.md` · plan `docs/plans/2026-10-01-room-segmentation.md`.
All numbers from `eval/roomseg_score.py` (5 cm voxels, 0.2 m boundary band ignored).

## Dev (Area_3) — method
Both scenes: floor −0.01 m, ceiling 2.71 m, band [2.11, 2.61] m, no warnings.

### Area_3_manhattan4 (dev) — runs/roomseg/Area_3_manhattan4

K=4 vs G=4 · F1@0.5 1.00 · F1@0.7 1.00 · PQ 0.999 · unassigned 0.00% · splits none · merges none

| GT room | segment | IoU |
|---|---|---|
| hallway_1 | seg_00 | 0.997 |
| office_3 | seg_02 | 1.000 |
| office_5 | seg_03 | 0.997 |
| office_7 | seg_01 | 1.000 |

containment: pred 14/15, GT partition itself 15/15

### Area_3_manhattan6 (dev) — runs/roomseg/Area_3_manhattan6

K=6 vs G=6 · F1@0.5 1.00 · F1@0.7 1.00 · PQ 0.998 · unassigned 0.00% · splits none · merges none

| GT room | segment | IoU |
|---|---|---|
| hallway_1 | seg_00 | 0.998 |
| office_3 | seg_03 | 0.999 |
| office_4 | seg_04 | 0.997 |
| office_5 | seg_01 | 0.996 |
| office_6 | seg_05 | 1.000 |
| office_7 | seg_02 | 1.000 |

containment: pred 18/19, GT partition itself 19/19

Both dev scenes pass spec §8 criterion 1 (room level): K = G, F1@0.7 = 1.00, and there are no splits or
merges. They also pass criterion 2 (containment): every end-to-end station's camera lies inside its matched
segment (see the containment note).

## Dev — baseline B1 (HOV-SG band)
Band [1.49, 2.41] m.

### Area_3_manhattan4 (dev) — runs/roomseg_b1/Area_3_manhattan4

K=4 vs G=4 · F1@0.5 1.00 · F1@0.7 1.00 · PQ 0.999 · unassigned 0.00% · splits none · merges none

| GT room | segment | IoU |
|---|---|---|
| hallway_1 | seg_00 | 0.997 |
| office_3 | seg_02 | 1.000 |
| office_5 | seg_03 | 0.997 |
| office_7 | seg_01 | 1.000 |

containment: pred 14/15, GT partition itself 15/15

### Area_3_manhattan6 (dev) — runs/roomseg_b1/Area_3_manhattan6

K=6 vs G=6 · F1@0.5 1.00 · F1@0.7 1.00 · PQ 0.996 · unassigned 0.00% · splits none · merges none

| GT room | segment | IoU |
|---|---|---|
| hallway_1 | seg_00 | 0.995 |
| office_3 | seg_03 | 0.997 |
| office_4 | seg_04 | 0.993 |
| office_5 | seg_01 | 0.996 |
| office_6 | seg_05 | 0.999 |
| office_7 | seg_02 | 0.999 |

containment: pred 18/19, GT partition itself 19/19

B1 did **not** leak on `Area_3_manhattan6`, unlike in the probe. On the dev scenes it equals the method:
F1@0.7 1.00 on both, and the same containment. Its IoUs are slightly lower (manhattan6 PQ 0.996 vs 0.998).
The dev scenes therefore do not show what the ceiling band buys over the HOV-SG band.

## Containment note
- **All panos.** Containment over every pano in the scene pose file is pred 14/15 (manhattan4) and
  18/19 (manhattan6). The GT partition itself scores 15/15 and 19/19. The method and B1 are the same.
- **The sole miss is `camera_5c2959c3…_office_5`, at XY (21.555, −2.931).** It stands in a door alcove
  about 0.55 m deep between the hallway wall (x ≈ 21.2) and the office_5 wall (x ≈ 21.9), under a door
  head at z ≈ 2.17 m. That head lies in the band, so the alcove counts as wall. `fill()` then splits the
  alcove at the nearest-room midpoint, x ≈ 21.575. The GT labels the whole alcove office_5. The camera
  ends up 2 cm on the hallway side: within 0.15 m, 969 points are hallway and 467 are office_5.
- **This camera is not an end-to-end station.** Point_360's `choose_stations.py` rejects it as standing
  outside office_5's ceiling outline.
- **End-to-end stations (spec §8 criterion 2).** All are inside their matched segment, for the method and
  for B1. The manhattan4 stations are d0834679 hallway_1, bbda2627 office_3, f758ab19 office_5 and
  66dbab01 office_7: 4/4. The manhattan6 stations are those four plus 995ce724 office_4 and 249f2e52
  office_6: 6/6.
- The plan's Task 7 pass rule (all panos) was stricter than the spec, and the spec governs.

## Tuning log
No changes: defaults passed both dev scenes.

## Frozen parameters
`Params()` defaults in `src/panopin/roomseg/params.py`:

| parameter | value | parameter | value |
|---|---|---|---|
| z_bin | 0.02 | footprint_close | 0.30 |
| peak_rel | 0.30 | wall_min_pts | 3 |
| peak_cluster_gap | 0.30 | gap_close_r | 0.15 |
| min_storey | 2.0 | min_room_area | 1.0 |
| max_storey | 6.0 | min_room_halfwidth | 0.30 |
| mid_slab_above_floor | 1.8 | max_fill_dist | 1.0 |
| mid_slab_below_ceiling | 0.5 | warn_unassigned_frac | 0.02 |
| cell | 0.05 | warn_small_room_area | 2.0 |
| band_bottom | 0.60 | band_from_floor | None |
| band_top | 0.10 | max_grid_cells | 40,000,000 |

B1 = `HOVSG` = the defaults with `band_from_floor=1.5`, `band_top=0.30`.

Frozen at commit c8b8928.

## Holdout (Area_2) — scored once at 42be7e1
Parameters frozen at c8b8928 (`params.py` unchanged since); one run per scene, no tuning.
Area_2_manhattan4: floor 0.08 m, ceiling 2.64 m, band [2.04, 2.54] m. Area_2_manhattan7: floor −0.01 m,
ceiling 2.59 m, band [1.99, 2.49] m. No warnings on either.

### Area_2_manhattan4 (holdout) — runs/roomseg/Area_2_manhattan4

K=4 vs G=4 · F1@0.5 1.00 · F1@0.7 1.00 · PQ 1.000 · unassigned 0.00% · splits none · merges none

| GT room | segment | IoU |
|---|---|---|
| hallway_2 | seg_00 | 1.000 |
| office_6 | seg_02 | 1.000 |
| office_7 | seg_03 | 0.999 |
| office_8 | seg_01 | 1.000 |

containment: pred 16/16, GT partition itself 16/16

### Area_2_manhattan7 (holdout) — runs/roomseg/Area_2_manhattan7

K=7 vs G=7 · F1@0.5 1.00 · F1@0.7 1.00 · PQ 0.994 · unassigned 0.00% · splits none · merges none

| GT room | segment | IoU |
|---|---|---|
| hallway_2 | seg_03 | 1.000 |
| hallway_3 | seg_00 | 1.000 |
| office_4 | seg_02 | 0.979 |
| office_5 | seg_01 | 0.980 |
| office_6 | seg_05 | 1.000 |
| office_7 | seg_06 | 0.999 |
| office_8 | seg_04 | 1.000 |

containment: pred 36/37, GT partition itself 35/37

### Holdout verdict
Both holdout scenes pass spec §8 criterion 1 (room level): K = G, F1@0.7 = 1.00, no splits or merges, every
GT room matched at IoU ≥ 0.7.
- **End-to-end stations, criterion 2 (containment)** (keys of Point_360 `data/full_runs/<scene>/stage0_localization/key_map.json`):
  manhattan4 4/4 inside their matched segment (123cfbc1 office_6, 618991c8 office_7, eeebfd9d office_8,
  12ea36aa hallway_2); manhattan7 7/7 (those four plus 243d8e79 office_4, 0c9d4638 office_5, c5377d49
  hallway_3).
- **Corridor door (convention A).** The `hallway_2`|`hallway_3` door at x = 7.77 is honoured: the corridor
  comes out as two segments (hallway_2 → seg_03, hallway_3 → seg_00, both IoU 1.000). In `rooms.png` the
  horizontal corridor and the vertical corridor are separate colours.
- **All-pano containment (reported, not gated).** The one predicted miss in manhattan7 is
  `camera_e70395f2…_hallway_3`, which falls in seg_03 (hallway_2). The GT partition itself also puts this
  camera outside hallway_3, as it does `camera_5a7060af…_hallway_3` (which the method places inside). Neither
  is an end-to-end station.

## End-to-end (bar b, criteria 2–3)
`experiments/roomseg/two_arm.py`: Stage 0 (PanoPin seed + upright-prior FGPL, Point_360
`run_localization.py --pose-source panopin --tau 0.10`) run fresh twice per scene with the same command.
The only difference is `--clouds-json`: **gt** = the S3DIS room clouds, **pred** = `runs/roomseg/<scene>/clouds.json`.
Outputs are in `runs/roomseg_e2e/<scene>/{gt,pred}/` and `two_arm.json`. Pass rule: pred room acc ≥ gt, pred translation
median ≤ gt + 0.05 m, every e2e pano contained in its matched segment. The Area_3 Stage-0 inputs were built
for this task (`experiments/roomseg/area3_stage0_selection.py` → Point_360 `data/full_runs/_scripts/loc_inputs.py`,
with one station per room from the `_multiroom{4,6}/scene.json` manifests).

| scene | room acc gt / pred | trans median gt / pred (m) | contained | room_acc | trans_median | containment |
|---|---|---|---|---|---|---|
| Area_3_manhattan4 (dev) | 1.00 / 1.00 | 1.893 / 1.893 | 4/4 | ✔ | ✔ | ✔ |
| Area_3_manhattan6 (dev) | 1.00 / 1.00 | 1.243 / 1.111 | 6/6 | ✔ | ✔ | ✔ |
| Area_2_manhattan4 (holdout) | 1.00 / 1.00 | 0.658 / 0.885 | 4/4 | ✔ | **✘** | ✔ |
| Area_2_manhattan7 (holdout) | 1.00 / 1.00 | 0.053 / 0.038 | 7/7 | ✔ | ✔ | ✔ |

### Area_2_manhattan4: per-pano rows (fails trans_median)

| pano | room | gt label | gt err (m) | pred label | pred err (m) |
|---|---|---|---|---|---|
| 123cfbc1 | office_6 | office_6 | 0.045 | seg_02 | 0.041 |
| 618991c8 | office_7 | office_7 | 0.036 | seg_03 | 0.047 |
| eeebfd9d | office_8 | office_8 | 1.272 | seg_01 | 1.724 |
| 12ea36aa | hallway_2 | hallway_2 | 1.869 | seg_00 | 2.200 |

With 4 panos the median is the mean of the 2nd and 3rd errors, so the two FGPL failures (office_8, hallway_2)
set it in both arms. Both arms pick every room correctly (4/4). The pred arm's errors are larger only on the
two panos FGPL already fails in the gt arm.

**A fresh run did not reproduce the 2026-10-01 `full_runs` result on identical inputs.** The fresh gt arm
(median 0.658 m) does not reproduce the 2026-10-01 `full_runs/Area_2_manhattan4` run (0.055 m) on the same
inputs and config (the `pose_config.json`
files differ only in their paths). There, hallway_2 was 0.019 m off; here it is 1.869 m off. The PanoPin
seed scores (hallway_2 0.048 → 0.091) and the FGPL ICP inlier counts change between runs, even though the
preflight reports "PanoPin determinism fix present". This is one fresh run against one historical run, so
it does not yet show that Stage 0 is non-deterministic. There is counter-evidence: on Area_3_manhattan4 the
two fresh arms, with different clouds, gave identical per-pano errors (see below). The cheap test that would
settle it is a second fresh GT-arm run on Area_2_manhattan4 (PROGRESS NEXT item 1). If the noise is real
(> 1 m on single panos), it is much larger than the 0.05 m margin, and the trans_median check on 4–7 panos
would not be a reliable test of the segments.

### Other per-pano notes
- Area_3_manhattan4: the gt and pred errors are identical per pano (hallway_1 3.257, office_3 2.850,
  office_5 0.058, office_7 0.937 m). FGPL fails on 3 of 4 Area_3 stations whatever the clouds are.
- Area_3_manhattan6: gt hallway_1 1.786 / office_3 2.888 / office_4 1.083 / office_5 0.060 / office_6 1.257 /
  office_7 1.229 m; pred 1.177 / 2.811 / 1.043 / 0.065 / 1.248 / 1.045 m. FGPL again fails in both arms.
- Area_2_manhattan7: gt office_5 2.039, hallway_2 4.244, hallway_3 4.743 m (others < 0.06); pred office_5
  2.020, office_8 0.826, hallway_3 4.451 m (others < 0.04).

### Bar (b) verdict
- Criterion 1 (room level, earlier sections): passes on all four scenes.
- Criterion 2 (containment): 4/4 scenes pass (every e2e pano contained).
- Criterion 3 (end-to-end): room accuracy pred ≥ gt on 4/4; translation median within gt + 0.05 m on 3/4 —
  **fails** on Area_2_manhattan4 (0.885 > 0.658 + 0.05). On the two Area_3 scenes FGPL fails on most
  stations in both arms, so those two passes compare failure with failure and carry little information.

**Bar (b) is not met** (criterion 3 fails on Area_2_manhattan4). The suspected cause, not yet confirmed (see
above), is that Stage 0 is not reproducible run to run, which this plan did not investigate: on identical
inputs the fresh GT arm gives 0.658 m against 0.055 m in the 2026-10-01 full_runs (hallway_2 0.019 → 1.869 m). Area_2_manhattan7 points the
same way: the same hallway_2 pano is 4.244 m off in the GT arm and 0.035 m off in the pred arm, a 4 m swing in
pred's favour that fits run noise rather than a segmentation effect.

## Known weak spots
From spec §5, plus one found since:
- A wall gap wider than ~0.3 m with no wall above it (glass, a big scan hole) merges two rooms.
- A duct or soffit spanning a corridor's full width splits it.
- Open-plan spaces.
- A room whose ceiling is lower than the scene's single ceiling height has its band inside the room
  volume, so it fills with points.
- **New:** a room whose ceiling is more than ~0.6 m below the scene ceiling has no walls in the band and
  merges SILENTLY with its neighbours (no warning fires; only K > N warns).
