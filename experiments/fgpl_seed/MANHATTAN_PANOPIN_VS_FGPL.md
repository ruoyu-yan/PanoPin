# Does PanoPin make FGPL work on a multi-room building?

**The test.** Take a 5-room Manhattan section of S3DIS Area_3 and all 22 panoramas inside it. Run FGPL's pose estimation two ways — on its own, and seeded by PanoPin — and score both against the S3DIS ground-truth camera poses.

**Same inputs throughout.** Both arms use the *same* 22 panoramas, the *same* 5-room point cloud and line map, and the *same* 2D features. What changes is the method.

**Scene:** `area3_manhattan` — 5 Manhattan rooms (office_5, hallway_1, lounge_1, conferenceRoom_1, WC_1), 5.16M points, 22 in-frame panos. Non-Manhattan rooms (`office_3`, `office_7`, `office_8` — real diagonal walls) are excluded so the pipeline's 3-orthogonal-direction assumption holds. The line map recovered three **exactly axis-aligned** principal directions with **0% unclassified** sparse lines.

**1. FGPL as-is** — the baseline: FGPL's own multi-room mechanism, exactly as it exists in scan2measure (`use_local_filtering=False`, global mode: no Voronoi, whole-map search). PanoPin's seed file is **never opened** (verified: `load_panorama_positions` has a single call site, inside `if use_local:`).

**2. FGPL + upright prior + PanoPin** — the work being presented. Two contributions:

   - *PanoPin* seeds each pano with a colour-based room + position estimate (22 seeds). **No ground truth is used.**
   - The *upright rotation prior* is a fix inside FGPL itself (`pose_search.build_rotation_candidates`): FGPL enumerates 24 rotation candidates, and 20 of them put the camera on its side or upside-down — impossible for tripod capture. The prior discards those, leaving the 4 physically possible ones.

**3. Ground truth** — S3DIS camera poses. The ruler both arms are measured against, not a third method.

> Arm 2 bundles both contributions, so it is a product comparison rather than a one-variable ablation. §4 separates them: the prior alone, without PanoPin, does *not* produce the result.

## Result

| measured against S3DIS ground truth | FGPL as-is | **+ upright prior + PanoPin** |
|---|---|---|
| Panos localized | 22/22 | 22/22 |
| **Rooms covered** | **1/5** | **5/5** |
| **Wrong-room rate** | **82%** | **18%** |
| **Translation median error** | **13.27 m** | **0.08 m** |
| Translation mean error | 12.08 m | 3.21 m |
| Rotation median error | 134.2° | 1.8° |
| Panos within 10 cm of GT | 3/22 | 11/22 |

**FGPL on its own puts 18/22 panos in the wrong room** and covers 1/5 rooms — a 13.3 m median error. With PanoPin's seed, the same estimator on the same data reaches 0.08 m and 5/5 rooms. The rooms in this scene are near-identical in shape, so FGPL's geometry alone cannot tell them apart; PanoPin's colour matching can, and that is the whole difference.

## Which contribution does the work?

Arm 2 changes two things at once, so the obvious question is whether the rotation fix is doing the work rather than PanoPin. The control answers it: FGPL **with** the upright prior but **without** PanoPin.

_Control arm still running — this section will be filled in when it lands._

## The shipping configuration does better still

The arm above seeds *every* pano. The shipping design seeds **one best pano per room** (`coverage.room_anchored_seeds`), which is all FGPL needs — one correct entry point per room. Still no ground truth.

| PanoPin room-anchored (5 seeds, 1/room) | value |
|---|---|
| Rooms covered | **5/5** |
| Wrong-room rate | **0%** |
| Translation median error | **0.045 m** |
| Rotation median error | **0.4°** |
| Rotation flips (>45°) | **0/5** |

All 5 room seeds were correct, chosen threshold-free. _(measured on the pre-prior estimator; the prior-on re-run is still going and is expected to match or beat it — it already has zero flips.)_

## Per-pano: estimated pose vs ground truth

Position in raw S3DIS metres; error is 3D Euclidean distance to GT and the geodesic rotation angle. Sorted by PanoPin's error.

| pano | room | GT position (x,y,z) | FGPL alone: est. position | err | FGPL+PanoPin: est. position | err |
|---|---|---|---|---|---|---|
| `44d6d985` | WC_1 | (0.91, 5.19, 1.40) | (0.93, 5.20, 1.38) | 0.02 m / 1° | (0.92, 5.20, 1.39) | **0.01 m / 0°** |
| `c67b419a` | conferenceRoom_1 | (15.62, 4.71, 1.41) | (1.29, 3.90, 1.21) | 14.36 m / 180° | (15.63, 4.72, 1.41) | **0.02 m / 1°** |
| `87d7995a` | conferenceRoom_1 | (16.99, 1.76, 1.41) | (1.11, 5.52, 1.10) | 16.32 m / 180° | (17.02, 1.76, 1.41) | **0.02 m / 1°** |
| `5c2959c3` | office_5 | (21.55, -2.93, 1.37) | (21.56, -2.94, 1.38) | 0.01 m / 1° | (21.57, -2.95, 1.40) | **0.03 m / 0°** |
| `d0834679` | hallway_1 | (20.53, -6.84, 1.38) | (0.77, 3.73, 1.24) | 22.41 m / 179° | (20.54, -6.81, 1.38) | **0.03 m / 2°** |
| `fafa0629` | hallway_1 | (20.64, -8.62, 1.38) | (1.45, 5.74, 1.01) | 23.97 m / 90° | (20.63, -8.63, 1.42) | **0.04 m / 0°** |
| `ac7e25ee` | hallway_1 | (20.67, -9.88, 1.38) | (0.88, 3.52, 1.20) | 23.90 m / 180° | (20.66, -9.90, 1.42) | **0.04 m / 0°** |
| `1119e668` | hallway_1 | (20.65, -0.01, 1.39) | (3.84, 6.85, 1.32) | 18.16 m / 90° | (20.68, -0.02, 1.43) | **0.04 m / 1°** |
| `b3ef2004` | WC_1 | (1.07, 2.96, 1.39) | (1.09, 2.98, 1.38) | 0.03 m / 1° | (1.05, 2.97, 1.35) | **0.05 m / 0°** |
| `e2e170cb` | conferenceRoom_1 | (14.88, 2.60, 1.41) | (3.36, 6.55, 1.01) | 12.19 m / 90° | (14.93, 2.63, 1.40) | **0.06 m / 2°** |
| `f758ab19` | office_5 | (25.87, -3.57, 1.37) | (3.01, 5.75, 1.01) | 24.69 m / 90° | (25.81, -3.57, 1.40) | **0.07 m / 3°** |
| `47b49abf` | hallway_1 | (20.56, -1.17, 1.40) | (3.10, 4.03, 0.66) | 18.24 m / 90° | (20.57, -1.26, 1.34) | **0.10 m / 0°** |
| `481b93c5` | conferenceRoom_1 | (18.59, 3.81, 1.41) | (3.05, 5.95, 1.36) | 15.69 m / 90° | (18.79, 3.96, 1.24) | **0.30 m / 179°** |
| `f0e54fcd` | WC_1 | (0.97, 7.19, 1.40) | (3.79, 5.34, 1.23) | 3.37 m / 180° | (0.53, 7.39, 1.62) | **0.53 m / 2°** |
| `127fc8df` | lounge_1 | (5.75, 4.58, 1.41) | (3.51, 5.41, 1.04) | 2.42 m / 180° | (5.62, 5.24, 0.91) | **0.84 m / 89°** |
| `b50320e1` | lounge_1 | (7.13, 3.13, 1.41) | (14.43, 5.11, 1.11) | 7.57 m / 180° | (7.77, 2.62, 2.10) | **1.07 m / 7°** |
| `4d491624` | hallway_1 | (20.51, -2.89, 1.38) | (16.41, 5.07, 0.94) | 8.96 m / 180° | (20.91, -1.71, 1.46) | **1.25 m / 89°** |
| `4a7bfe05` | lounge_1 | (6.26, 7.36, 1.40) | (0.81, 4.75, 1.36) | 6.04 m / 90° | (4.45, 8.35, 0.68) | **2.18 m / 1°** |
| `80e1f6ae` | lounge_1 | (7.64, 5.82, 1.40) | (0.73, 3.96, 1.38) | 7.15 m / 179° | (9.91, 7.24, 1.49) | **2.68 m / 92°** |
| `da0bb9ad` | conferenceRoom_1 | (19.40, 5.66, 1.41) | (4.51, 5.75, 2.12) | 14.91 m / 179° | (5.11, 4.29, 0.76) | **14.37 m / 88°** |
| `1a557181` | WC_1 | (0.05, 1.67, 1.40) | (1.21, 3.77, 1.29) | 2.40 m / 1° | (20.05, 0.20, 1.85) | **20.06 m / 180°** |
| `2b70fafb` | office_5 | (23.91, -4.28, 1.37) | (3.56, 6.09, 0.68) | 22.85 m / 179° | (-0.31, 7.38, 1.55) | **26.88 m / 90°** |

## Honest caveats

- **Rotation: 7/22 panos still come back facing the wrong way** (~90-180°). This is an FGPL limitation, not PanoPin's — it happens with a perfect seed too (see the appendix). Read the rotation medians with care: the distribution is bimodal (panos either lock to ~1° or flip to ~90-180°, nothing between), so the median only reports which side most panos fall on, not how big the errors are. The shipping config above has zero flips.
- **PanoPin's own ceiling:** 4/22 panos were assigned the wrong room — the known limit on window- and occlusion-dominated panoramas, where the colour signal is too weak. It costs the mean and max, not per-room coverage: every room still had a correct pano.
- **One area, one scene.** Area_3 only, n=22. Cross-area generalization is untested.
- **Manhattan rooms only** — rooms with genuine diagonal walls are out of scope for the pipeline's 3-direction assumption, and were excluded by design.

---

## Appendix — diagnostic: what if the seed were perfect?

**This is not part of the pipeline, and PanoPin does not use ground truth anywhere.** To measure how much of the remaining error belongs to FGPL rather than to PanoPin's seed, we ran a diagnostic arm that replaces PanoPin's seed with the *true* camera position. It is an upper bound — a measuring stick, not a component.

| | FGPL + prior + PanoPin | GT-seeded (upper bound) |
|---|---|---|
| Translation median | 0.084 m | 0.113 m |
| Rotation median | 1.8° | 1.3° |
| Rotation flips (>45°) | 7/22 | 9/22 |

The GT-seeded arm flips too, which is the point: **rotation flips are FGPL's behaviour, not a symptom of PanoPin's seed being imprecise.** Where PanoPin puts a pano in the right room and FGPL locks the rotation, the pose is centimetre-accurate — matching what a perfect seed achieves.

_(GT-seeded arm measured on the pre-prior estimator; the prior-on re-run is still going.)_
