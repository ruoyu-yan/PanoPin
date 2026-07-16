# PanoPin + FGPL vs FGPL alone — pose accuracy on Manhattan rooms

**Question:** does PanoPin's colour-based room seeding actually make FGPL work on a multi-room building of same-shape rooms?

**Setup — one variable.** All arms use the *same* 22 panoramas, the *same* 5-room point cloud and line map, the *same* 2D features, and the *same* estimator (`multiroom_pose_estimation.py`). The only thing that changes is how FGPL is seeded. Poses are scored against S3DIS ground truth.

**Scene:** `area3_manhattan` — 5 Manhattan rooms (office_5, hallway_1, lounge_1, conferenceRoom_1, WC_1), 5.16M points, 22 in-frame panos. Non-Manhattan rooms (`office_3`, `office_7`, `office_8` — real diagonal walls) are excluded so the pipeline's 3-orthogonal-direction assumption holds. The line map recovered three **exactly axis-aligned** principal directions with **0% unclassified** sparse lines.

- *FGPL alone* = `use_local_filtering=False`, FGPL's original global mode. No Voronoi; PanoPin's seed file is **never opened** (verified: `load_panorama_positions` has one call site, inside `if use_local:`). This is FGPL's own multi-room mechanism.
- *FGPL + PanoPin* = one PanoPin colour seed per pano (22 seeds).
- *FGPL + GT seed* = ground-truth camera position per pano — the upper bound for this map, showing what the refiner can do with a perfect seed.

## Headline

| | FGPL alone | **FGPL + PanoPin** | FGPL + GT seed |
|---|---|---|---|
| Panos localized | 22/22 | 22/22 | 22/22 |
| **Rooms covered** | **1/5** | **5/5** | **5/5** |
| **Wrong-room rate** | **82%** | **18%** | **9%** |
| **Translation median** | **13.27 m** | **0.96 m** | **0.11 m** |
| Translation mean | 12.08 m | 3.42 m | 0.58 m |
| Rotation median | 134.2° | 89.9° | 1.3° |
| Panos within 10 cm | 3/22 | 10/22 | 10/22 |

**FGPL alone puts 18/22 panos in the wrong room** and covers only 1/5 rooms — a 13.3 m median error. Adding PanoPin's seed takes the same estimator, on the same data, to 0.96 m and 5/5 rooms. That gap is what PanoPin contributes.

## The deployment configuration is better still

The 22-seed arm above seeds *every* pano. The shipping design (`coverage.room_anchored_seeds`) instead seeds **one best pano per room** — and that is what the pipeline actually needs, since FGPL only needs one correct entry point per room.

| PanoPin room-anchored (5 seeds, 1/room) | value |
|---|---|
| Rooms covered | **5/5** |
| Wrong-room rate | **0%** |
| Translation median | **0.045 m** |
| Rotation median | **0.4°** |
| Rotation flips | **0/5** |

All 5 room seeds were correct (room-anchored selection is threshold-free). **Survey-grade: 4.5 cm median, no flips, every room covered.**

## Per-pano detail — estimated vs ground truth

Translation error in metres, rotation error in degrees (geodesic). Sorted by FGPL+PanoPin error.

| pano | room | FGPL alone | FGPL + PanoPin | FGPL + GT seed | GT position (x,y,z) |
|---|---|---|---|---|---|
| `c67b419a` | conferenceRoom_1 | 14.36 m / 180° | 0.02 m / 1° | 0.02 m / 1° | (15.62, 4.71, 1.41) |
| `44d6d985` | WC_1 | 0.02 m / 1° | 0.02 m / 0° | 0.02 m / 0° | (0.91, 5.19, 1.40) |
| `87d7995a` | conferenceRoom_1 | 16.32 m / 180° | 0.02 m / 1° | 0.05 m / 1° | (16.99, 1.76, 1.41) |
| `5c2959c3` | office_5 | 0.01 m / 1° | 0.03 m / 0° | 0.03 m / 1° | (21.55, -2.93, 1.37) |
| `47b49abf` | hallway_1 | 18.24 m / 90° | 0.04 m / 0° | 0.86 m / 120° | (20.56, -1.17, 1.40) |
| `fafa0629` | hallway_1 | 23.97 m / 90° | 0.04 m / 1° | 0.09 m / 1° | (20.64, -8.62, 1.38) |
| `ac7e25ee` | hallway_1 | 23.90 m / 180° | 0.04 m / 0° | 0.04 m / 0° | (20.67, -9.88, 1.38) |
| `1119e668` | hallway_1 | 18.16 m / 90° | 0.04 m / 1° | 1.36 m / 120° | (20.65, -0.01, 1.39) |
| `b3ef2004` | WC_1 | 0.03 m / 1° | 0.07 m / 1° | 0.05 m / 1° | (1.07, 2.96, 1.39) |
| `f758ab19` | office_5 | 24.69 m / 90° | 0.07 m / 3° | 0.12 m / 3° | (25.87, -3.57, 1.37) |
| `b50320e1` | lounge_1 | 7.57 m / 180° | 0.86 m / 179° | 0.89 m / 178° | (7.13, 3.13, 1.41) |
| `f0e54fcd` | WC_1 | 3.37 m / 180° | 1.06 m / 120° | 0.05 m / 1° | (0.97, 7.19, 1.40) |
| `d0834679` | hallway_1 | 22.41 m / 179° | 1.34 m / 179° | 1.34 m / 179° | (20.53, -6.84, 1.38) |
| `4d491624` | hallway_1 | 8.96 m / 180° | 1.39 m / 180° | 1.39 m / 180° | (20.51, -2.89, 1.38) |
| `e2e170cb` | conferenceRoom_1 | 12.19 m / 90° | 1.42 m / 120° | 1.42 m / 120° | (14.88, 2.60, 1.41) |
| `481b93c5` | conferenceRoom_1 | 15.69 m / 90° | 1.74 m / 89° | 0.03 m / 1° | (18.59, 3.81, 1.41) |
| `127fc8df` | lounge_1 | 2.42 m / 180° | 1.79 m / 178° | 0.10 m / 1° | (5.75, 4.58, 1.41) |
| `4a7bfe05` | lounge_1 | 6.04 m / 90° | 1.89 m / 179° | 1.33 m / 90° | (6.26, 7.36, 1.40) |
| `80e1f6ae` | lounge_1 | 7.15 m / 179° | 2.05 m / 179° | 2.05 m / 179° | (7.64, 5.82, 1.40) |
| `da0bb9ad` | conferenceRoom_1 | 14.91 m / 179° | 15.34 m / 178° | 0.43 m / 91° | (19.40, 5.66, 1.41) |
| `1a557181` | WC_1 | 2.40 m / 1° | 19.94 m / 91° | 0.04 m / 1° | (0.05, 1.67, 1.40) |
| `2b70fafb` | office_5 | 22.85 m / 179° | 26.03 m / 120° | 1.13 m / 1° | (23.91, -4.28, 1.37) |

## Honest caveats

- **Rotation is FGPL's weak point, not PanoPin's.** The GT-seeded arm still flips 9/22 panos ~90-180°, so a perfect seed does not prevent it. Cause: with 22 seeds in 5 rooms the Voronoi subdivides each room's 3D lines until the rotation search cannot tell the true rotation from its 180° twin. A controlled re-run of aliased panos with 4 seeds instead of 22 (identical seed positions) recovered 3/4. The deployment config (5 seeds) has zero flips.
- **One area, one scene.** Area_3 only, n=22. Cross-area generalization is untested.
- **PanoPin's own ceiling:** 4/22 panos got a wrong room (the known weak-lock limit — window/occlusion-dominated panos). It costs the mean/max, not per-room coverage: every room still had a correct pano.
- The `tau=0.10` confidence gate tuned on an earlier pool does NOT transfer here; the threshold-free room-anchored seeding does.
