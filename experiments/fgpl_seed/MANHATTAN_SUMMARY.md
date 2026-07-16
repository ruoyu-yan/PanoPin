# Strictly-Manhattan PanoPin -> FGPL: configuration comparison

Scene `area3_manhattan`: 5 Manhattan rooms (office_5, hallway_1, lounge_1, conferenceRoom_1, WC_1), 22 in-frame panos, 5.16M-point map built from Manhattan rooms only. Non-Manhattan rooms (office_3/7/8) excluded per Point_360 `roadmap.md` §5. The line map recovered 3 exactly axis-aligned principal directions with 0% unclassified sparse lines.

All arms share the same panos, map, features and estimator. **The only variable is how FGPL is seeded.**

| configuration | panos posed | rooms covered | trans median | rot median | rotation locked | wrong-room |
|---|---|---|---|---|---|---|
| FGPL alone (no PanoPin) | 22 | 1/5 | 13.273 m | 134.2° | 4/22 | 0.82 |
| PanoPin, 22 seeds (1/pano) | 22 | 5/5 | 0.960 m | 89.9° | 10/22 | 0.18 |
| GT seed, 22 seeds (reference) | 22 | 5/5 | 0.113 m | 1.3° | 13/22 | 0.09 |
| PanoPin, 5 seeds (1/room) = DEPLOYMENT | 5 | 5/5 | 0.045 m | 0.4° | 5/5 | 0.00 |

## The three findings

**1. PanoPin is what makes multi-room work.** FGPL's own global search puts 18/22 panos in the WRONG ROOM (13.3 m median, 1/5 rooms covered) on same-shape rooms. PanoPin's seeds take that to 0.96 m and 5/5. This also refutes the 'give FGPL more search area' idea: global mode IS that idea's maximum, and it is catastrophic.

**2. The deployment config is survey-grade.** One seed per room (D32 room-anchored) gives **5/5 rooms, 0 wrong rooms, 0 rotation flips, 4.5 cm median**. The 22-seed arm was an unnaturally hard configuration: 22 seeds subdivide 5 rooms until each pano's Voronoi cell is starved of the 3D lines the rotation search needs.

**3. Pano crowding causes the 180° flips — mostly.** Controlled test (`manhattan_aliased_solo`): 4 previously-aliased panos, IDENTICAL seed positions, only the number of competing panos changed (22 -> 4). **3/4 now lock** (+`127fc8df` in the anchored arm = 4/5 overall), with geometry rising 3-15x (e.g. `80e1f6ae` 99->302 dense lines: 179.4° -> 0.3°).

### Caveats — state these aloud

- **Not universal:** `d0834679` still flips at 179.7° with 481 dense lines (15x more). Some panos genuinely alias; de-crowding is necessary, not sufficient.
- **Rotation lock does NOT imply cm position when cells are large.** `f0e54fcd` locks at 1.1° but sits 1.80 m off. Small cells pin translation while starving rotation; large cells free rotation but loosen translation. It is a trade-off, not a monotonic win.
- **The anchored arm poses 5 of 22 panos** — it is the answer for per-ROOM coverage, not for posing every pano. Those are different deliverables.
- **Weak panos stay weak:** the solo arm (previously-aliased panos) medians 0.923 m even with a full room, vs the anchored arm's 0.045 m (highest-confidence pano per room).
- Area_3, n=22, one scene. `tau=0.10` does NOT transfer to this pool (admits 21/22 incl. 3 wrong-room); threshold-free room-anchored seeding (5/5 correct) is the claim that holds.

## Actionable: FGPL already computes a rotation-confidence signal it discards

`multiroom_pose_estimation.py:471` sorts candidates by `(-n_tight, avg_dist)` and takes [0], throwing away the runner-up. The **margin between the best and second-best DISTINCT rotation hypothesis** separates good from aliased poses at **42/44**, certifying 21/23 good poses with ZERO false positives (good margins span 3-56, aliased 0-11; several aliased poses have margin **0** — an exact tie broken arbitrarily).

It measures the right thing (was the rotation choice decisive?) and, unlike an absolute `n_tight` threshold, is computed WITHIN a pano so it does not inherit the geometry-scale calibration trap that `tau` did. Carrying `rot_idx` into the `candidates` dict (already on `top_poses[i]`) and emitting the margin into `result_entry` costs no extra compute. Threshold value (13) is tuned on this scene and needs checking elsewhere.
