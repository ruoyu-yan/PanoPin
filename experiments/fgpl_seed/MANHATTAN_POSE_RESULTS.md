# Strictly-Manhattan PanoPin -> FGPL pose round-trip

**Scene `area3_manhattan`: 5 Manhattan rooms, 22 panos.** Both the PanoPin candidate rooms AND the FGPL line map are built from Manhattan rooms only, so the Stage-3 3-orthogonal-direction assumption holds throughout.
Rooms: WC_1, conferenceRoom_1, hallway_1, lounge_1, office_5. Excluded as non-Manhattan: office_3, office_7, office_8.

The line map recovered three exactly axis-aligned principal directions with **0% unclassified** sparse lines — the Manhattan restriction is doing what it should.

PanoPin seeds every pano at its low-percentile argmin room (no gate): the question is 'estimated pose per pano vs GT'. The oracle arm seeds each pano at its GT position in its true room, isolating FGPL's own refinement error on this same map.

## Accuracy vs S3DIS ground truth
| arm | localized | per-room coverage | trans median | trans mean | trans max | rot median | wrong-room |
|---|---|---|---|---|---|---|---|
| PanoPin seed (color) | 22/22 | 5/5 | 0.960 | 3.420 | 26.032 | 89.9 | 0.18 |
| GT seed (oracle ref) | 22/22 | 5/5 | 0.113 | 0.584 | 2.047 | 1.3 | 0.09 |

Right-room slice (panos PanoPin seeded into their true room): median **0.464 m** over 18 panos.

## Per-pano translation error (PanoPin arm)
| pano | true room | seed room | seed correct? | trans err (m) |
|---|---|---|---|---|
| `c67b419a` | conferenceRoom_1 | conferenceRoom_1 | YES | 0.017 |
| `44d6d985` | WC_1 | WC_1 | YES | 0.018 |
| `87d7995a` | conferenceRoom_1 | conferenceRoom_1 | YES | 0.025 |
| `5c2959c3` | office_5 | hallway_1 | NO | 0.029 |
| `47b49abf` | hallway_1 | hallway_1 | YES | 0.038 |
| `fafa0629` | hallway_1 | hallway_1 | YES | 0.041 |
| `ac7e25ee` | hallway_1 | hallway_1 | YES | 0.042 |
| `1119e668` | hallway_1 | hallway_1 | YES | 0.043 |
| `b3ef2004` | WC_1 | WC_1 | YES | 0.067 |
| `f758ab19` | office_5 | office_5 | YES | 0.068 |
| `b50320e1` | lounge_1 | lounge_1 | YES | 0.859 |
| `f0e54fcd` | WC_1 | WC_1 | YES | 1.061 |
| `d0834679` | hallway_1 | hallway_1 | YES | 1.341 |
| `4d491624` | hallway_1 | hallway_1 | YES | 1.386 |
| `e2e170cb` | conferenceRoom_1 | conferenceRoom_1 | YES | 1.420 |
| `481b93c5` | conferenceRoom_1 | conferenceRoom_1 | YES | 1.735 |
| `127fc8df` | lounge_1 | lounge_1 | YES | 1.794 |
| `4a7bfe05` | lounge_1 | lounge_1 | YES | 1.895 |
| `80e1f6ae` | lounge_1 | lounge_1 | YES | 2.047 |
| `da0bb9ad` | conferenceRoom_1 | lounge_1 | NO | 15.343 |
| `1a557181` | WC_1 | hallway_1 | NO | 19.937 |
| `2b70fafb` | office_5 | WC_1 | NO | 26.032 |

Seeded into the right room: 18/22 panos, median 0.464 m. Wrong-room seeds: 4, median 17.640 m.

**Gate caveat:** the D34 `tau=0.10` gate would admit 21/22 panos here, including 3 wrong-room ones — on this pool the score distribution sits lower than the pool tau was tuned on, so the fixed threshold does not transfer. The threshold-free room-anchored seeding (5/5 correct, MANHATTAN_RESULTS.md) is the claim that holds.

## Rotation lock drives translation accuracy
FGPL's rotation is Manhattan-aliased: it either locks the true rotation (a few degrees) or snaps to a ~90/120/180 deg alias. That lock — not a gradual seed error — is what decides the final position.

| arm | rotation locked (<=45 deg) | trans median (locked) | trans median (aliased) |
|---|---|---|---|
| PanoPin seed (color) | 10/22 | 0.040 | 1.765 |
| GT seed (oracle ref) | 13/22 | 0.048 | 1.341 |

When FGPL locks rotation, translation is centimetre-accurate; when it aliases, position lands 1-2 m off. Aliasing is an FGPL property, not a PanoPin regression: it affects the GT-seeded oracle too. But seed quality shifts how OFTEN it happens, and the effect is not monotonic in seed error — on some panos the PanoPin seed locks where the GT seed aliases (e.g. `47b49abf`, `1119e668`) and vice versa.

**This corrects D25/D35**, which concluded rotation was hopeless because the oracle was itself ~90 deg off. That was measured on a map containing 2 non-Manhattan rooms. On a strictly Manhattan map the GT-seeded oracle reaches a 1.3 deg median — FGPL's rotation is usable when the Manhattan assumption actually holds.

### Read the medians with care
The overall medians (0.960 m PanoPin vs 0.113 m oracle) look like a 9x gap but are an ARTEFACT of a bimodal distribution straddled by a ~50% lock rate: the median simply falls on the locked side for the oracle (13/22) and the aliased side for PanoPin (10/22). The honest comparison is within-group — and there, **the PanoPin seed matches the GT seed** (locked: 0.040 vs 0.048 m). The whole difference is a 3-pano lock-rate gap at n=22, which is not clearly distinguishable from noise.

## Verdict

- **PanoPin delivers every room a correct seed** (room coverage 5/5, room-anchored 5/5) and FGPL localized 22/22 panos, zero errors.
- **Where rotation locks, the color seed is as good as ground truth** (0.040 vs 0.048 m) — D25's 'color position seed ~= oracle' reproduces under a strict Manhattan restriction, now measured through FGPL.
- **The binding constraint is FGPL's rotation aliasing, not PanoPin.** It hits the GT-seeded oracle nearly as often (9/22 vs 12/22), and it is chaotic rather than monotonic in seed error — a better seed does not reliably prevent it.
- **Manhattan restriction materially helped FGPL:** the oracle reaches 0.113 m / 1.3 deg here, vs 0.696 m / 89.9 deg for D35's oracle on a map with 2 non-Manhattan rooms (different pool, so indicative rather than controlled).
- Remaining PanoPin-side gap = 4 wrong-room seeds out of 22 (the D23/D31 weak-lock ceiling), which cost the mean/max but not per-room coverage.
