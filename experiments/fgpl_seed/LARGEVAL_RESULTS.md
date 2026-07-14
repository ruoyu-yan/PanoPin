# Larger-n validation (D33) — 32 panos, 8 pool rooms

Pool (diverse, all in-frame panos): office_3, office_5, office_7, office_8, hallway_1, lounge_1, conferenceRoom_1, WC_1. Monte-Carlo over covered-room
subsets. Tests whether D30 (low-pct score) + D32 (per-room coverage) hold beyond n=12.

## Per-pano recall@1 (low-pct argmin vs CPO min-loss)
| score | k=3 | k=4 | k=5 | k=6 | k=7 | k=8 |
|---|---|---|---|---|---|---|
| low-pct | 85% | 82% | 80% | 79% | 79% | 78% |
| CPO-loss | 74% | 68% | 64% | 62% | 60% | 59% |

## Per-ROOM coverage (the deployment metric, D32)
| method | k=3 | k=4 | k=5 | k=6 | k=7 | k=8 |
|---|---|---|---|---|---|---|
| room-anchored (correct seeds) | 100% | 100% | 100% | 100% | 100% | 100% |
| winner-gate (cov @ precision 1.0) | 100% | 100% | 100% | 100% | 100% | 100% |

## Conclusion — D30 + D32 CONFIRMED at n=32 on diverse rooms
- **Loss-sink present:** raw CPO min-loss recall = 19/32 (hallway_1 is the new
  loss-sink, capturing ~3 impostors) — a real stress test, not a benign pool.
- **Per-pano (D30):** low-pct 78-85% beats CPO-loss 59-74% at EVERY k (by ~11-19 pts) — the low-pct advantage HOLDS.
- **Per-ROOM coverage (D32):** room-anchored 100% and winner-gate@precision1 100% at ALL k in [3, 4, 5, 6, 7, 8] — the deployment claim reproduces beyond n=12, WITH a loss-sink.
- **=> the deployable hand-off `panopin.seed.seed_rooms` is validated** (room-anchored low-pct
  seeding gives every room a correct seed for FGPL). Per-pano recall ceiling (D31) is real but
  does NOT bind the per-room deployment metric.
