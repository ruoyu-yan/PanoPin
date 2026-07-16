# Strictly-Manhattan room assignment — 22 panos, 5 rooms

Both the candidate rooms and the panos are confined to Manhattan-world rooms (Point_360 `roadmap.md` §5), matching the Stage-3 3-orthogonal-direction assumption.
Pool: office_5, hallway_1, lounge_1, conferenceRoom_1, WC_1.
Excluded as non-Manhattan (real diagonal wall): office_3, office_7, office_8.
Offline rescoring of cached D33 grids (no GPU); GT used only to score.

## Per-pano recall@1 (low-pct argmin vs raw CPO min-loss)
| score | k=3 | k=4 | k=5 |
|---|---|---|---|
| low-pct (deployed) | 89% | 85% | 82% |
| CPO min-loss | 71% | 66% | 64% |

## Per-ROOM coverage (the deployment metric, D32)
| method | k=3 | k=4 | k=5 |
|---|---|---|---|
| room-anchored (correct seeds) | 100% | 100% | 100% |
| winner-gate (cov @ precision 1.0) | 100% | 100% | 100% |

## Demo configuration (all 5 rooms) — room-anchored seeds
Each room picks its own best-matching pano. **5/5 seeds correct.**

| room | seed pano | score | correct? |
|---|---|---|---|
| office_5 | `f758ab19` | 0.0263 | YES |
| hallway_1 | `fafa0629` | 0.0387 | YES |
| lounge_1 | `127fc8df` | 0.0305 | YES |
| conferenceRoom_1 | `87d7995a` | 0.0442 | YES |
| WC_1 | `44d6d985` | 0.0281 | YES |

## Per-pano confidence ranking (winner-score gate)
Per-pano argmin is correct for **18/22** panos. Sorted most-confident first; a gate admits from the top. Weak-lock panos (D23) should sink to the bottom.

| rank | pano | picked room | true room | score | correct? |
|---|---|---|---|---|---|
| 1 | `f758ab19` | office_5 | office_5 | 0.0263 | YES |
| 2 | `44d6d985` | WC_1 | WC_1 | 0.0281 | YES |
| 3 | `127fc8df` | lounge_1 | lounge_1 | 0.0305 | YES |
| 4 | `80e1f6ae` | lounge_1 | lounge_1 | 0.0332 | YES |
| 5 | `4a7bfe05` | lounge_1 | lounge_1 | 0.0346 | YES |
| 6 | `b50320e1` | lounge_1 | lounge_1 | 0.0382 | YES |
| 7 | `fafa0629` | hallway_1 | hallway_1 | 0.0387 | YES |
| 8 | `4d491624` | hallway_1 | hallway_1 | 0.0388 | YES |
| 9 | `ac7e25ee` | hallway_1 | hallway_1 | 0.0422 | YES |
| 10 | `b3ef2004` | WC_1 | WC_1 | 0.0429 | YES |
| 11 | `87d7995a` | conferenceRoom_1 | conferenceRoom_1 | 0.0442 | YES |
| 12 | `e2e170cb` | conferenceRoom_1 | conferenceRoom_1 | 0.0465 | YES |
| 13 | `481b93c5` | conferenceRoom_1 | conferenceRoom_1 | 0.0483 | YES |
| 14 | `c67b419a` | conferenceRoom_1 | conferenceRoom_1 | 0.0498 | YES |
| 15 | `47b49abf` | hallway_1 | hallway_1 | 0.0563 | YES |
| 16 | `f0e54fcd` | WC_1 | WC_1 | 0.0683 | YES |
| 17 | `d0834679` | hallway_1 | hallway_1 | 0.0696 | YES |
| 18 | `2b70fafb` | WC_1 | office_5 | 0.0836 | NO |
| 19 | `1a557181` | hallway_1 | WC_1 | 0.0849 | NO |
| 20 | `1119e668` | hallway_1 | hallway_1 | 0.0945 | YES |
| 21 | `5c2959c3` | hallway_1 | office_5 | 0.0970 | NO |
| 22 | `da0bb9ad` | lounge_1 | conferenceRoom_1 | 0.1466 | NO |

## Verdict
- Raw CPO min-loss picks the right room for 14/22 panos — the pool is not benign (hallway_1 is a real loss-sink).
- **Per-room coverage at the demo config (k=5): room-anchored 100%, winner-gate@precision-1 100%.**
- Per-pano recall (low-pct) 82% at k=5 — the D31/D23 color ceiling is present here too, and (per D32) does not bind per-room coverage.
