# Loss-sink attack — per-pano-relative vs joint assignment (low-pct q20 base)

Recall over covered-room subsets, both caches. Method families: per-pano (minmax/rel_*) vs joint (rank). Baseline = argmin (plain low-pct).

## whole-area cache
| method | family | k=3 | k=4 | k=5 | k=6 |
|--------|--------|-----|-----|-----|-----|
| argmin | control | 92% | 92% | 92% | 92% |
| minmax | per-pano | 88% | 88% | 90% | 92% |
| rel_median | per-pano | 88% | 88% | 87% | 83% |
| rel_q25 | per-pano | 88% | 89% | 90% | 92% |
| rank | JOINT | 89% | 90% | 88% | 92% |

## 6-room cache
| method | family | k=3 | k=4 | k=5 | k=6 |
|--------|--------|-----|-----|-----|-----|
| argmin | control | 75% | 75% | 75% | 75% |
| minmax | per-pano | 76% | 75% | 75% | 75% |
| rel_median | per-pano | 72% | 72% | 73% | 75% |
| rel_q25 | per-pano | 74% | 75% | 75% | 75% |
| rank | JOINT | 76% | 75% | 75% | 75% |

## Diagnosis (6-room, k=6): full assignment per method (confirms trade vs net-fix)
| pano | true | argmin | minmax | rel_median | rel_q25 | rank |
|------|------|---|---|---|---|---|
| b47f836a | hallway_3 | OK | OK | OK | OK | OK |
| decc136d | hallway_3 | OK | OK | OK | OK | OK |
| 614b6342 | office_1 | OK | OK | OK | OK | OK |
| 7e48d75e | office_1 | OK | OK | OK | OK | OK |
| 870532d7 | office_4 | hallway_3 | office_7 | office_7 | office_7 | office_7 |
| 995ce724 | office_4 | OK | OK | OK | OK | OK |
| 2b70fafb | office_5 | OK | OK | OK | OK | OK |
| 5c2959c3 | office_5 | hallway_3 | hallway_3 | hallway_3 | hallway_3 | hallway_3 |
| 249f2e52 | office_6 | OK | OK | OK | OK | OK |
| d6f28d9f | office_6 | OK | OK | OK | OK | OK |
| 0e30c45e | office_7 | hallway_3 | office_6 | hallway_3 | hallway_3 | hallway_3 |
| 66dbab01 | office_7 | OK | OK | OK | OK | OK |

## Conclusion — loss-sink attack REFUTED (the sink is a symptom, not the cause)
- **Neither architecture net-improves recall:** per-pano-relative (minmax / rel_median /
  rel_q25) AND joint (rank) all land at plain low-pct argmin (75% 6-room / 92% whole-area).
- **Why:** removing hallway_3's pull moves the 3 impostors OFF the sink but into OTHER wrong
  rooms (870532d7 office_4->office_7; 0e30c45e office_7->office_6/hallway) — never their
  true room. The sink is a SYMPTOM of the miss, not its cause.
- **Root cause = weak ABSOLUTE color lock:** the 3 impostors match their own true room at
  low-pct 0.15-0.25 vs ~0.06-0.08 for clean panos (loss_sink_probe) — the D23 window/
  occlusion hard floor (~1/3 of panos can't self-localize). No score-matrix method recovers
  a room the color signal does not support.
- **Ceiling reached for color-only room assignment = low-pct argmin (D30).** Further recall
  needs a NON-color cue (geometry/rotation) for weak-lock panos, OR a CONFIDENCE GATE to
  abstain on them — which in the all-covered deployment (>=1 pano/room) still yields full
  per-ROOM coverage via each room's strong panos. Per-pano recall is not the deployment
  metric; per-room coverage is. (-> next: the gate/coverage reframe, not a sink method.)
