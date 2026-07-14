# Gate/coverage reframe — can a confidence gate cover every room at high precision?

Deployment metric = per-ROOM coverage (>=1 correctly-assigned confident pano/room), not
per-pano recall. low-pct q20 score; confidence = margin (gap to runner-up) or winner (-score).

## whole-area cache (k=6 full covered set, 12 panos)

**signal=margin:** cover **6/6** rooms at precision=1.0; 6/6 rooms if all panos admitted (ungated).
| admitted | precision | rooms_covered | pano | pick | ok |
|----------|-----------|---------------|------|------|----|
| 1 | 1.00 | 1/6 | 66dbab01 | office_7 | OK |
| 2 | 1.00 | 2/6 | 995ce724 | office_4 | OK |
| 3 | 1.00 | 3/6 | 7e48d75e | office_1 | OK |
| 4 | 1.00 | 3/6 | 870532d7 | office_4 | OK |
| 5 | 1.00 | 4/6 | 249f2e52 | office_6 | OK |
| 6 | 1.00 | 5/6 | 2b70fafb | office_5 | OK |
| 7 | 1.00 | 5/6 | 614b6342 | office_1 | OK |
| 8 | 1.00 | 6/6 | decc136d | hallway_3 | OK |
| 9 | 1.00 | 6/6 | d6f28d9f | office_6 | OK |
| 10 | 1.00 | 6/6 | b47f836a | hallway_3 | OK |
| 11 | 1.00 | 6/6 | 5c2959c3 | office_5 | OK |
| 12 | 0.92 | 6/6 | 0e30c45e | office_5 | X |

**signal=winner:** cover **6/6** rooms at precision=1.0; 6/6 rooms if all panos admitted (ungated).
| admitted | precision | rooms_covered | pano | pick | ok |
|----------|-----------|---------------|------|------|----|
| 1 | 1.00 | 1/6 | 249f2e52 | office_6 | OK |
| 2 | 1.00 | 2/6 | 7e48d75e | office_1 | OK |
| 3 | 1.00 | 3/6 | 995ce724 | office_4 | OK |
| 4 | 1.00 | 3/6 | 614b6342 | office_1 | OK |
| 5 | 1.00 | 4/6 | 66dbab01 | office_7 | OK |
| 6 | 1.00 | 5/6 | 2b70fafb | office_5 | OK |
| 7 | 1.00 | 5/6 | d6f28d9f | office_6 | OK |
| 8 | 1.00 | 5/6 | 870532d7 | office_4 | OK |
| 9 | 1.00 | 6/6 | decc136d | hallway_3 | OK |
| 10 | 1.00 | 6/6 | b47f836a | hallway_3 | OK |
| 11 | 1.00 | 6/6 | 5c2959c3 | office_5 | OK |
| 12 | 0.92 | 6/6 | 0e30c45e | office_5 | X |

## 6-room cache (k=6 full covered set, 12 panos)

**signal=margin:** cover **5/6** rooms at precision=1.0; 6/6 rooms if all panos admitted (ungated).
| admitted | precision | rooms_covered | pano | pick | ok |
|----------|-----------|---------------|------|------|----|
| 1 | 1.00 | 1/6 | 995ce724 | office_4 | OK |
| 2 | 1.00 | 2/6 | b47f836a | hallway_3 | OK |
| 3 | 1.00 | 3/6 | 66dbab01 | office_7 | OK |
| 4 | 1.00 | 4/6 | d6f28d9f | office_6 | OK |
| 5 | 1.00 | 5/6 | 7e48d75e | office_1 | OK |
| 6 | 0.83 | 5/6 | 5c2959c3 | hallway_3 | X |
| 7 | 0.86 | 5/6 | 249f2e52 | office_6 | OK |
| 8 | 0.88 | 5/6 | 614b6342 | office_1 | OK |
| 9 | 0.89 | 5/6 | decc136d | hallway_3 | OK |
| 10 | 0.80 | 5/6 | 870532d7 | hallway_3 | X |
| 11 | 0.73 | 5/6 | 0e30c45e | hallway_3 | X |
| 12 | 0.75 | 6/6 | 2b70fafb | office_5 | OK |

**signal=winner:** cover **6/6** rooms at precision=1.0; 6/6 rooms if all panos admitted (ungated).
| admitted | precision | rooms_covered | pano | pick | ok |
|----------|-----------|---------------|------|------|----|
| 1 | 1.00 | 1/6 | 249f2e52 | office_6 | OK |
| 2 | 1.00 | 2/6 | 7e48d75e | office_1 | OK |
| 3 | 1.00 | 3/6 | 995ce724 | office_4 | OK |
| 4 | 1.00 | 3/6 | 614b6342 | office_1 | OK |
| 5 | 1.00 | 4/6 | 66dbab01 | office_7 | OK |
| 6 | 1.00 | 4/6 | d6f28d9f | office_6 | OK |
| 7 | 1.00 | 5/6 | b47f836a | hallway_3 | OK |
| 8 | 1.00 | 6/6 | 2b70fafb | office_5 | OK |
| 9 | 1.00 | 6/6 | decc136d | hallway_3 | OK |
| 10 | 0.90 | 6/6 | 5c2959c3 | hallway_3 | X |
| 11 | 0.82 | 6/6 | 0e30c45e | hallway_3 | X |
| 12 | 0.75 | 6/6 | 870532d7 | hallway_3 | X |

## Aggregate — mean room-coverage at precision=1.0 over covered-room subsets
| cache | signal | k=3 | k=4 | k=5 | k=6 |
|-------|--------|-----|-----|-----|-----|
| whole-area | margin | 100% | 100% | 100% | 100% |
| whole-area | winner | 100% | 100% | 100% | 100% |
| 6-room | margin | 87% | 90% | 87% | 83% |
| 6-room | winner | 100% | 100% | 100% | 100% |

## Room-anchored assignment — each room r seeded by argmin_p score[p][r]
| cache | k=3 | k=4 | k=5 | k=6 |  (mean fraction of rooms whose best-pano is a TRUE pano)
|-------|-----|-----|-----|-----|
| whole-area | 100% | 100% | 100% | 100% |
| 6-room | 100% | 100% | 100% | 100% |

## Conclusion — the deployment metric (per-room coverage) is SOLVED
- **Per-pano recall 75-92% becomes per-ROOM coverage 100% at precision 1.0** on both caches,
  every k in {3,4,5,6}. The weak-lock panos (D23/D31) carry the HIGHEST absolute low-pct
  score, so they rank LAST — a confidence gate on the winner-score admits all correct panos
  first and hands FGPL ZERO wrong seeds while covering every room via its strong panos.
- **winner-score >> margin** as the confidence signal (6-room: 100% vs 83-90%) — absolute
  low-pct value directly measures lock quality (genuine ~0.06-0.08, weak ~0.12+).
- **Room-anchored assignment (argmin_p score[p][r]) = 100% correct seeds, threshold-free,**
  loss-sink-immune by construction: each room's genuine panos match it best, so it self-seeds.
- **Shipped:** `panopin.coverage` (`room_anchored_seeds`, `pano_confidence`). Caveat: n=12;
  larger-pano GPU run should confirm. This is the deployable PanoPin->FGPL hand-off.
