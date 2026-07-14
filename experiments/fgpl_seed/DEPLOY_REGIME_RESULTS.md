# Deployment-regime characterization — recall vs candidate-set size (12 panos)

Real deployment = a k-room cloud, EVERY room covered by a pano. Rescores the cached
whole-area residual grids restricted to subsets of the 6 covered rooms. Offline, no GPU.
Covered rooms (6): office_1, office_4, office_5, office_6, office_7, hallway_3. n=12 panos (2/room).

## Fidelity check (k=23, full candidate list — must match WHOLEAREA_RESULTS.md)
| score | whole-area recall@1 |
|-------|---------------------|
| CPO-loss | 6/12 |
| raw-mean | 8/12 |
(expect CPO-loss 6/12, raw-mean 8/12)

## Q1 — recall vs candidate-set size k (all rooms covered)
| score | k=2 | k=3 | k=4 | k=5 | k=6 | k=23 |
|---|---|---|---|---|---|---|
| CPO-loss | 92% | 88% | 87% | 85% | 83% | 50% |
| raw-mean | 90% | 88% | 87% | 85% | 83% | 67% |

## Q2 — best per-room score statistic, by k (covered-room subsets)
| score statistic | k=3 | k=4 | k=5 | k=6 | k=23 |
|---|---|---|---|---|---|
| CPO-loss | 88% | 87% | 85% | 83% | 50% |
| raw-mean | 88% | 87% | 85% | 83% | 67% |
| median | 92% | 92% | 92% | 92% | 75% |
| trimmed_mean_k10 | 92% | 92% | 92% | 92% | 67% |
| trimmed_mean_k20 | 92% | 92% | 92% | 92% | 67% |
| trimmed_mean_k30 | 92% | 92% | 92% | 92% | 75% |
| low_pct_q10 | 92% | 92% | 92% | 92% | 75% |
| low_pct_q25 | 92% | 92% | 92% | 92% | 83% |
| low_pct_q40 | 92% | 92% | 92% | 92% | 83% |

- best @ k=3: **median** (92%)
- best @ k=4: **median** (92%)
- best @ k=5: **median** (92%)
- best @ k=6: **median** (92%)
- best @ k=23: **low_pct_q25** (83%)

## Q3 — failures at k=6 (full covered set), raw-mean score
| pano | true room | picked | pick<true margin | verdict |
|------|-----------|--------|------------------|---------|
| b47f836a | hallway_3 | hallway_3 | +0.0000 | OK |
| decc136d | hallway_3 | hallway_3 | +0.0000 | OK |
| 614b6342 | office_1 | office_1 | +0.0000 | OK |
| 7e48d75e | office_1 | office_1 | +0.0000 | OK |
| 870532d7 | office_4 | office_4 | +0.0000 | OK |
| 995ce724 | office_4 | office_4 | +0.0000 | OK |
| 2b70fafb | office_5 | office_5 | +0.0000 | OK |
| 5c2959c3 | office_5 | hallway_3 | +0.0059 | MISS |
| 249f2e52 | office_6 | office_6 | +0.0000 | OK |
| d6f28d9f | office_6 | office_6 | +0.0000 | OK |
| 0e30c45e | office_7 | hallway_3 | +0.0938 | MISS |
| 66dbab01 | office_7 | office_7 | +0.0000 | OK |

loss-sink competitors (room: #panos it wrongly captured): hallway_3=2

