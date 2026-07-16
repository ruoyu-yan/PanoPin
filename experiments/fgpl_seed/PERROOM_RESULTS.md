# Per-room architecture — REFUTED (2026-07-16)

**Hypothesis.** The 22-seed arm's rotation flips come from the Voronoi starving each pano of 3D
geometry (a room's lines split among the panos sharing it). Give every pano its WHOLE room —
per-room line map + global mode, no Voronoi — and all 22 should reach the 5-seed arm's quality
(0.045 m, zero flips) while still posing every pano.

**Result: worse, not better.** On the 18 correctly-assigned panos (the fair subset; the other 4
were deliberately sent to the wrong room's map by PanoPin's own assignment):

| | 22-seed + upright prior | per-room global |
|---|---|---|
| rotation locked | **14/18** | 12/18 |
| within 10 cm | **10/18** | 9/18 |
| trans median | **~0.08 m** | 0.112 m |

It fixed 3 panos (`481b93c5` 178.8->0.7 deg, `80e1f6ae` 92.1->0.4, `127fc8df` 89.4->1.3 — exactly
the lounge_1 cases predicted) but regressed ~6, several badly:

| pano | 22-seed | per-room |
|---|---|---|
| `fafa0629` | 0.040 m / 0.4 deg | **8.832 m** / 0.7 deg |
| `d0834679` | 0.032 m / 1.6 deg | 6.378 m / 179.4 deg |
| `87d7995a` | 0.025 m / 0.6 deg | 3.566 m / 179.3 deg |

**Why the hypothesis was wrong.** `fafa0629` is the tell: rotation locked at 0.7 deg, position 8.8 m
down the hallway. Correct orientation, wrong place. The seed's Voronoi cell does **two** jobs —
it starves the rotation search of geometry (the failure we chased) but it also **pins the
position** (the job we ignored). Removing it returns the geometry and discards the constraint.
A repetitive corridor has no positional signal without it.

**Implication.** There is no single lever. The 5-seed anchored arm wins because it happens to land
on both — roughly a room of geometry AND one seed per room, so position stays pinned. Per-room
global mode is the extreme of "large cell" and falls off the other side. Room-scoped filtering
(filter lines by room, keep the per-pano seed for position) is the untested middle: it would keep
BOTH properties. That is the one worth trying next, and it does require the FGPL change
(`get_local_mask` by `room_label` instead of Voronoi) that this experiment was designed to avoid.

**Best config for posing all 22 panos remains the 22-seed arm + upright prior** (0.084 m median,
15/22 locked). The 5-seed anchored arm (0.045 m, 0 flips) still poses only 5.

Code: `experiments/fgpl_seed/manhattan_perroom.py`. PanoPin supplied only the room assignment
(18/22 correct); no seed positions (global mode never reads the alignment); no GT.
