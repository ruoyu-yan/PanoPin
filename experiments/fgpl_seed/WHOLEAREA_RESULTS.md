# Whole-area validation — CPO-loss vs raw-mean room score (12 panos x 23 rooms)

Does the raw-mean score (D28) beat CPO's deployed match_color+weighted loss on the whole-area
candidate set where CPO-loss collapsed (v1 D21: whole-area recall 17-33% raw / 58% v1-calibrated)?
SCORE quality = uncalibrated argmin recall. GATE = the SAME minmax calibrate on top; prefix_correct
= confidently+correctly committed before the first wrong room.

| room score | uncalibrated recall@1 | +minmax gate: prefix_correct | +minmax gate: recall@1 |
|------------|-----------------------|------------------------------|------------------------|
| CPO-loss | 6/12 | 2/12 | 5/12 |
| raw-mean (D28) | 8/12 | 2/12 | 6/12 |

**SCORE: raw-mean BEATS CPO-loss uncalibrated (8/12 vs 6/12) — the D28 score advantage HOLDS at 23 rooms.**
**GATE: minmax calibration HURTS both at scale** (CPO 6->5, raw-mean 8->6); confident-correct prefix collapses to 2/12 (raw-mean) vs 2/12 (CPO) — the D28 6-room prefix advantage (3->9) does NOT survive.
- => the raw-mean SCORE generalizes (best raw classifier at scale, 8/12); the OPEN problem is the CONFIDENCE GATE over it (minmax, tuned on the easy 6-room case, breaks at 23 rooms).
- Caveat: n=12 (prefix tail-sensitive); this expands candidate ROOMS (6->23, hard loss-sink regime), NOT the pano count. More panos = the ~12h whole-area localization.
