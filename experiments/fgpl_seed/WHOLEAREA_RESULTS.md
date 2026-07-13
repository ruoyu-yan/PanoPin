# Whole-area validation — CPO-loss gate vs raw-mean gate (12 panos x 23 rooms)

Does the raw-mean score (D28) beat CPO's deployed match_color+weighted loss on the whole-area
candidate set where CPO-loss collapsed (v1 D21: whole-area recall 17-33% raw / 58% v1-calibrated)?
Both scores feed the SAME calibrate gate; only the score differs. prefix_correct = confidently+
correctly committed before the first wrong room.

Raw min-loss (no calibration) recall@1 = **6/12**.

| gate score | prefix_correct | recall@1 |
|------------|----------------|----------|
| CPO-loss gate (deployed) | 2/12 | 5/12 |
| raw-mean gate (D28) | 2/12 | 6/12 |

**CPO-loss gate: prefix 2/12, recall 5/12. raw-mean gate: prefix 2/12, recall 6/12.**
- raw-mean **ties** CPO-loss on prefix_correct (2 vs 2); recall 6 vs 5.
- Caveat: n=12 (prefix tail-sensitive); this expands the candidate ROOMS (6->23, the hard loss-sink regime), NOT the pano count. More panos = the ~12h whole-area localization.
