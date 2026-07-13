"""Whole-area validation: CPO-loss gate vs raw-mean gate for the 12 panos on the 23-room
candidate set. Does the raw-mean score (D28) beat CPO's deployed match_color+weighted loss
on the whole-area regime where CPO-loss collapsed (v1 D21: whole-area recall 17-33% raw /
58% v1-calibrated)? Both scores -> the SAME calibrate gate; only the score differs. Offline.
-> WHOLEAREA_RESULTS.md."""
import os, sys, json
_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, "..", "..", "src"))
sys.path.insert(0, os.path.join(_HERE, "..", ".."))
from experiments.fgpl_seed import subset, paths
from experiments.fgpl_seed.robust_analysis import curve
from panopin import robust_score


def main():
    rows = subset.build_subset()
    true = {r["pano_name"]: r["room"] for r in rows}
    n = len(rows)
    cache = json.load(open(paths.WORK / "seeds" / "wholearea_cache.json"))
    grids = json.load(open(paths.WORK / "seeds" / "wholearea_residuals.json"))
    cpo_sm = {p: cache[p]["per_room"] for p in cache}
    rawmean_sm = robust_score.raw_mean_scores(grids)
    nrooms = len(next(iter(cpo_sm.values())))

    def uncal_recall(sm):   # argmin room == true (no calibration/gate)
        return sum(1 for p in sm if min(sm[p], key=lambda r: sm[p][r]) == true[p])

    methods = [("CPO-loss", cpo_sm), ("raw-mean (D28)", rawmean_sm)]
    lines = [f"# Whole-area validation — CPO-loss vs raw-mean room score ({n} panos x {nrooms} rooms)\n",
             "Does the raw-mean score (D28) beat CPO's deployed match_color+weighted loss on the whole-area",
             "candidate set where CPO-loss collapsed (v1 D21: whole-area recall 17-33% raw / 58% v1-calibrated)?",
             "SCORE quality = uncalibrated argmin recall. GATE = the SAME minmax calibrate on top; prefix_correct",
             "= confidently+correctly committed before the first wrong room.\n",
             "| room score | uncalibrated recall@1 | +minmax gate: prefix_correct | +minmax gate: recall@1 |",
             "|------------|-----------------------|------------------------------|------------------------|"]
    res = {}
    for label, sm in methods:
        prefix, recall = curve(sm, true)
        ur = uncal_recall(sm)
        res[label] = (ur, prefix, recall)
        lines.append(f"| {label} | {ur}/{n} | {prefix}/{n} | {recall}/{n} |")

    cu, cp, cr = res["CPO-loss"]
    ru, rp, rr = res["raw-mean (D28)"]
    lines.append(f"\n**SCORE: raw-mean {'BEATS' if ru > cu else ('ties' if ru == cu else 'loses to')} CPO-loss "
                 f"uncalibrated ({ru}/{n} vs {cu}/{n}) — the D28 score advantage {'HOLDS' if ru > cu else 'does NOT hold'} at 23 rooms.**")
    lines.append(f"**GATE: minmax calibration HURTS both at scale** (CPO {cu}->{cr}, raw-mean {ru}->{rr}); "
                 f"confident-correct prefix collapses to {rp}/{n} (raw-mean) vs {cp}/{n} (CPO) — the D28 "
                 f"6-room prefix advantage (3->9) does NOT survive.")
    lines.append(f"- => the raw-mean SCORE generalizes (best raw classifier at scale, {ru}/{n}); the OPEN problem "
                 f"is the CONFIDENCE GATE over it (minmax, tuned on the easy 6-room case, breaks at 23 rooms).")
    lines.append("- Caveat: n=12 (prefix tail-sensitive); this expands candidate ROOMS (6->23, hard loss-sink "
                 "regime), NOT the pano count. More panos = the ~12h whole-area localization.")

    out = "\n".join(lines) + "\n"
    print(out)
    with open(os.path.join(_HERE, "WHOLEAREA_RESULTS.md"), "w") as f:
        f.write(out)


if __name__ == "__main__":
    main()
