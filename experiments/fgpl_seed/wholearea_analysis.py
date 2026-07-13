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

    raw_hits = sum(1 for p in cache
                   if min(cache[p]["per_room"], key=lambda r: cache[p]["per_room"][r]) == true[p])

    methods = [("CPO-loss gate (deployed)", cpo_sm), ("raw-mean gate (D28)", rawmean_sm)]
    lines = [f"# Whole-area validation — CPO-loss gate vs raw-mean gate ({n} panos x {nrooms} rooms)\n",
             "Does the raw-mean score (D28) beat CPO's deployed match_color+weighted loss on the whole-area",
             "candidate set where CPO-loss collapsed (v1 D21: whole-area recall 17-33% raw / 58% v1-calibrated)?",
             "Both scores feed the SAME calibrate gate; only the score differs. prefix_correct = confidently+",
             "correctly committed before the first wrong room.\n",
             f"Raw min-loss (no calibration) recall@1 = **{raw_hits}/{n}**.\n",
             "| gate score | prefix_correct | recall@1 |",
             "|------------|----------------|----------|"]
    res = {}
    for label, sm in methods:
        prefix, recall = curve(sm, true)
        res[label] = (prefix, recall)
        lines.append(f"| {label} | {prefix}/{n} | {recall}/{n} |")

    cp, cr = res["CPO-loss gate (deployed)"]
    rp, rr = res["raw-mean gate (D28)"]
    verdict = "BEATS" if rp > cp else ("ties" if rp == cp else "LOSES to")
    lines.append(f"\n**CPO-loss gate: prefix {cp}/{n}, recall {cr}/{n}. raw-mean gate: prefix {rp}/{n}, "
                 f"recall {rr}/{n}.**")
    lines.append(f"- raw-mean **{verdict}** CPO-loss on prefix_correct ({rp} vs {cp}); recall {rr} vs {cr}.")
    lines.append("- Caveat: n=12 (prefix tail-sensitive); this expands the candidate ROOMS (6->23, the hard "
                 "loss-sink regime), NOT the pano count. More panos = the ~12h whole-area localization.")

    out = "\n".join(lines) + "\n"
    print(out)
    with open(os.path.join(_HERE, "WHOLEAREA_RESULTS.md"), "w") as f:
        f.write(out)


if __name__ == "__main__":
    main()
