"""Does per-room minmax calibration (calibrate.minmax_scores, the designed loss-sink fix)
lift recall on top of the low-percentile score in the DEPLOYMENT regime (k covered rooms,
>=1 pano each)? Within each covered-room subset we restrict the score matrix to that
scenario's rooms+panos, calibrate (leave-one-out per room across the scenario's panos),
then argmin. Compares CPO / raw-mean / low-pct, each raw-argmin vs +minmax. Offline, no GPU.
GT used only to score (D5). -> appended to DEPLOY_REGIME_RESULTS.md by the caller."""
import os, sys, json, itertools
_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, "..", "..", "src"))
sys.path.insert(0, os.path.join(_HERE, "..", ".."))
from experiments.fgpl_seed import subset, paths
from experiments.fgpl_seed.deploy_regime import COVERED
from panopin import robust_score, calibrate


def recall(sm, true, k, gate):
    """Mean per-pano recall over size-k covered-room subsets. gate: 'argmin' or 'minmax'."""
    correct = total = 0
    for S in itertools.combinations(COVERED, k):
        Sset = set(S)
        panos = [p for p in sm if true[p] in Sset]
        if not panos:
            continue
        sub = {p: {r: sm[p][r] for r in S} for p in panos}      # scenario score matrix
        if gate == "minmax":
            sub = calibrate.minmax_scores(sub)
        for p in panos:
            total += 1
            if min(S, key=lambda r: sub[p][r]) == true[p]:
                correct += 1
    return correct / total if total else 0.0


def main():
    rows = subset.build_subset()
    true = {r["pano_name"]: r["room"] for r in rows}
    caches = {"whole-area": "wholearea_residuals.json", "6-room": "residuals.json"}
    cpo_cache = json.load(open(paths.WORK / "seeds" / "cpo_cache.json"))
    cpo_wa = json.load(open(paths.WORK / "seeds" / "wholearea_cache.json"))

    L = []
    def out(s=""): L.append(s); print(s)

    out("\n## Q4 — does minmax calibration lift the deployment regime? (low-pct q=20)\n")
    for cache_lbl, fname in caches.items():
        grids = json.load(open(paths.WORK / "seeds" / fname))
        cpo_sm = ({p: cpo_wa[p]["per_room"] for p in cpo_wa} if cache_lbl == "whole-area"
                  else {p: cpo_cache[p]["per_room"] for p in cpo_cache})
        scores = {
            "CPO-loss": cpo_sm,
            "raw-mean": robust_score.raw_mean_scores(grids),
            "low-pct q20": robust_score.robust_scores(grids, "low_percentile", q=20),
        }
        out(f"### {cache_lbl} cache")
        out("| score | gate | k=3 | k=4 | k=5 | k=6 |")
        out("|-------|------|-----|-----|-----|-----|")
        for lbl, sm in scores.items():
            for gate in ("argmin", "minmax"):
                cells = " | ".join(f"{recall(sm, true, k, gate)*100:.0f}%" for k in (3, 4, 5, 6))
                out(f"| {lbl} | {gate} | {cells} |")
        out("")

    with open(os.path.join(_HERE, "DEPLOY_REGIME_GATE.md"), "w") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
