"""Cross-validation of the deploy_regime finding on the INDEPENDENT 6-room residual cache
(residuals.json — poses refined against the 6-room candidate clouds, D28), vs the whole-area
cache used by deploy_regime.py. If robust low-tail stats still beat raw-mean for recall here,
the finding is not a whole-area-cache artifact. Offline, no GPU. GT used only to score (D5)."""
import os, sys, json
_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, "..", "..", "src"))
sys.path.insert(0, os.path.join(_HERE, "..", ".."))
from experiments.fgpl_seed import subset, paths
from experiments.fgpl_seed.deploy_regime import recall_over_subsets, whole_recall, COVERED
from panopin import robust_score

STATS = [
    ("CPO-loss",         None),
    ("raw-mean",         ("trimmed_mean", {"k": 0})),
    ("median",           ("median", {})),
    ("trimmed_mean_k20", ("trimmed_mean", {"k": 20})),
    ("trimmed_mean_k30", ("trimmed_mean", {"k": 30})),
    ("low_pct_q10",      ("low_percentile", {"q": 10})),
    ("low_pct_q25",      ("low_percentile", {"q": 25})),
    ("low_pct_q40",      ("low_percentile", {"q": 40})),
]


def main():
    rows = subset.build_subset()
    true = {r["pano_name"]: r["room"] for r in rows}
    n = len(true)
    grids = json.load(open(paths.WORK / "seeds" / "residuals.json"))
    cpo_cache = json.load(open(paths.WORK / "seeds" / "cpo_cache.json"))
    cpo_sm = {p: cpo_cache[p]["per_room"] for p in cpo_cache}

    print(f"# Cross-check on residuals.json (6-room cache, {n} panos, 6 covered rooms)\n")
    ks = [3, 4, 5, 6]
    print("| score statistic | " + " | ".join(f"k={k}" for k in ks) + " |")
    print("|" + "---|" * (len(ks) + 1))
    for lbl, spec in STATS:
        sm = cpo_sm if spec is None else robust_score.robust_scores(grids, spec[0], **spec[1])
        cells = []
        for k in ks:
            rec, _ = recall_over_subsets(sm, true, COVERED, k)
            cells.append(f"{rec*100:.0f}%")
        print(f"| {lbl} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
