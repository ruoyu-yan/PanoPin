"""D28: attribute the a(deployed, 3/12) -> a'(raw-mean, 9/12) prefix_correct gap (D27) to
match_color / weighting / subsample. Every a'-style score is the mean of a residual grid
(trimmed_mean k=0), differing only in match_color/subsample; weighting is inferred by
elimination (its score = the deployed cache `a`, which is match_color+weighted+CPOsub).
Reuses robust_analysis.curve(). Offline, no GPU. Writes DRIVER_ISOLATION.md."""
import os, sys, json
_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, "..", "..", "src"))
sys.path.insert(0, os.path.join(_HERE, "..", ".."))
from experiments.fgpl_seed import subset, paths
from experiments.fgpl_seed.robust_analysis import curve
from panopin import robust_score


def _mean_grid(grids):
    """a'-style score = mean of the residual grid (trimmed_mean k=0)."""
    return robust_score.robust_scores(grids, "trimmed_mean", k=0)


def main():
    rows = subset.build_subset()
    true_room = {r["pano_name"]: r["room"] for r in rows}
    n = len(rows)
    cache = json.load(open(paths.WORK / "seeds" / "cpo_cache.json"))
    load = lambda name: json.load(open(paths.WORK / "seeds" / name))
    raw = load("residuals.json")       # a'  : raw + unweighted + seed0
    mc = load("residuals_mc.json")     # a'_mc: match_color + unweighted + seed0
    s1 = load("residuals_seed1.json")  # a'_seed1: raw + unweighted + seed1
    s2 = load("residuals_seed2.json")  # a'_seed2: raw + unweighted + seed2

    methods = [
        ("a  (matchcolor+weighted+CPOsub) [cache]", {p: cache[p]["per_room"] for p in cache}),
        ("a' (raw+unweighted+seed0)", _mean_grid(raw)),
        ("a'_mc (matchcolor+unweighted+seed0)", _mean_grid(mc)),
        ("a'_seed1 (raw+unweighted+seed1)", _mean_grid(s1)),
        ("a'_seed2 (raw+unweighted+seed2)", _mean_grid(s2)),
    ]
    res = {}
    lines = [f"# Driver isolation — why does a′(raw-mean) beat a(deployed) on the gate? (n={n})\n",
             "prefix_correct = panos confidently+correctly committed before the first wrong room.",
             "All a′-variants are unweighted grid means; only match_color/subsample differ. `a` is the",
             "deployed match_color+weighted+CPO-subsample cache loss.\n",
             "| variant | prefix_correct | recall@1 |",
             "|---------|----------------|----------|"]
    for label, sm in methods:
        prefix, recall = curve(sm, true_room)
        res[label] = prefix
        lines.append(f"| {label} | {prefix}/{n} | {recall}/{n} |")

    a = res["a  (matchcolor+weighted+CPOsub) [cache]"]
    ap = res["a' (raw+unweighted+seed0)"]
    amc = res["a'_mc (matchcolor+unweighted+seed0)"]
    as1 = res["a'_seed1 (raw+unweighted+seed1)"]
    as2 = res["a'_seed2 (raw+unweighted+seed2)"]

    def verdict(delta, big=3):
        return "MAJOR driver" if delta >= big else ("minor factor" if delta >= 1 else "NOT a driver")

    lines.append(f"\n**Gap to attribute: a={a}/{n} → a′={ap}/{n} (Δ={ap - a}).**")
    lines.append(f"- **match_color**: a′_mc={amc}/{n}. Turning match_color ON drops a′ by {ap - amc} "
                 f"→ match_color is a **{verdict(ap - amc)}** of the gap.")
    lines.append(f"- **subsample**: a′_seed1={as1}, a′_seed2={as2} vs a′={ap} "
                 f"(max Δ={max(abs(as1 - ap), abs(as2 - ap))}) → subsample is a "
                 f"**{verdict(max(abs(as1 - ap), abs(as2 - ap)))}**.")
    residual_gap = (a) - (amc)  # how far match_color-unweighted still sits above the deployed weighted a
    lines.append(f"- **weighting** (by elimination): a′_mc={amc}/{n} vs deployed a={a}/{n}. The part of the "
                 f"a→a′ gap NOT explained by match_color/subsample is weighting; a′_mc still "
                 f"{'>>' if amc - a >= 3 else ('>' if amc > a else '≈')} a by {amc - a} "
                 f"→ weighting is a **{verdict(amc - a)}** of the gap.")

    out = "\n".join(lines) + "\n"
    print(out)
    with open(os.path.join(_HERE, "DRIVER_ISOLATION.md"), "w") as f:
        f.write(out)


if __name__ == "__main__":
    main()
