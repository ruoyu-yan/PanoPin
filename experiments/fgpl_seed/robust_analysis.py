"""Precision/coverage compare: minmax gate on MEAN loss (a) vs on ROBUST residual
scores (c). Both use calibrate.minmax_scores; only the score matrix differs. Winner =
most panos confidently+correctly committed before the first wrong room. Offline, no GPU."""
import os, sys, json
_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, "..", "..", "src"))
sys.path.insert(0, os.path.join(_HERE, "..", ".."))
from experiments.fgpl_seed import subset, paths
from panopin import calibrate, robust_score

def curve(score_matrix, true_room):
    """Return (prefix_correct, recall_all): add panos in ascending confidence (most
    confident first); prefix_correct = leading all-correct run length; recall_all =
    total correct at full coverage."""
    ss = calibrate.minmax_scores(score_matrix)
    winner = {p: min(ss[p], key=lambda r: ss[p][r]) for p in ss}
    conf = {p: ss[p][winner[p]] for p in ss}
    order = sorted(conf, key=lambda p: conf[p])
    prefix, still = 0, True
    recall = 0
    for i, p in enumerate(order):
        ok = winner[p] == true_room[p]
        recall += ok
        if still and ok:
            prefix += 1
        elif still:
            still = False
    return prefix, recall

def main():
    rows = subset.build_subset()
    true_room = {r["pano_name"]: r["room"] for r in rows}
    n = len(rows)
    cache = json.load(open(paths.WORK / "seeds" / "cpo_cache.json"))
    grids = json.load(open(paths.WORK / "seeds" / "residuals.json"))

    methods = [("a: mean-loss", {p: cache[p]["per_room"] for p in cache})]
    for stat, params, label in [
        # a' = mean of the SAME raw residuals residuals_at_pose gives (trimmed_mean k=0). This
        # is the clean baseline that isolates mean->robust: (a) is match_color+weighted (deployed),
        # (a') is raw+unweighted like the robust variants, so c-vs-a' attributes any gain to
        # ROBUST aggregation alone (not color-space/weighting differences).
        ("trimmed_mean", {"k": 0}, "a': raw-mean"),
        ("median", {}, "c: median"),
        ("low_percentile", {"q": 10}, "c: low-pct@10"),
        ("low_percentile", {"q": 20}, "c: low-pct@20"),
        ("low_percentile", {"q": 25}, "c: low-pct@25"),
        ("trimmed_mean", {"k": 10}, "c: trim-top@10"),
        ("trimmed_mean", {"k": 20}, "c: trim-top@20"),
        ("trimmed_mean", {"k": 30}, "c: trim-top@30"),
    ]:
        methods.append((label, robust_score.robust_scores(grids, stat, **params)))

    lines = ["# Robust-gate comparison (precision/coverage, n=%d)\n" % n,
             "Metric: **prefix_correct** = panos confidently+correctly committed before the FIRST wrong",
             "room (= coverage-at-100%%-precision x n). recall@1 = correct at full coverage.\n",
             "| method | prefix_correct (cov@100%%prec) | recall@1 (all) |",
             "|--------|------------------------------|----------------|"]
    results = []
    for label, sm in methods:
        prefix, recall = curve(sm, true_room)
        results.append((label, prefix, recall))
        lines.append(f"| {label} | {prefix}/{n} ({100*prefix/n:.0f}%) | {recall}/{n} ({100*recall/n:.0f}%) |")
    by_label = {r[0]: r[1] for r in results}
    a, a_raw = by_label["a: mean-loss"], by_label["a': raw-mean"]
    robust = {r[0]: r[1] for r in results if r[0].startswith("c:")}
    best_c_label = max(robust, key=lambda k: robust[k]); best_c = robust[best_c_label]
    lines.append(f"\n**Deployed mean gate (a) = {a}/{n}; raw-mean (a') = {a_raw}/{n}; "
                 f"best robust (c) = {best_c_label} at {best_c}/{n}.**")
    lines.append(f"- Robustness isolated (same raw residuals): c {'BEATS' if best_c > a_raw else 'does NOT beat'} a' ({best_c} vs {a_raw}).")
    lines.append(f"- Deployment: c {'BEATS' if best_c > a else 'does NOT beat'} the deployed match_color+weighted mean gate a ({best_c} vs {a}).")
    lines.append("(n=12 is noisy — D26 caveat; treat small differences as within noise.)")

    out = "\n".join(lines) + "\n"
    print(out)
    with open(os.path.join(_HERE, "ROBUST_RESULTS.md"), "w") as f:
        f.write(out)

if __name__ == "__main__":
    main()
