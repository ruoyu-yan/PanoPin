"""Pin the low-percentile q: sweep q on BOTH residual caches (whole-area + 6-room), and
report which panos low-pct rescues vs raw-mean (mechanism check). Offline, no GPU.
Chooses the q that maximises the worst-case (min over both caches) recall at deployment
sizes k=4 and k=6. GT used only to score (D5)."""
import os, sys, json
_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, "..", "..", "src"))
sys.path.insert(0, os.path.join(_HERE, "..", ".."))
from experiments.fgpl_seed import subset, paths
from experiments.fgpl_seed.deploy_regime import recall_over_subsets, COVERED
from panopin import robust_score

QS = [5, 10, 15, 20, 25, 30]
CACHES = [("whole-area", "wholearea_residuals.json"), ("6-room", "residuals.json")]


def main():
    rows = subset.build_subset()
    true = {r["pano_name"]: r["room"] for r in rows}
    grids_by = {lbl: json.load(open(paths.WORK / "seeds" / f)) for lbl, f in CACHES}

    print("# Low-percentile q-pin (recall over covered-room subsets)\n")
    for k in (4, 6):
        print(f"## k={k}")
        print("| q | " + " | ".join(lbl for lbl, _ in CACHES) + " | min |")
        print("|---|" + "---|" * (len(CACHES) + 1))
        # raw-mean baseline row
        base = {}
        for lbl, _ in CACHES:
            sm = robust_score.raw_mean_scores(grids_by[lbl])
            base[lbl], _ = recall_over_subsets(sm, true, COVERED, k)
        print(f"| raw-mean | " + " | ".join(f"{base[l]*100:.0f}%" for l, _ in CACHES)
              + f" | {min(base.values())*100:.0f}% |")
        for q in QS:
            cells = {}
            for lbl, _ in CACHES:
                sm = robust_score.robust_scores(grids_by[lbl], "low_percentile", q=q)
                cells[lbl], _ = recall_over_subsets(sm, true, COVERED, k)
            mn = min(cells.values())
            print(f"| q={q} | " + " | ".join(f"{cells[l]*100:.0f}%" for l, _ in CACHES)
                  + f" | {mn*100:.0f}% |")
        print()

    # --- mechanism: per-pano raw-mean vs low_pct(q=20), 6-room cache, full covered set ---
    q = 20
    print(f"## Rescue detail (6-room cache, k=6 full covered set): raw-mean vs low_pct q={q}")
    grids = grids_by["6-room"]
    raw = robust_score.raw_mean_scores(grids)
    lp = robust_score.robust_scores(grids, "low_percentile", q=q)
    print("| pano | true | raw-mean pick | low-pct pick | change |")
    print("|------|------|---------------|--------------|--------|")
    for p, tr in sorted(true.items(), key=lambda kv: kv[1]):
        rp = min(COVERED, key=lambda r: raw[p][r])
        lpp = min(COVERED, key=lambda r: lp[p][r])
        chg = ""
        if rp != tr and lpp == tr:   chg = "RESCUED"
        elif rp == tr and lpp != tr: chg = "BROKEN"
        print(f"| {p[:8]} | {tr} | {rp} | {lpp} | {chg} |")


if __name__ == "__main__":
    main()
