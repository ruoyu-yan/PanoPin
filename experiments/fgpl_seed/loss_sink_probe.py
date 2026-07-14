"""Design-grounding probe for the loss-sink attack (D30 next-step). Using the low-pct q20
score on BOTH residual caches, characterize the hallway_3 loss-sink:
  (1) which covered rooms are sinks: #panos each is argmin-for vs #panos it actually owns;
  (2) THE pivotal question: within the sink, do its TRUE panos score LOWER than the panos it
      wrongly captures? If yes -> a per-room/assignment method over color scores can separate
      them (color suffices). If no -> color alone can't fix it; need another cue.
Offline, no GPU. GT used only to label true/false capture (analysis only, D5)."""
import os, sys, json
_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, "..", "..", "src"))
sys.path.insert(0, os.path.join(_HERE, "..", ".."))
from experiments.fgpl_seed import subset, paths
from panopin import robust_score

COVERED = subset.ROOMS


def main():
    rows = subset.build_subset()
    true = {r["pano_name"]: r["room"] for r in rows}
    for cache_lbl, fname in [("whole-area", "wholearea_residuals.json"),
                             ("6-room", "residuals.json")]:
        grids = json.load(open(paths.WORK / "seeds" / fname))
        sm = robust_score.low_percentile_scores(grids, q=20)   # per-pano per-room score
        picked = {p: min(COVERED, key=lambda r: sm[p][r]) for p in sm}

        print(f"\n================ {cache_lbl} cache (low-pct q20, {len(sm)} panos) ================")
        print("\n(1) capture profile per covered room:")
        print(f"{'room':11s} owns  captures  false-captures")
        for r in COVERED:
            owns = sum(1 for p in true if true[p] == r)
            caps = sum(1 for p in picked if picked[p] == r)
            false = sum(1 for p in picked if picked[p] == r and true[p] != r)
            tag = "  <-- SINK" if false >= 2 else ""
            print(f"{r:11s} {owns:4d}  {caps:8d}  {false:13d}{tag}")

        # (2) separability inside hallway_3
        sink = "hallway_3"
        print(f"\n(2) inside '{sink}': score of TRUE panos vs FALSELY-captured panos")
        print(f"{'pano':10s} {'belongs_to':11s} {'score@sink':>10s} {'score@true':>10s} {'sink-true':>10s} {'kind':>6s}")
        true_scores, false_scores = [], []
        for p in sm:
            s_sink = sm[p][sink]
            s_true = sm[p][true[p]]
            if true[p] == sink:
                kind = "TRUE"; true_scores.append(s_sink)
            elif picked[p] == sink:
                kind = "FALSE"; false_scores.append(s_sink)
            else:
                continue
            print(f"{p[:8]:10s} {true[p]:11s} {s_sink:10.4f} {s_true:10.4f} {s_sink - s_true:+10.4f} {kind:>6s}")
        if true_scores and false_scores:
            mx_true, mn_false = max(true_scores), min(false_scores)
            sep = mn_false > mx_true
            print(f"\n  true@sink max = {mx_true:.4f}   false@sink min = {mn_false:.4f}  "
                  f"=> {'SEPARABLE (a per-room threshold splits true<false)' if sep else 'OVERLAP (color score alone cannot split)'}")


if __name__ == "__main__":
    main()
