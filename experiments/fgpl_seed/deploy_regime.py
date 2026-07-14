"""Deployment-regime characterization (OFFLINE, no GPU).

The real deployment is NOT 12 panos vs all 23 rooms. It is a 3-5 room point cloud where
EVERY room is covered by >=1 pano (user, 2026-07-14). This rescores the cached whole-area
residual grids (12 panos x 23 rooms, 101-percentile per pair) restricted to subsets of the
6 COVERED rooms, to answer:

  Q1  recall vs candidate-set size k (k rooms, all covered)  -- the real target metric
  Q2  which per-room score statistic maximises recall in that regime (raw-mean vs robust vs CPO)
  Q3  which panos still fail at k=6, and which room out-competes truth (loss-sink diagnosis)

Design note: for a fixed k, every pano's true room lies in exactly C(5, k-1) of the C(6,k)
covered-room subsets, so micro-recall == macro-recall (uniform pano weight). Each pano is thus
scored across all ways its (k-1) distractor rooms can be drawn from the other 5 covered rooms.
Fair: reads only residual grids + CPO per_room losses; GT (true room) used ONLY to score (D5).
Reproduces WHOLEAREA_RESULTS.md at k=23 as a fidelity check. -> DEPLOY_REGIME_RESULTS.md.
"""
import os, sys, json, itertools
_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, "..", "..", "src"))
sys.path.insert(0, os.path.join(_HERE, "..", ".."))
from experiments.fgpl_seed import subset, paths
from panopin import robust_score

COVERED = subset.ROOMS   # the 6 rooms actually covered by the 12 panos


def _argmin_correct(score_p, cand, true_room):
    """Is argmin over candidate rooms == true room? (ties: first by cand order)."""
    best = min(cand, key=lambda r: score_p[r])
    return best == true_room


def recall_over_subsets(sm, true, cand_pool, k):
    """Mean per-pano recall over all size-k subsets of cand_pool (each subset scores only the
    panos whose true room is in it). Returns (recall_fraction, n_panos_evaluated_total)."""
    correct = total = 0
    for S in itertools.combinations(cand_pool, k):
        Sset = set(S)
        for p, tr in true.items():
            if tr in Sset:
                total += 1
                if _argmin_correct(sm[p], S, tr):
                    correct += 1
    return (correct / total if total else 0.0), total


def whole_recall(sm, true, cand):
    """Plain argmin recall over the full candidate list (no subsetting)."""
    return sum(1 for p in sm if _argmin_correct(sm[p], cand, true[p]))


def build_scores(grids, cache):
    """All candidate per-room score matrices to compare (lower = better room)."""
    cpo = {p: cache[p]["per_room"] for p in cache}
    stats = {
        "CPO-loss":            cpo,
        "raw-mean":            robust_score.raw_mean_scores(grids),
        "median":              robust_score.robust_scores(grids, "median"),
        "trimmed_mean_k10":    robust_score.robust_scores(grids, "trimmed_mean", k=10),
        "trimmed_mean_k20":    robust_score.robust_scores(grids, "trimmed_mean", k=20),
        "trimmed_mean_k30":    robust_score.robust_scores(grids, "trimmed_mean", k=30),
        "low_pct_q10":         robust_score.robust_scores(grids, "low_percentile", q=10),
        "low_pct_q25":         robust_score.robust_scores(grids, "low_percentile", q=25),
        "low_pct_q40":         robust_score.robust_scores(grids, "low_percentile", q=40),
    }
    return stats


def main():
    rows = subset.build_subset()
    true = {r["pano_name"]: r["room"] for r in rows}
    n = len(true)
    cache = json.load(open(paths.WORK / "seeds" / "wholearea_cache.json"))
    grids = json.load(open(paths.WORK / "seeds" / "wholearea_residuals.json"))
    all_rooms = sorted(next(iter(grids.values())).keys())
    scores = build_scores(grids, cache)

    L = []
    def out(s=""): L.append(s); print(s)

    out(f"# Deployment-regime characterization — recall vs candidate-set size ({n} panos)\n")
    out("Real deployment = a k-room cloud, EVERY room covered by a pano. Rescores the cached")
    out("whole-area residual grids restricted to subsets of the 6 covered rooms. Offline, no GPU.")
    out(f"Covered rooms (6): {', '.join(COVERED)}. n={n} panos (2/room).\n")

    # --- Fidelity check: reproduce WHOLEAREA_RESULTS.md at k=23 (full candidate list) ---
    out("## Fidelity check (k=23, full candidate list — must match WHOLEAREA_RESULTS.md)")
    out(f"| score | whole-area recall@1 |")
    out(f"|-------|---------------------|")
    for lbl in ("CPO-loss", "raw-mean"):
        out(f"| {lbl} | {whole_recall(scores[lbl], true, all_rooms)}/{n} |")
    out("(expect CPO-loss 6/12, raw-mean 8/12)\n")

    # --- Q1: recall vs k for the two headline scores ---
    out("## Q1 — recall vs candidate-set size k (all rooms covered)")
    ks = [2, 3, 4, 5, 6]
    hdr = "| score | " + " | ".join(f"k={k}" for k in ks) + " | k=23 |"
    out(hdr); out("|" + "---|" * (len(ks) + 2))
    for lbl in ("CPO-loss", "raw-mean"):
        cells = []
        for k in ks:
            rec, _ = recall_over_subsets(scores[lbl], true, COVERED, k)
            cells.append(f"{rec*100:.0f}%")
        wa = whole_recall(scores[lbl], true, all_rooms) / n
        out(f"| {lbl} | " + " | ".join(cells) + f" | {wa*100:.0f}% |")
    out("")

    # --- Q2: full score-statistic sweep at the deployment sizes ---
    out("## Q2 — best per-room score statistic, by k (covered-room subsets)")
    ks2 = [3, 4, 5, 6]
    out("| score statistic | " + " | ".join(f"k={k}" for k in ks2) + " | k=23 |")
    out("|" + "---|" * (len(ks2) + 2))
    best_by_k = {k: (None, -1.0) for k in ks2 + [23]}
    for lbl, sm in scores.items():
        cells = []
        for k in ks2:
            rec, _ = recall_over_subsets(sm, true, COVERED, k)
            cells.append(rec)
            if rec > best_by_k[k][1]:
                best_by_k[k] = (lbl, rec)
        wa = whole_recall(sm, true, all_rooms) / n
        if wa > best_by_k[23][1]:
            best_by_k[23] = (lbl, wa)
        row = " | ".join(f"{c*100:.0f}%" for c in cells)
        out(f"| {lbl} | {row} | {wa*100:.0f}% |")
    out("")
    for k in ks2 + [23]:
        lbl, rec = best_by_k[k]
        out(f"- best @ k={k}: **{lbl}** ({rec*100:.0f}%)")
    out("")

    # --- Q3: failure diagnosis at k=6 (full covered set), raw-mean ---
    out("## Q3 — failures at k=6 (full covered set), raw-mean score")
    sm = scores["raw-mean"]
    out("| pano | true room | picked | pick<true margin | verdict |")
    out("|------|-----------|--------|------------------|---------|")
    sink = {}
    for p, tr in sorted(true.items(), key=lambda kv: kv[1]):
        picked = min(COVERED, key=lambda r: sm[p][r])
        margin = sm[p][tr] - sm[p][picked]      # >0 => truth lost by this much
        ok = "OK" if picked == tr else "MISS"
        if picked != tr:
            sink[picked] = sink.get(picked, 0) + 1
        out(f"| {p[:8]} | {tr} | {picked} | {margin:+.4f} | {ok} |")
    if sink:
        out("\nloss-sink competitors (room: #panos it wrongly captured): "
            + ", ".join(f"{r}={c}" for r, c in sorted(sink.items(), key=lambda kv: -kv[1])))
    out("")

    md = "\n".join(L) + "\n"
    with open(os.path.join(_HERE, "DEPLOY_REGIME_RESULTS.md"), "w") as f:
        f.write(md)


if __name__ == "__main__":
    main()
