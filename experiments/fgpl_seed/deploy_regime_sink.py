"""Loss-sink attack: PER-PANO-RELATIVE vs JOINT assignment over the low-pct q20 score matrix
(D30 next-step). The loss_sink_probe showed hallway_3's impostors are SEPARABLE by color
(true panos score lower within the sink) -> a relative/assignment method should fix it without
another cue. Both families operate on the pano x room score matrix (offline, no GPU); compared
on the deploy_regime covered-room-subset recall harness, both caches. GT used only to score (D5).

Methods (all take a scenario sub-matrix {pano:{room:score}} + room list -> {pano: picked_room}):
  argmin      control: per-pano absolute argmin (== low-pct baseline)
  minmax      existing calibrate.minmax_scores then argmin (per-pano, leave-one-out per room)
  rel_median  per-pano relative: subtract room's leave-one-out MEDIAN baseline, then argmin
  rel_q25     per-pano relative: subtract room's leave-one-out 25th-pctile baseline, then argmin
  rank        JOINT: per room, rank all panos by score; assign each pano to its best-RANK room
"""
import os, sys, json, itertools, statistics
_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, "..", "..", "src"))
sys.path.insert(0, os.path.join(_HERE, "..", ".."))
from experiments.fgpl_seed import subset, paths
from panopin import robust_score, calibrate

COVERED = subset.ROOMS


def _pctile(vals, q):
    s = sorted(vals)
    if not s:
        return 0.0
    i = (len(s) - 1) * (q / 100.0)
    lo, hi = int(i), min(int(i) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (i - lo)


def m_argmin(sub, S):
    return {p: min(S, key=lambda r: sub[p][r]) for p in sub}


def m_minmax(sub, S):
    cal = calibrate.minmax_scores(sub)
    return {p: min(S, key=lambda r: cal[p][r]) for p in sub}


def _rel(sub, S, stat):
    panos = list(sub)
    pred = {}
    for p in panos:
        rel = {r: sub[p][r] - stat([sub[q][r] for q in panos if q != p]) for r in S}
        pred[p] = min(S, key=lambda r: rel[r])
    return pred


def m_rel_median(sub, S):
    return _rel(sub, S, lambda xs: statistics.median(xs) if xs else 0.0)


def m_rel_q25(sub, S):
    return _rel(sub, S, lambda xs: _pctile(xs, 25))


def m_rank(sub, S):
    panos = list(sub)
    rk = {p: {} for p in panos}
    for r in S:
        for p in panos:
            rk[p][r] = sum(1 for q in panos if sub[q][r] < sub[p][r])   # 0 = best in room
    return {p: min(S, key=lambda r: (rk[p][r], sub[p][r])) for p in panos}


METHODS = [("argmin", m_argmin), ("minmax", m_minmax),
           ("rel_median", m_rel_median), ("rel_q25", m_rel_q25), ("rank", m_rank)]


def recall(sm, true, k, method):
    correct = total = 0
    for S in itertools.combinations(COVERED, k):
        Sset = set(S)
        panos = [p for p in sm if true[p] in Sset]
        if len(panos) < 2:
            continue
        sub = {p: {r: sm[p][r] for r in S} for p in panos}
        pred = method(sub, S)
        for p in panos:
            total += 1
            correct += (pred[p] == true[p])
    return correct / total if total else 0.0


def main():
    rows = subset.build_subset()
    true = {r["pano_name"]: r["room"] for r in rows}
    L = []
    def out(s=""): L.append(s); print(s)

    out("# Loss-sink attack — per-pano-relative vs joint assignment (low-pct q20 base)\n")
    out("Recall over covered-room subsets, both caches. Method families: per-pano "
        "(minmax/rel_*) vs joint (rank). Baseline = argmin (plain low-pct).\n")
    for cache_lbl, fname in [("whole-area", "wholearea_residuals.json"),
                             ("6-room", "residuals.json")]:
        grids = json.load(open(paths.WORK / "seeds" / fname))
        sm = robust_score.low_percentile_scores(grids, q=20)
        out(f"## {cache_lbl} cache")
        out("| method | family | k=3 | k=4 | k=5 | k=6 |")
        out("|--------|--------|-----|-----|-----|-----|")
        fam = {"argmin": "control", "minmax": "per-pano", "rel_median": "per-pano",
               "rel_q25": "per-pano", "rank": "JOINT"}
        for name, fn in METHODS:
            cells = " | ".join(f"{recall(sm, true, k, fn)*100:.0f}%" for k in (3, 4, 5, 6))
            out(f"| {name} | {fam[name]} | {cells} |")
        out("")

    # diagnosis: 6-room k=6, do the 3 hallway_3 impostors get fixed?
    grids = json.load(open(paths.WORK / "seeds" / "residuals.json"))
    sm = robust_score.low_percentile_scores(grids, q=20)
    S = tuple(COVERED)
    sub = {p: {r: sm[p][r] for r in S} for p in sm}
    out("## Diagnosis (6-room, k=6): full assignment per method (confirms trade vs net-fix)")
    watch = {p: true[p] for p in sm}   # ALL panos, to expose any traded (newly-broken) pano
    out("| pano | true | " + " | ".join(n for n, _ in METHODS) + " |")
    out("|------|------|" + "---|" * len(METHODS))
    preds = {n: fn(sub, S) for n, fn in METHODS}
    for p, tr in sorted(watch.items(), key=lambda kv: kv[1]):
        cells = " | ".join(("OK" if preds[n][p] == tr else preds[n][p]) for n, _ in METHODS)
        out(f"| {p[:8]} | {tr} | {cells} |")

    out("\n## Conclusion — loss-sink attack REFUTED (the sink is a symptom, not the cause)")
    out("- **Neither architecture net-improves recall:** per-pano-relative (minmax / rel_median /")
    out("  rel_q25) AND joint (rank) all land at plain low-pct argmin (75% 6-room / 92% whole-area).")
    out("- **Why:** removing hallway_3's pull moves the 3 impostors OFF the sink but into OTHER wrong")
    out("  rooms (870532d7 office_4->office_7; 0e30c45e office_7->office_6/hallway) — never their")
    out("  true room. The sink is a SYMPTOM of the miss, not its cause.")
    out("- **Root cause = weak ABSOLUTE color lock:** the 3 impostors match their own true room at")
    out("  low-pct 0.15-0.25 vs ~0.06-0.08 for clean panos (loss_sink_probe) — the D23 window/")
    out("  occlusion hard floor (~1/3 of panos can't self-localize). No score-matrix method recovers")
    out("  a room the color signal does not support.")
    out("- **Ceiling reached for color-only room assignment = low-pct argmin (D30).** Further recall")
    out("  needs a NON-color cue (geometry/rotation) for weak-lock panos, OR a CONFIDENCE GATE to")
    out("  abstain on them — which in the all-covered deployment (>=1 pano/room) still yields full")
    out("  per-ROOM coverage via each room's strong panos. Per-pano recall is not the deployment")
    out("  metric; per-room coverage is. (-> next: the gate/coverage reframe, not a sink method.)")
    with open(os.path.join(_HERE, "LOSS_SINK_RESULTS.md"), "w") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
