"""Strictly-Manhattan room assignment (2026-07-16 demo, offline / no GPU).

Re-runs the D30 (low-pct score) + D32 (per-room coverage) metrics with BOTH the candidate rooms
and the panos confined to Manhattan-world rooms (see `manhattan.py` for why neither previously
validated pool qualifies). Reuses the D33 metric functions verbatim so numbers are directly
comparable to LARGEVAL_RESULTS.md. Cached grids only — each (pano, room) localization is
independent, so restricting the candidate set offline == having only searched these rooms.

GT is used only to score (D5). ->  MANHATTAN_RESULTS.md
    conda run -n panopin python -m experiments.fgpl_seed.manhattan_analysis
"""
import os
import sys
import json

_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, "..", "..", "src"))
sys.path.insert(0, os.path.join(_HERE, "..", ".."))

from experiments.fgpl_seed import manhattan, largeval, paths
from experiments.fgpl_seed.largeval_analysis import (
    per_pano_recall, room_anchored_cov, gated_cov_at_prec1)
from panopin import robust_score, coverage

POOL = manhattan.POOL_ROOMS


def _restrict(matrix, panos, rooms):
    """Confine a {pano: {room: score}} matrix to the Manhattan panos x Manhattan rooms."""
    return {p: {r: matrix[p][r] for r in rooms} for p in panos}


def main():
    rows = manhattan.build_pool()
    true = {r["pano_name"]: r["room"] for r in rows}
    panos = list(true)

    grids = json.load(open(paths.WORK / "seeds" / "largeval_residuals.json"))
    cache = json.load(open(paths.WORK / "seeds" / "largeval_cache.json"))
    sm = _restrict(robust_score.low_percentile_scores(grids, q=20), panos, POOL)
    cpo_sm = _restrict({p: cache[p]["per_room"] for p in cache}, panos, POOL)

    ks = [3, 4, 5]
    L = []

    def out(s=""):
        L.append(s)
        print(s)

    out(f"# Strictly-Manhattan room assignment — {len(panos)} panos, {len(POOL)} rooms\n")
    out("Both the candidate rooms and the panos are confined to Manhattan-world rooms "
        "(Point_360 `roadmap.md` §5), matching the Stage-3 3-orthogonal-direction assumption.")
    out(f"Pool: {', '.join(POOL)}.")
    out(f"Excluded as non-Manhattan (real diagonal wall): "
        f"{', '.join(r for r in largeval.POOL_ROOMS if r not in POOL)}.")
    out("Offline rescoring of cached D33 grids (no GPU); GT used only to score.\n")

    out("## Per-pano recall@1 (low-pct argmin vs raw CPO min-loss)")
    out("| score | " + " | ".join(f"k={k}" for k in ks) + " |")
    out("|" + "---|" * (len(ks) + 1))
    out("| low-pct (deployed) | "
        + " | ".join(f"{per_pano_recall(sm, true, POOL, k)*100:.0f}%" for k in ks) + " |")
    out("| CPO min-loss | "
        + " | ".join(f"{per_pano_recall(cpo_sm, true, POOL, k)*100:.0f}%" for k in ks) + " |")

    out("\n## Per-ROOM coverage (the deployment metric, D32)")
    out("| method | " + " | ".join(f"k={k}" for k in ks) + " |")
    out("|" + "---|" * (len(ks) + 1))
    out("| room-anchored (correct seeds) | "
        + " | ".join(f"{room_anchored_cov(sm, true, POOL, k)*100:.0f}%" for k in ks) + " |")
    out("| winner-gate (cov @ precision 1.0) | "
        + " | ".join(f"{gated_cov_at_prec1(sm, true, POOL, k)*100:.0f}%" for k in ks) + " |")

    # --- the actual demo configuration: the full 5-room Manhattan map ---
    seeds = coverage.room_anchored_seeds(sm)
    conf = coverage.pano_confidence(sm)
    n_ok = sum(1 for r, (p, _) in seeds.items() if true[p] == r)
    out(f"\n## Demo configuration (all {len(POOL)} rooms) — room-anchored seeds")
    out(f"Each room picks its own best-matching pano. **{n_ok}/{len(POOL)} seeds correct.**\n")
    out("| room | seed pano | score | correct? |")
    out("|---|---|---|---|")
    for r in POOL:
        p, s = seeds[r]
        out(f"| {r} | `{p[:8]}` | {s:.4f} | {'YES' if true[p] == r else 'NO'} |")

    ranked = sorted(conf.items(), key=lambda kv: -kv[1][1])
    n_gate = sum(1 for p, (r, _) in conf.items() if r == true[p])
    out(f"\n## Per-pano confidence ranking (winner-score gate)")
    out(f"Per-pano argmin is correct for **{n_gate}/{len(panos)}** panos. Sorted most-confident "
        "first; a gate admits from the top. Weak-lock panos (D23) should sink to the bottom.\n")
    out("| rank | pano | picked room | true room | score | correct? |")
    out("|---|---|---|---|---|---|")
    for i, (p, (r, c)) in enumerate(ranked, 1):
        out(f"| {i} | `{p[:8]}` | {r} | {true[p]} | {-c:.4f} | "
            f"{'YES' if r == true[p] else 'NO'} |")

    raw_hits = sum(1 for p in panos if min(cpo_sm[p], key=lambda r: cpo_sm[p][r]) == true[p])
    ra5 = room_anchored_cov(sm, true, POOL, 5)
    wg5 = gated_cov_at_prec1(sm, true, POOL, 5)
    out("\n## Verdict")
    out(f"- Raw CPO min-loss picks the right room for {raw_hits}/{len(panos)} panos — the pool is "
        "not benign (hallway_1 is a real loss-sink).")
    out(f"- **Per-room coverage at the demo config (k=5): room-anchored {ra5*100:.0f}%, "
        f"winner-gate@precision-1 {wg5*100:.0f}%.**")
    out(f"- Per-pano recall (low-pct) {per_pano_recall(sm, true, POOL, 5)*100:.0f}% at k=5 — the "
        "D31/D23 color ceiling is present here too, and (per D32) does not bind per-room coverage.")

    (paths.HERE / "MANHATTAN_RESULTS.md").write_text("\n".join(L) + "\n")
    print(f"\nwrote {paths.HERE / 'MANHATTAN_RESULTS.md'}")


if __name__ == "__main__":
    main()
