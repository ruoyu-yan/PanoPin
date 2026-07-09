"""FAIR (no-GT) offline sweep of calibration formulas over the saved loss matrix
(runs/calib_matrix.json). Per-room baselines are leave-one-out on the PANO ONLY — they do
NOT use room labels (an earlier version excluded rows by true room, which LEAKED GT and
inflated the results; see DECISIONS). Lower score = better; we rank the true room and
report recall@1/3/5. This is what a real solver (which has no GT) can actually achieve.

Also reports, for the best formula, the winner-score of correct vs wrong panos — the
honest test of whether a confidence gate can separate them.
"""
import argparse, json, os
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", default=os.path.join(os.path.dirname(__file__), "..", "runs", "calib_matrix.json"))
    args = ap.parse_args()
    d = json.load(open(args.matrix))
    cands = d["candidates"]; rows = d["rows"]
    L = np.array([[r["losses"][c] for c in cands] for r in rows])  # (P, R)
    true_idx = [cands.index(r["query_room"]) for r in rows]
    P, R = L.shape
    eps = 1e-9

    # FAIR leave-one-out baseline stat for (pano i, room j): stat over losses L[k, j], k != i.
    def loo(i, j, fn):
        vals = np.array([L[k, j] for k in range(P) if k != i])
        return fn(vals) if len(vals) else 0.0

    def build(score_fn):
        return np.array([[score_fn(i, j) for j in range(R)] for i in range(P)])

    formulas = {
        "raw":         build(lambda i, j: L[i, j]),
        "sub_median":  build(lambda i, j: L[i, j] - loo(i, j, np.median)),
        "sub_mean":    build(lambda i, j: L[i, j] - loo(i, j, np.mean)),
        "zscore":      build(lambda i, j: (L[i, j] - loo(i, j, np.mean)) / (loo(i, j, np.std) + eps)),
        "minmax":      build(lambda i, j: (L[i, j] - loo(i, j, np.min)) / (loo(i, j, np.max) - loo(i, j, np.min) + eps)),
        "percentile":  build(lambda i, j: np.mean(np.array([L[k, j] for k in range(P) if k != i]) <= L[i, j])),
        # percentile with raw-loss tiebreak (percentile dominates; raw loss breaks ties)
        "pctl+raw":    None,
    }
    # pctl+raw as a composite score: percentile*1000 + raw loss
    formulas["pctl+raw"] = formulas["percentile"] * 1000.0 + L

    def ranks(score):
        return [int(np.where(np.argsort(score[i]) == true_idx[i])[0][0]) + 1 for i in range(P)]

    print(f"{'formula':12s} {'recall@1':>9s} {'recall@3':>9s} {'recall@5':>9s}   per-pano ranks")
    best = None
    for name, score in formulas.items():
        rk = ranks(score)
        r1 = sum(x <= 1 for x in rk); r3 = sum(x <= 3 for x in rk); r5 = sum(x <= 5 for x in rk)
        print(f"{name:12s} {r1:>4d}/{P:<4d} {r3:>4d}/{P:<4d} {r5:>4d}/{P:<4d}   {rk}")
        if best is None or r1 > best[1]:
            best = (name, r1, score)

    # confidence separation for the best formula: winner score for correct vs wrong panos
    name, _, score = best
    cor, wr = [], []
    for i in range(P):
        order = np.argsort(score[i]); w = order[0]
        (cor if w == true_idx[i] else wr).append(round(float(score[i][w]), 3))
    print(f"\nbest formula = {name}. winner-score of CORRECT panos: {sorted(cor)}")
    print(f"best formula = {name}. winner-score of WRONG   panos: {sorted(wr)}")
    print(f"query rooms: {[r['query_room'] for r in rows]}")


if __name__ == "__main__":
    main()
