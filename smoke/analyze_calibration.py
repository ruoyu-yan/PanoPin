"""Offline (no-GPU) sweep of calibration formulas over the saved loss matrix
(runs/calib_matrix.json from exp_calibration.py). For each formula, lower score = better;
we rank the true room among all candidates and report recall@1/3/5. Per-room baseline
stats are leave-one-out: computed over query rows whose true room != that candidate."""
import argparse, json, os, sys
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

    # per-candidate leave-one-out baseline arrays: for candidate j, use rows i whose true room != cands[j]
    def base_stat(fn):
        out = np.zeros(R)
        for j in range(R):
            vals = np.array([L[i, j] for i in range(P) if cands[true_idx[i]] != cands[j]])
            out[j] = fn(vals) if len(vals) else 0.0
        return out
    med = base_stat(np.median); mean = base_stat(np.mean)
    std = base_stat(lambda v: np.std(v)); mn = base_stat(np.min); mx = base_stat(np.max)

    def pct_score(L):  # percentile of each loss within its candidate's baseline distribution
        S = np.zeros_like(L)
        for j in range(R):
            vals = np.array([L[i, j] for i in range(P) if cands[true_idx[i]] != cands[j]])
            for i in range(P):
                S[i, j] = np.mean(vals <= L[i, j]) if len(vals) else 0.5
        return S

    formulas = {
        "raw":            L,
        "sub_median":     L - med,
        "sub_mean":       L - mean,
        "sub_half_med":   L - 0.5 * med,
        "zscore":         (L - mean) / (std + eps),
        "minmax":         (L - mn) / (mx - mn + eps),
        "percentile":     pct_score(L),
    }

    def recall(score):
        ranks = []
        for i in range(P):
            order = np.argsort(score[i]); ranks.append(int(np.where(order == true_idx[i])[0][0]) + 1)
        return ranks

    print(f"{'formula':14s} {'recall@1':>9s} {'recall@3':>9s} {'recall@5':>9s}   per-pano ranks")
    for name, score in formulas.items():
        rk = recall(score)
        r1 = sum(x <= 1 for x in rk); r3 = sum(x <= 3 for x in rk); r5 = sum(x <= 5 for x in rk)
        print(f"{name:14s} {r1:>4d}/{P:<4d} {r3:>4d}/{P:<4d} {r5:>4d}/{P:<4d}   {rk}")
    print(f"\nquery rooms order: {[r['query_room'] for r in rows]}")

if __name__ == "__main__":
    main()
