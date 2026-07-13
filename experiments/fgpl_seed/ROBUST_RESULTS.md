# Robust-gate comparison (precision/coverage, n=12)

Metric: **prefix_correct** = panos confidently+correctly committed before the FIRST wrong
room (= coverage-at-100%%-precision x n). recall@1 = correct at full coverage.

| method | prefix_correct (cov@100%%prec) | recall@1 (all) |
|--------|------------------------------|----------------|
| a: mean-loss | 3/12 (25%) | 8/12 (67%) |
| a': raw-mean | 9/12 (75%) | 9/12 (75%) |
| c: median | 8/12 (67%) | 9/12 (75%) |
| c: low-pct@10 | 9/12 (75%) | 9/12 (75%) |
| c: low-pct@20 | 9/12 (75%) | 9/12 (75%) |
| c: low-pct@25 | 9/12 (75%) | 9/12 (75%) |
| c: trim-top@10 | 9/12 (75%) | 9/12 (75%) |
| c: trim-top@20 | 9/12 (75%) | 9/12 (75%) |
| c: trim-top@30 | 9/12 (75%) | 9/12 (75%) |

**Deployed mean gate (a) = 3/12; raw-mean (a') = 9/12; best robust (c) = c: low-pct@10 at 9/12.**
- Robustness isolated (same raw residuals): c does NOT beat a' (9 vs 9).
- Deployment: c BEATS the deployed match_color+weighted mean gate a (9 vs 3).
(n=12 is noisy — D26 caveat; treat small differences as within noise.)
