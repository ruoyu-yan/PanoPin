# Driver isolation — why does a′(raw-mean) beat a(deployed) on the gate? (n=12)

prefix_correct = panos confidently+correctly committed before the first wrong room.
All a′-variants are unweighted grid means; only match_color/subsample differ. `a` is the
deployed match_color+weighted+CPO-subsample cache loss.

| variant | prefix_correct | recall@1 |
|---------|----------------|----------|
| a  (matchcolor+weighted+CPOsub) [cache] | 3/12 | 8/12 |
| a' (raw+unweighted+seed0) | 9/12 | 9/12 |
| a'_mc (matchcolor+unweighted+seed0) | 6/12 | 7/12 |
| a'_seed1 (raw+unweighted+seed1) | 9/12 | 9/12 |
| a'_seed2 (raw+unweighted+seed2) | 9/12 | 9/12 |

**Gap to attribute: a=3/12 → a′=9/12 (Δ=6).**
- **match_color**: a′_mc=6/12. Turning match_color ON drops a′ by 3 → match_color is a **MAJOR driver** of the gap.
- **subsample**: a′_seed1=9, a′_seed2=9 vs a′=9 (max Δ=0) → subsample is a **NOT a driver**.
- **weighting** (by elimination): a′_mc=6/12 vs deployed a=3/12. The part of the a→a′ gap NOT explained by match_color/subsample is weighting; a′_mc still >> a by 3 → weighting is a **MAJOR driver** of the gap.
