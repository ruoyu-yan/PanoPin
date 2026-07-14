
## Q4 — does minmax calibration lift the deployment regime? (low-pct q=20)

### whole-area cache
| score | gate | k=3 | k=4 | k=5 | k=6 |
|-------|------|-----|-----|-----|-----|
| CPO-loss | argmin | 88% | 87% | 85% | 83% |
| CPO-loss | minmax | 94% | 92% | 92% | 92% |
| raw-mean | argmin | 88% | 87% | 85% | 83% |
| raw-mean | minmax | 89% | 88% | 85% | 83% |
| low-pct q20 | argmin | 92% | 92% | 92% | 92% |
| low-pct q20 | minmax | 88% | 88% | 90% | 92% |

### 6-room cache
| score | gate | k=3 | k=4 | k=5 | k=6 |
|-------|------|-----|-----|-----|-----|
| CPO-loss | argmin | 78% | 73% | 70% | 67% |
| CPO-loss | minmax | 85% | 77% | 72% | 67% |
| raw-mean | argmin | 77% | 73% | 70% | 67% |
| raw-mean | minmax | 78% | 78% | 77% | 75% |
| low-pct q20 | argmin | 75% | 75% | 75% | 75% |
| low-pct q20 | minmax | 76% | 75% | 75% | 75% |

