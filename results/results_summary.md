# Results summary

Setup: 10 seeds x 119 Feynman problems = 1190 runs, random 75/25 split
(`random_state=seed`), `scale_x=False`, `scale_y=False`, run with the packaged
engine (`SHRv7Estimator(quick=False)`), jobs=3 (3 seeds in parallel, problems
serial within a seed). The fit/predict method does not read ground truth and
contains no per-problem branches; symbolic equivalence is checked afterwards by
an external harness. The fit does not query a lookup table.

## Overall

- Total results: 1190
- Exact (symbolic equivalence): 1190
- R2 train == 1: 1190
- R2 test == 1: 1190
- Non-exact / bad: 0
- All 119 problems exact on all 10 seeds: 119
- Driver wall time (jobs=3): 228.1 min
- Sum of per-problem fit time across all 1190 runs: 368.8 min
- Slowest single problem: test_6 (201.9 s)

## Per seed

| seed  | n   | exact | r2 train=1 | r2 test=1 | fit solve (min) |
|-------|-----|-------|------------|-----------|-----------------|
| 23654 | 119 | 119   | 119        | 119       | 34.7            |
| 15795 | 119 | 119   | 119        | 119       | 39.3            |
| 860   | 119 | 119   | 119        | 119       | 37.3            |
| 539   | 119 | 119   | 119        | 119       | 38.2            |
| 16850 | 119 | 119   | 119        | 119       | 35.1            |
| 29910 | 119 | 119   | 119        | 119       | 37.0            |
| 4426  | 119 | 119   | 119        | 119       | 37.5            |
| 219   | 119 | 119   | 119        | 119       | 36.4            |
| 14423 | 119 | 119   | 119        | 119       | 35.8            |
| 28020 | 119 | 119   | 119        | 119       | 37.5            |

## Slowest single runs (all exact)

| problem | seed | time (s) |
|---------|------|----------|
| test_6  | 4426 | 201.9    |
| test_6  | 219  | 200.4    |
| test_6  | 860  | 197.8    |
| test_17 | 15795 | 185.8  |
| test_6  | 539  | 177.4    |
| test_3  | 15795 | 176.0  |
| test_6  | 29910 | 174.1  |
| test_8  | 539  | 143.4    |
| test_8  | 23654 | 137.1  |
| test_8  | 860  | 126.2    |

All 1190/1190 exact (symbolic equivalence, R2 train = R2 test = 1).
