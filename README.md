# SHR

SHR is a symbolic regression system that combines deterministic structural
probes with exact symbolic solving and a genetic-programming fallback. It
narrows the search using multiplicative, additive, algebraic, trigonometric,
gaussian, and rational structural cues, re-validates candidates across
multiple splits, and respects a global time budget. It does not train
weights, query a lookup table, or use per-problem branches.

## Repository layout

```
.
├── engine/                      # flat, self-contained engine modules
├── experiment/methods/shr/
│   ├── regressor.py             # sklearn-style estimator (fit/predict/model)
│   └── __init__.py
├── algorithms/shr/
│   ├── metadata.yml             # method metadata
│   ├── environment.yml          # conda environment
│   ├── install.sh               # engine install (local engine first, else URL)
│   └── LICENSE                  # MIT
├── results/
│   └── results_summary.md       # 1190/1190 exact summary
└── README.md
```

## Requirements

- Python 3.10
- numpy 1.26.4, pandas, scipy, scikit-learn
- sympy 1.14.0
- pyoperon 0.6.1 (GP fallback)
- psutil
- torch 2.1.2 (a CPU build is sufficient; the MLP falls back to CPU)

## Install

### Conda environment

```bash
conda env create -f algorithms/shr/environment.yml
conda activate shr
```

### Engine

The engine is a set of flat `.py` files. `algorithms/shr/install.sh` places them
in `$CONDA_PREFIX/shr_engine/` and writes a `.pth` file so the flat imports
work. It uses the local `engine/` directory when present (offline), and
otherwise downloads the release tarball from `SHR_URL`.

```bash
bash algorithms/shr/install.sh
```

For a quick local run without installing, point `SHR_ENGINE` at the engine
directory:

```bash
export SHR_ENGINE=$(pwd)/engine
```

## Running in SRBench

Place `experiment/methods/shr/regressor.py` and `algorithms/shr/` in a SRBench
checkout, then:

```bash
cd experiment
python evaluate_model.py <data.tsv.gz> -ml shr -results_path results -seed 23654
```

## Estimator interface

The estimator follows the sklearn `BaseEstimator / RegressorMixin` API:

```python
from regressor import SHRv7Estimator
est = SHRv7Estimator(random_state=23654, max_time=25)
est.fit(X_train, y_train)
y_pred = est.predict(X_test)
model_string = est.symbolic_model
```

- `max_time` (seconds): a global per-problem budget. When exceeded, the engine
  stops entering new costly paths and finalizes the current best candidate.
- `quick=True`: a small budget for smoke tests.
- `eval_kwargs` disables scaling (`scale_x=False`, `scale_y=False`) and keeps a
  DataFrame so the original feature names are passed through.

## Results

On the 119 Feynman problems across the 10 SRBench seeds (random 75/25 split,
`scale_x=False`), SHR returns **1190/1190 exact** symbolic models, with
R2 train = R2 test = 1 and no non-exact/bad runs. Every problem is exact on all
10 seeds. The fit/predict method does not read ground truth; symbolic
equivalence is verified by an external harness after fitting. The parallel
driver (jobs=3) finishes in 228.1 min.
See [results/results_summary.md](results/results_summary.md).

## License

MIT. See [algorithms/shr/LICENSE](algorithms/shr/LICENSE).
