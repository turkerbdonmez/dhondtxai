# DhondtXAI Benchmark Plan

This directory contains lightweight reproducible benchmark scripts and a plan
for larger SHAP/LIME comparison studies.

Current scripts:

- `synthetic_additive.py`: compares DhondtXAI against a known additive linear
  ground truth.
- `synthetic_interaction.py`: checks whether automatic alliances expose a
  synthetic interaction.
- `runtime_scaling.py`: measures runtime as feature count increases.
- `stability_bootstrap.py`: estimates global-importance rank stability across
  two background seeds.

Run any script from the repository root:

```bash
python benchmarks/synthetic_additive.py
```

Scripts write CSV outputs under `benchmarks/results/`.

Recommended benchmark families:

- Synthetic linear additive data with known attribution ground truth.
- Synthetic interaction/XOR-style data for alliance behavior.
- Correlated and redundant tabular features for perturbation sensitivity.
- Monotonic nonlinear models for ranking stability.
- Breast cancer and diabetes sklearn datasets for package examples.
- Mixed numeric/categorical tabular pipelines.
- RandomForest, logistic regression, SVM, MLP, XGBoost/CatBoost wrappers, and
  custom `predict_fn` models.

Recommended metrics:

- Local completeness error.
- Projection residual ratio and absolute projection residual.
- Deletion/insertion faithfulness.
- Bootstrap stability.
- Sensitivity to `n_background`, `allocation_seats`, and perturbation mode.
- Runtime versus feature count and interaction setting.
- Rank correlation against SHAP, LIME, and permutation-importance baselines.
