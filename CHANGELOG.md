# Changelog

## 0.9.3

- Fixed custom two-dimensional score output selection so `class_index` and
  `class_index="predicted"` are respected when `target_index` is not provided.
- Fixed one-dimensional and single-column binary probability outputs so class 0
  explains `1 - p` and class 1 explains `p`.
- Strengthened high projection-residual warnings in summaries and plots.
- Applied alliance-level `top_k` in local explanation plots.
- Added residual-hidden warnings to waterfall plots.
- Optimized signed parliament visualization with readable seat snapping,
  configurable display seat counts, zero-seat messages, and capped legends.
- Added current visual example generation and refreshed `exampleimages/`.
- Stabilized XGBoost tests and CI by limiting native thread pools.

## 0.9.2

- Added SHAP-like `Explainer(model, background)` and callable values API:
  `dhondtxai_values = explainer(X)`.
- Added `DhondtValues` with `.values`, `.dhondtxai_values`, `.base_values`,
  `.scores`, `.feature_names`, and row-level explanation access.
- Added `score_fn` as the public custom scoring-function alias.
- Added `model_adapter="auto"` scoring support for native XGBoost Booster,
  LightGBM Booster, CatBoost, PyTorch modules, and Keras-like models.
- Added `input_format="auto"` default to select DataFrame, NumPy, DMatrix, or
  Tensor inputs based on the resolved adapter.
- Added optional dependency extras for XGBoost, LightGBM, CatBoost, Torch, and
  all model adapters.
- Added compatibility metadata reporting for resolved model adapters.
- Added adapter tests for XGBoost, LightGBM, CatBoost, PyTorch, and Keras-like
  models when those optional packages are installed.

## 0.9.1

- Added PyPI-ready metadata, README long description, and package `__version__`.
- Fixed regression/prediction summaries so they no longer report a classifier
  class target.
- Fixed `check_model_compatibility(X_sample=...)` when no background has been
  set yet.
- Fixed NumPy, integer, and mixed-type feature names in alliance display and
  pairwise interaction keys.
- Added clearer projection residual warnings in text summaries.
- Added regression, compatibility, mixed feature-name, and packaging tests.

## 0.9.0

- Repositioned DhondtXAI as a SHAP-independent D'Hondt-projected
  removal-effect attribution operator.
- Added safer default attribution resolution with high `allocation_seats`.
- Added `resolved_output_type`, `perturbation`, `affinity_mode`, and
  `tie_break` metadata to local explanations.
- Added `conditional_knn` perturbation, explicit D'Hondt tie-breaking, and
  absolute-interaction affinity.
- Added batch removal scoring and heap-based D'Hondt allocation.
- Fixed default `target_index=None` behavior for one-column 2D prediction
  outputs.
- Added residual-aware global random seed handling.
- Added self-contained sklearn usage example and expanded unit tests.
