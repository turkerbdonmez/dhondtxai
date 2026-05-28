# Changelog

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
