# Changelog

## 0.9.5.5

- Fixed ambiguous `target=1` behavior for auto-resolved multi-output
  prediction/custom outputs; integer targets now map to `target_index` once the
  resolved output is non-classification.
- Added probability range validation for `output="probability"` with
  `validate_probability` and `probability_tolerance`.
- Made `summary(style="plain")` a genuinely shorter, less technical report.
- Improved waterfall readability with wrapped labels, constrained layout, and
  a clearer distinction between residual/correction terms and aggregated
  `other features`.
- Documented neural-logit handling through `output_adapter` and clarified that
  `allocation_error_tolerance` controls allocation resolution rather than a
  formal feature-level error guarantee.

## 0.9.5.4

- Restored English-only public reports and plot labels for a global package
  surface.
- Removed non-English summary and plot text paths added during the previous
  iteration while keeping the baseline, adapter, residual-label, and simple API
  improvements.

## 0.9.5.3

- Added `baseline_mode="full" | "sample" | "auto"` plus baseline sampling
  diagnostics so users can distinguish true projection correction from
  background-sampling mismatch.
- Separated D'Hondt allocation votes from raw attribution value masses:
  `lambda_alliance_vote` now changes seat priority without inflating the
  explanatory value pool.
- Added `predict_kwargs` and `output_adapter` hooks for native boosters,
  framework-specific raw outputs, remote wrappers, and custom output matrices.
- Added class-label target selection, multiclass default-target warnings, and
  `log_odds` as an alias for probability log-odds.
- Added friendly residual labels and separated residual rendering in local bar
  and waterfall plots.
- Added the top-level `dhondtxai.explain(...)` one-call API and
  `explanation.plot(...)` convenience method.
- Enforced `max_model_rows` more strictly by capping background rows and
  screened interaction pairs before expensive scoring.
- Hardened adapter state restoration, PyTorch train/eval restoration, empty
  input behavior, and negative parliament-seat validation.

## 0.9.5.1

- Fixed local `DhondtExplanation.values` so the array always follows the
  original model feature order, including user-defined alliances whose member
  order differs from the training columns.
- Removed non-English public strings and metadata diacritics; public text output
  is English-only and unsupported language codes now raise a clear error.
- Added `projection_residual_threshold` so `projection_mode="auto"` keeps high
  projection corrections in `__projection_residual__` instead of silently
  redistributing them to features.
- Added mixed-sign alliance diagnostics and summary warnings.
- Added a simpler constructor path, `preset` defaults, `feature_names`,
  lightweight `dxai.maskers`, and `Explainer.from_score_function(...)`.
- Improved local bar and waterfall plots with horizontal bars, value labels,
  residual-aware coloring, and English-only labels.

## 0.9.5

- Added stricter input and output validation: duplicate feature names, NaN/inf
  model scores, and invalid D'Hondt votes now raise clear errors.
- Preserved D'Hondt scale invariance for very small positive vote values.
- Added `projection_mode="auto" | "redistribute" | "residual"` and a
  `__projection_residual__` bucket for explanations where raw D'Hondt evidence
  is absent or intentionally kept separate from feature attributions.
- Added strict residualization defaults for excluded features and
  below-threshold alliances so interaction effects are not silently assigned to
  active/eligible features.
- Added separate `lambda_alliance_vote` and `lambda_member_split` controls while
  keeping `lambda_interaction` as a backward-compatible shortcut.
- Improved `class_index=None` handling for binary probability and decision
  outputs, finite output checks, and deep compatibility probing.
- Improved PyTorch adapter device handling and added a clearer error for
  ambiguous multi-label list-of-arrays outputs.
- Implemented signed parliament palettes, English plot labels, projection
  quality badges, clearer snapped-seat legend text, and active use of
  `additional_rows`.
- Updated additive and runtime benchmarks to reduce sampling artifacts and
  include auto-alliance runtime measurements.

## 0.9.4

- Reworked parliament plots with a smaller inner half-circle and denser evidence
  ring closer to the legacy parliament layout.
- Replaced signed parliament red/blue gradients with a high-contrast
  qualitative palette so many groups remain visually separable.
- Improved global alliance co-occurrence heatmaps by hiding the uninformative
  diagonal, adding a clearer explanation, and showing a no-co-occurrence message
  when all features are singletons.
- Regenerated README visual examples with meaningful alliance co-occurrence
  blocks.

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

- Added `Explainer(model, background)` and callable values API:
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

- Repositioned DhondtXAI as a D'Hondt-projected
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
