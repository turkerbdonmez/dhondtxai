# DhondtXAI

DhondtXAI is a SHAP-independent, D'Hondt-based post-hoc attribution library for
tabular models. It does not compute SHAP values or approximate Shapley values.
Instead, it defines a separate D'Hondt-projected removal-effect attribution
operator. SHAP can still be used as an external benchmark.

DhondtXAI can explain any tabular model that can be adapted to a row-wise
numeric scoring function. It includes automatic adapters for common model
families such as sklearn-style estimators, XGBoost, LightGBM, CatBoost, PyTorch
modules, and Keras-like models.

> **Status:** DhondtXAI 0.9.2 is an experimental/beta tabular XAI library. It is
> suitable for research, model inspection, and controlled pilot use. For
> high-stakes deployment, validate explanations against task-specific benchmarks
> and compare them with established methods such as SHAP and LIME.

## Install

```bash
pip install dhondtxai==0.9.2
```

For local development from this repository:

```bash
pip install -e .[dev]
```

`scikit-learn` and other ML frameworks are optional for the core library. Use
extras when you want adapter dependencies:

```bash
pip install "dhondtxai[sklearn]"
pip install "dhondtxai[xgboost]"
pip install "dhondtxai[lightgbm]"
pip install "dhondtxai[catboost]"
pip install "dhondtxai[torch]"
pip install "dhondtxai[all-models]"
```

## Quick Start

```python
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from dhondtxai import Explainer

dataset = load_breast_cancer()
X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
y = pd.Series(dataset.target)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)
explainer = Explainer(model, X_train)

dhondtxai_values = explainer(
    X_test.head(5),
    threshold=0.05,
    redistribute=True,
    n_background=50,
)

print(dhondtxai_values.values)
print(dhondtxai_values.base_values)
print(dhondtxai_values.to_frame(row=0, top_k=10))
print(dhondtxai_values.summary(row=0))

explainer.plot_waterfall(dhondtxai_values[0], top_k=10)
```

`dhondtxai_values.values` is the DhondtXAI equivalent of SHAP-style attribution
arrays. The values are not Shapley values; they are D'Hondt-projected
removal-effect attributions.

## SHAP-Like Values API

The recommended public API mirrors the familiar `shap.Explainer` workflow:

```python
import dhondtxai as dxai

explainer = dxai.Explainer(model, X_background)
dhondtxai_values = explainer(X_to_explain)

dhondtxai_values.values          # numpy attribution array
dhondtxai_values.dhondtxai_values # alias for values
dhondtxai_values.base_values     # baselines
dhondtxai_values.scores          # model scores explained by DhondtXAI
dhondtxai_values.feature_names   # feature order
```

For a single row, `dhondtxai_values.values` has shape `(n_features,)`. For a
table, it has shape `(n_rows, n_features)`. Detailed local objects are still
available:

```python
local_explanation = dhondtxai_values[0]
local_explanation.dhondtxai_values
local_explanation.to_feature_frame()
local_explanation.summary()
```

Residual categories such as excluded or below-threshold effects are stored in
`dhondtxai_values.residual_values` so the main value matrix remains aligned with
the original feature columns.

## Method Summary

For a trained model score `g_c(x)`, DhondtXAI computes a baseline

```text
mu_c = E[g_c(X)]
```

and the local model difference

```text
Delta_c(x) = g_c(x) - mu_c
```

For a feature or feature alliance `A`, the method estimates a
background-interventional removal score

```text
R_A^D(x) = (1 / M) sum_m g_c(x_-A, z_A^(m))
```

and the local removal effect

```text
e_A^D(x) = g_c(x) - R_A^D(x)
```

The samples `z_A^(m)` come from the background data provided through `fit(...)`
or `background_data`. This is not a Shapley-value or SHAP computation.

These effects are converted into positive and negative explanatory votes. The
D'Hondt rule allocates explanatory seats separately for supporting and opposing
evidence. Signed source back-projection keeps positive and negative source
effects separate. A final conservative projection maps the seat-based
representation back onto the model difference:

```text
sum_i phi_i^D(x) = g_c(x) - mu_c
```

DhondtXAI therefore produces signed local attributions in a SHAP-like additive
format, but the values should be interpreted as D'Hondt-projected removal-effect
attributions rather than Shapley values.

## Main Features

- SHAP-independent local feature attributions.
- Background-interventional feature removal using a background dataset.
- Optional conditional KNN perturbation for more local replacements.
- Manual, automatic, hybrid, or no feature alliances.
- Same-direction or absolute-interaction affinity for automatic alliances.
- Optional threshold/barrier mechanism.
- Optional redistribution of below-threshold alliance votes.
- Positive and negative D'Hondt evidence parliaments.
- Explicit stable or random D'Hondt tie-breaking.
- Excluded-feature and below-threshold residual reporting.
- Projection residual diagnostics.
- Separate attribution resolution and display seat counts.
- Local and global explanation outputs.
- Global alliance co-occurrence matrix for automatic/hybrid alliances.
- Backward-compatible legacy `feature_importances_` allocation API.

## What DhondtXAI Is Not

DhondtXAI is not SHAP. It does not estimate Shapley values, does not average
marginal contributions over all feature coalitions, and its explanations should
not be interpreted as Shapley values. It is a separate D'Hondt-based attribution
operator that can be compared with SHAP in experiments.

## When To Use

Use DhondtXAI when you want signed local feature attributions, alliance-level
explanations, threshold/barrier analysis, parliamentary representation of model
evidence, and global alliance co-occurrence analysis.

## Local Explanation Example

```python
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from dhondtxai import Explainer, plot_signed_parliament

dataset = load_breast_cancer()
X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
y = pd.Series(dataset.target)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y,
)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)
explainer = Explainer(model, X_train)

explanation = explainer.explain(
    X_test.iloc[0],
    seats=120,
    allocation_seats=5000,
    threshold=0.05,
    redistribute=True,
    alliance_mode="user",
    user_alliances=[
        ["mean concavity", "mean concave points"],
        ["mean radius", "mean perimeter", "mean area"],
    ],
    n_background=50,
    lambda_interaction=0.2,
)

print(explanation.to_feature_frame())
print(explanation.to_alliance_frame())
print(explanation.summary(top_k=5))
print(explanation.diagnostics())
print(explanation.projection_residual_ratio)

explainer.plot_local_bar(explanation, top_k=10)
explainer.plot_waterfall(explanation, top_k=10)
plot_signed_parliament(explanation, mode="signed")
```

## Model Compatibility

DhondtXAI can explain any tabular model that maps input rows to numeric scores.
It does not require model internals, gradients, tree structure, or SHAP values.

Supported model families:

- sklearn estimators and sklearn `Pipeline` objects
- XGBoost sklearn estimators
- native XGBoost `Booster`
- native LightGBM `Booster`
- CatBoost estimators
- PyTorch `nn.Module`
- Keras-like models
- custom scoring functions

The recommended API does not require you to call model-specific scoring methods
yourself. The default `model_adapter="auto"` and `input_format="auto"` select an
input format automatically:

| Model family | Typical object | Automatic input |
|---|---|---|
| sklearn / sklearn Pipeline | `RandomForestClassifier`, `Pipeline`, `XGBClassifier` sklearn API | pandas DataFrame |
| native XGBoost | `xgboost.Booster` | `xgboost.DMatrix` |
| native LightGBM | `lightgbm.Booster` | pandas DataFrame |
| CatBoost | `CatBoostClassifier`, `CatBoostRegressor` | pandas DataFrame |
| PyTorch | `torch.nn.Module` | `torch.float32` tensor |
| Keras-like | Keras-compatible model object | NumPy array |
| custom | row-wise scoring function | controlled by `input_format` or `input_adapter` |

For already-trained models, you can pass the background data directly:

```python
explainer = Explainer(trained_model, X_train)
dhondtxai_values = explainer(X_test)
```

### XGBoost

sklearn-style XGBoost estimators work directly:

```python
from xgboost import XGBClassifier

model = XGBClassifier(...).fit(X_train, y_train)

explainer = Explainer(model, X_train)
dhondtxai_values = explainer(X_test)
```

Native XGBoost `Booster` objects are adapted automatically:

```python
import xgboost as xgb

dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=list(X_train.columns))
booster = xgb.train({"objective": "binary:logistic"}, dtrain)

explainer = Explainer(booster, X_train)
dhondtxai_values = explainer(X_test)
```

For native binary boosters, DhondtXAI uses the numeric score returned by the
booster as the explained model output.

### LightGBM

```python
import lightgbm as lgb

dataset = lgb.Dataset(X_train, label=y_train)
booster = lgb.train({"objective": "binary"}, dataset)

explainer = Explainer(booster, X_train)
dhondtxai_values = explainer(X_test)
```

### CatBoost

```python
from catboost import CatBoostClassifier

model = CatBoostClassifier(verbose=False).fit(X_train, y_train)

explainer = Explainer(model, X_train)
dhondtxai_values = explainer(X_test)
```

### PyTorch

```python
import torch

class Net(torch.nn.Module):
    def forward(self, X):
        return torch.sigmoid(self.linear(X)).squeeze(-1)

model = Net()

explainer = Explainer(model, X_train)
dhondtxai_values = explainer(X_test)
```

DhondtXAI converts tabular rows to `torch.float32` tensors automatically.

### Keras-like Models

```python
explainer = Explainer(
    keras_model,
    X_train,
    model_adapter="keras",
)

dhondtxai_values = explainer(X_test)
```

Keras-like models receive NumPy arrays by default.

For custom, Keras, PyTorch, ONNX, or remote models, pass a callable:

```python
def score_fn(X):
    return my_model_score_function(X)

explainer = Explainer(
    score_fn=score_fn,
    background_data=X_train,
)
```

If your model expects NumPy arrays instead of pandas DataFrames:

```python
explainer = Explainer(
    score_fn=lambda X: keras_model.predict(X, verbose=0)[:, 1],
    background_data=X_train,
    input_format="numpy",
)
```

For more complex conversions, use `input_adapter`:

```python
explainer = Explainer(
    score_fn=score_fn,
    background_data=X_train,
    input_adapter=lambda X: X.to_numpy(dtype="float32"),
)
```

When automatic inference is not enough, override the adapter explicitly:

```python
explainer = Explainer(
    model,
    X_train,
    model_adapter="xgboost",   # sklearn, xgboost, lightgbm, catboost, torch, keras
    input_format="auto",
)
```

For multi-output regression or custom score matrices, select the target column:

```python
explainer = Explainer(
    model,
    X_train,
    target_index=1,
)
```

For multiclass classification, you may explain the predicted class directly:

```python
explanation = explainer.explain(
    X_test.iloc[0],
    class_index="predicted",
)
```

Check compatibility before running a large explanation job:

```python
print(explainer.check_model_compatibility())
```

If no background data has been set yet, pass a sample directly:

```python
print(explainer.check_model_compatibility(X_sample=X_train.head()))
```

## Alliance Modes

`alliance_mode="none"` treats each feature as its own actor.

`alliance_mode="user"` uses only user-defined disjoint alliances and keeps
remaining features as individual actors.

`alliance_mode="auto"` estimates pairwise interaction affinity and forms
automatic alliances. Use `auto_alliance_method="connected_components"` for the
default graph component rule or `auto_alliance_method="complete_linkage"` for a
stricter rule requiring all pairs inside an alliance to meet the affinity
threshold.

`alliance_mode="hybrid"` preserves user-defined alliances and applies automatic
alliance formation only to the remaining features.

## Threshold And Redistribution

Set `threshold=None` or `threshold_enabled=False` to disable the barrier system.

Set `threshold=0.05` to require at least 5 percent of the explanatory vote.

If `redistribute=True`, below-threshold alliance votes are transferred to
eligible alliances according to affinity. If `redistribute=False`, below-threshold
alliances are reported but do not receive seats. Their model contribution is
kept in `__below_threshold__` and `explanation.below_threshold_residual` instead
of being forced into eligible features.

If `exclude_features=[...]` is used, excluded feature influence is not assigned
to the remaining active features. It is reported through `__excluded__` and
`explanation.excluded_residual`.

## Attribution Resolution And Display Seats

`allocation_seats` controls the numerical D'Hondt resolution used to compute
continuous attributions. Larger values reduce integer seat rounding effects.

`seats` controls the visible parliament size used in the plotted explanation.
For example, `allocation_seats=10000` and `seats=100` produces a high-resolution
attribution with a compact 100-seat visualization.

If `allocation_seats` is not provided, DhondtXAI uses:

```text
max(5000, 100 * number_of_active_features, seats)
```

This default prevents small display parliaments such as `seats=10` from
zeroing out meaningful low-rank features in the numerical attribution.

## Perturbation, Affinity, And Tie-Breaking

`perturbation="interventional"` is the default. It replaces the removed feature
or feature group with values sampled from the background data:

```text
g_c(x_-A, z_A)
```

`perturbation="conditional_knn"` uses nearest background rows according to the
non-removed features before taking replacement values. This is still an
approximation, but it reduces unrealistic replacements when correlated tabular
features are present.

For domain-specific replacements, use `perturbation="user_sampler"`:

```python
def sampler(x, group, background, n):
    rows = background.sample(n=n, replace=True, random_state=42).reset_index(drop=True)
    # edit rows[group] here using domain-specific rules
    return rows

explainer = Explainer(
    model,
    X_train,
    perturbation="user_sampler",
    perturbation_sampler=sampler,
)
```

Automatic alliances can use:

- `affinity_mode="same_direction"`: only same-sign single-feature effects can
  form high-affinity alliances.
- `affinity_mode="absolute_interaction"`: pairwise interaction magnitude can
  create affinity even when single-feature effects are weak or opposite.

D'Hondt ties are explicit:

- `tie_break="stable"`: deterministic order-preserving tie-break.
- `tie_break="random"`: seeded random tie-break using `random_state`.

Stable tie-breaking is reproducible but can favor earlier feature/alliance
order in exact ties. Increase `allocation_seats` to reduce the practical impact
of integer ties.

## Outputs

`explanation.to_feature_frame()` returns local feature attributions:

- `feature`
- `attribution`
- `abs_attribution`
- `source_alliance`
- `effect`
- `direction`
- `relative_share`
- `sign_consistent`
- `is_residual`
- residual rows such as `__excluded__` or `__below_threshold__` when relevant

`explanation.to_alliance_frame()` returns alliance-level votes and seats:

- `votes`
- `positive_votes`
- `negative_votes`
- `positive_seats`
- `negative_seats`
- `source_attribution`
- `represented_attribution`
- `source_raw_attribution`
- `represented_raw_attribution`
- threshold status

Diagnostic fields on the explanation object include:

- `raw_attribution_sum`
- `projection_target`
- `projection_residual`
- `projection_residual_ratio`
- `excluded_residual`
- `below_threshold_residual`
- `resolved_output_type`
- `perturbation`
- `affinity_mode`
- `tie_break`

## Global Explanation

```python
explanations = explainer.explain_many(
    X_test.head(50),
    random_state=42,
    reuse_background_sample=False,
    n_background=50,
)

global_frame = explainer.explain_global(
    X_test,
    max_rows=50,
    random_state=42,
    reuse_background_sample=False,
    seats=100,
    alliance_mode="none",
    n_background=50,
)

print(global_frame)
print(explainer.global_alliance_matrix_)

explainer.plot_global_importance(global_frame)
explainer.plot_global_alliance_heatmap()
```

The global output includes:

- `global_abs`: mean absolute DhondtXAI attribution
- `directional`: mean signed attribution
- `positive`: mean positive attribution
- `negative`: mean negative attribution
- `threshold_survival`: frequency of threshold eligibility
- `is_residual`: marks `__excluded__` and `__below_threshold__` rows

By default, global explanations use controlled but different background samples
for each row. Set `reuse_background_sample=True` when you want the same random
background sample reused across all rows.

## Diagnostics And Reports

```python
print(explanation.summary(top_k=8))
print(explanation.summary(top_k=8, language="tr"))
print(explanation.diagnostics())
```

`projection_residual_ratio` should be monitored. A low value means the raw
D'Hondt representation naturally matches the local model difference; a high
value means the conservative projection made a larger correction and the
explanation should be interpreted with more caution.

Interpretation guide:

- `0.00 - 0.10`: low correction; explanation is closer to raw D'Hondt evidence.
- `0.10 - 0.50`: medium correction; interpret with caution.
- `> 0.50`: high correction; raw D'Hondt evidence required a large projection.

The text report prints an explicit warning when this correction is medium or
high.

## Plots

- `plot_local_bar(...)`: signed local feature bar plot.
- `plot_waterfall(...)`: baseline-to-score additive waterfall.
- `plot_signed_parliament(...)`: positive/negative evidence parliament.
- `plot_global_importance(...)`: residual-aware global importance.
- `plot_global_alliance_heatmap(...)`: global alliance co-occurrence matrix.

## Limitations

- DhondtXAI is a beta tabular XAI library, not a mature SHAP/LIME replacement.
- Current removal sampling is background-interventional, not fully conditional.
- `conditional_knn` is an approximate local sampler, not a causal conditional
  distribution estimator.
- `user_sampler` can improve domain realism, but its validity depends on the
  sampler supplied by the user.
- Low `allocation_seats` values can produce sparse or order-sensitive
  attributions; the default uses high-resolution allocation.
- Stable D'Hondt tie-breaking is order-preserving and should be documented when
  exact ties matter.
- Background replacement can create out-of-distribution rows for strongly
  correlated features.
- Auto-alliance and interaction estimation can be expensive for high-dimensional data.
- Projection residuals should be monitored and reported.
- Probability-scale explanations may be less additive than logit-scale explanations.
- Below-threshold and excluded residuals should be interpreted explicitly.

## Building A PyPI-Ready Package

The package metadata includes README long description, version, license,
citation, and optional sklearn/dev extras. To build local distribution files:

```bash
python -m pip install --upgrade build twine
python -m build
python -m twine check dist/*
```

Then a built wheel can be installed locally with:

```bash
pip install dist/dhondtxai-0.9.2-py3-none-any.whl
```

Publishing to PyPI requires a PyPI API token:

```bash
python -m twine upload dist/*
```

## Legacy API

The previous global feature-importance workflow is still available:

```python
features, votes, excluded = explainer.apply_dhondt(
    num_votes=100000000,
    num_mps=600,
    threshold=5,
)

seats = explainer.dhondt_method(votes, 600, excluded)
explainer.plot_results(features, seats)
```

This legacy path uses `model.feature_importances_`. The new proposed method is
`explain(...)`.

## Citation

T. B. Donmez, "Explainable AI through a Democratic Lens: DhondtXAI for
Proportional Feature Importance Using the D'Hondt Method," 2024.
https://doi.org/10.48550/arXiv.2411.05196
