import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import matplotlib

matplotlib.use("Agg")

import dhondtxai as dxai
from dhondtxai import DhondtValues, DhondtXAI, Explainer, __version__


class AddModel:
    def predict(self, X):
        return np.asarray(X).sum(axis=1)


class ProbaModel:
    classes_ = np.array([0, 1])

    def predict_proba(self, X):
        values = np.clip(np.asarray(X).sum(axis=1) / 10.0, 0.0, 1.0)
        return np.column_stack([1.0 - values, values])


class ThreeClassProbaModel:
    classes_ = np.array(["low", "mid", "high"])

    def predict_proba(self, X):
        arr = np.asarray(X)
        score = arr.sum(axis=1)
        p0 = np.clip(0.6 - 0.1 * score, 0.05, 0.9)
        p2 = np.clip(0.1 + 0.2 * score, 0.05, 0.9)
        p1 = np.maximum(1.0 - p0 - p2, 0.05)
        probs = np.column_stack([p0, p1, p2])
        return probs / probs.sum(axis=1, keepdims=True)


class BinaryDecisionModel:
    classes_ = np.array([0, 1])

    def decision_function(self, X):
        return np.asarray(X).sum(axis=1)


class MultiOutputModel:
    def predict(self, X):
        arr = np.asarray(X)
        return np.column_stack([arr[:, 0], arr[:, 0] + 2 * arr[:, 1]])


class SingleColumn2DModel:
    def predict(self, X):
        return np.asarray(X).sum(axis=1).reshape(-1, 1)


class CategoricalProbaModel:
    classes_ = np.array([0, 1])

    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        gender = np.asarray(X["gender"] == "M", dtype=float)
        age = np.asarray(X["age"], dtype=float) / 100.0
        values = np.clip(0.2 + 0.2 * gender + age, 0.0, 1.0)
        return np.column_stack([1.0 - values, values])


class LabelModel:
    def predict(self, X):
        return np.array(["yes"] * len(X))


class NaNModel:
    def predict(self, X):
        values = np.asarray(X, dtype=float).sum(axis=1)
        values[0] = np.nan
        return values


class ProductModel:
    def predict(self, X):
        arr = np.asarray(X, dtype=float)
        return arr[:, 0] * arr[:, 1]


class KerasLikeModel:
    def predict(self, X, verbose=0):
        return np.asarray(X).sum(axis=1)


class SingleColumnProbabilityModel:
    def predict(self, X, verbose=0):
        probabilities = np.clip(np.asarray(X).sum(axis=1) / 10.0, 0.0, 1.0)
        return probabilities.reshape(-1, 1)


class InvalidProbabilityModel:
    def predict_proba(self, X):
        values = np.asarray(X).sum(axis=1) + 2.0
        return np.column_stack([1.0 - values, values])


def make_add_explainer():
    background = pd.DataFrame({"a": [0.0] * 20, "b": [0.0] * 20, "c": [0.0] * 20})
    return DhondtXAI(AddModel(), background_data=background, output_type="prediction")


def make_adapter_data():
    X = pd.DataFrame(
        {
            "a": [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 1.5, -1.5],
            "b": [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, -0.5, 0.5],
        }
    )
    y = ((X["a"] + X["b"]) > 0).astype(int)
    return X, y


def test_no_shap_dependency_loaded():
    assert "shap" not in sys.modules


def test_local_completeness():
    explainer = make_add_explainer()
    explanation = explainer.explain(
        pd.Series({"a": 10.0, "b": 3.0, "c": 1.0}),
        seats=20,
        allocation_seats=1000,
        n_background=5,
    )
    assert abs(sum(explanation.feature_attributions.values()) - explanation.delta) < 1e-9
    assert explanation.diagnostics()["completeness_error"] < 1e-9


def test_linear_additive_recovery_with_zero_background():
    explainer = make_add_explainer()
    explanation = explainer.explain(
        pd.Series({"a": 10.0, "b": 3.0, "c": 1.0}),
        seats=20,
        allocation_seats=10000,
        n_background=5,
    )
    assert explanation.feature_attributions["a"] == pytest.approx(10.0, abs=0.02)
    assert explanation.feature_attributions["b"] == pytest.approx(3.0, abs=0.02)
    assert explanation.feature_attributions["c"] == pytest.approx(1.0, abs=0.02)


def test_excluded_residual_does_not_leak_to_active_features():
    explainer = make_add_explainer()
    explanation = explainer.explain(
        pd.Series({"a": 10.0, "b": 0.0, "c": 0.0}),
        exclude_features=["a"],
        seats=20,
        allocation_seats=1000,
        n_background=5,
    )
    assert explanation.feature_attributions["b"] == pytest.approx(0.0)
    assert explanation.feature_attributions["c"] == pytest.approx(0.0)
    assert explanation.feature_attributions["__excluded__"] == pytest.approx(10.0)


def test_below_threshold_residual_when_redistribution_disabled():
    explainer = make_add_explainer()
    explanation = explainer.explain(
        pd.Series({"a": 10.0, "b": -1.0, "c": 0.0}),
        threshold=0.5,
        redistribute=False,
        seats=20,
        allocation_seats=1000,
        n_background=5,
    )
    assert explanation.feature_attributions["a"] == pytest.approx(10.0)
    assert explanation.feature_attributions["b"] == pytest.approx(0.0)
    assert explanation.feature_attributions["__below_threshold__"] == pytest.approx(-1.0)


def test_signed_redistribution_preserves_negative_source():
    explainer = make_add_explainer()
    explanation = explainer.explain(
        pd.Series({"a": 10.0, "b": -1.0, "c": 0.0}),
        threshold=0.5,
        redistribute=True,
        seats=20,
        allocation_seats=1000,
        n_background=5,
    )
    assert explanation.feature_attributions["a"] > 0
    assert explanation.feature_attributions["b"] < 0
    assert abs(sum(explanation.feature_attributions.values()) - explanation.delta) < 1e-9


def test_opposite_sign_members_in_user_alliance_keep_signs():
    explainer = make_add_explainer()
    explanation = explainer.explain(
        pd.Series({"a": 10.0, "b": -1.0, "c": 0.0}),
        alliance_mode="user",
        user_alliances=[["a", "b"]],
        seats=20,
        allocation_seats=1000,
        n_background=5,
    )
    assert explanation.feature_attributions["a"] > 0
    assert explanation.feature_attributions["b"] < 0
    assert explanation.diagnostics()["mixed_sign_alliance_count"] == 1
    assert "both supporting and opposing" in explanation.summary()


def test_plain_summary_is_shorter_and_less_technical_than_standard_summary():
    explainer = make_add_explainer()
    explanation = explainer.explain(pd.Series({"a": 1.0, "b": -2.0, "c": 0.0}), n_background=5)
    plain = explanation.summary(style="plain")
    standard = explanation.summary(style="standard")
    assert "DhondtXAI explanation" in plain
    assert "Diagnostics:" not in plain
    assert len(plain.splitlines()) < len(standard.splitlines())


def test_local_explanation_values_keep_original_feature_order_with_user_alliances():
    background = pd.DataFrame({"a": [0.0] * 5, "b": [0.0] * 5, "c": [0.0] * 5})
    explainer = DhondtXAI(AddModel(), background_data=background, output_type="prediction")

    explanation = explainer.explain(
        pd.Series({"a": 1.0, "b": 2.0, "c": 3.0}),
        alliance_mode="user",
        user_alliances=[["b", "a"]],
        n_background=5,
        allocation_seats=1000,
    )

    assert explanation.feature_names == ["a", "b", "c"]
    assert explanation.values.tolist() == pytest.approx([1.0, 2.0, 3.0])


def test_global_output_keeps_residual_rows():
    explainer = make_add_explainer()
    X = pd.DataFrame(
        [
            {"a": 10.0, "b": -1.0, "c": 0.0},
            {"a": 8.0, "b": -1.0, "c": 0.0},
        ]
    )
    global_frame = explainer.explain_global(
        X,
        threshold=0.5,
        redistribute=False,
        seats=20,
        allocation_seats=1000,
        n_background=5,
    )
    residual = global_frame[global_frame["feature"] == "__below_threshold__"].iloc[0]
    assert residual["is_residual"]
    assert residual["global_abs"] == pytest.approx(1.0)
    assert np.isnan(residual["threshold_survival"])


def test_to_shap_includes_residuals_by_default(monkeypatch):
    class FakeExplanation:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    monkeypatch.setitem(sys.modules, "shap", types.SimpleNamespace(Explanation=FakeExplanation))
    explainer = make_add_explainer()
    explanation = explainer.explain(
        pd.Series({"a": 1.0, "b": 2.0, "c": 0.0}),
        exclude_features=["c"],
        n_background=5,
    )
    shap_exp = explanation.to_shap()
    assert "__excluded__" in shap_exp.feature_names
    assert float(shap_exp.base_values + np.sum(shap_exp.values)) == pytest.approx(explanation.score)


def test_values_to_shap_includes_residuals_by_default(monkeypatch):
    class FakeExplanation:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    monkeypatch.setitem(sys.modules, "shap", types.SimpleNamespace(Explanation=FakeExplanation))
    explainer = make_add_explainer()
    values = explainer.dhondtxai_values(
        pd.Series({"a": 1.0, "b": 2.0, "c": 0.0}),
        exclude_features=["c"],
        n_background=5,
    )
    shap_exp = values.to_shap()
    assert "__excluded__" in shap_exp.feature_names
    assert float(shap_exp.base_values + np.sum(shap_exp.values)) == pytest.approx(values.scores)


def test_invalid_class_index_raises():
    background = pd.DataFrame({"a": [0.0, 1.0], "b": [0.0, 1.0]})
    explainer = DhondtXAI(ProbaModel(), background_data=background, output_type="probability")
    with pytest.raises(ValueError, match="class_index"):
        explainer.explain(pd.Series({"a": 1.0, "b": 1.0}), class_index=5)


def test_predicted_class_index_supported_for_multiclass_probability():
    background = pd.DataFrame({"a": [0.0, 0.0], "b": [0.0, 0.0]})
    explainer = DhondtXAI(ThreeClassProbaModel(), background_data=background, output_type="probability")
    explanation = explainer.explain(pd.Series({"a": 2.0, "b": 2.0}), class_index="predicted", n_background=2)
    assert explanation.class_index == 2
    assert explanation.class_label == "high"
    assert "class=high" in explanation.summary()


def test_unknown_feature_raises():
    explainer = make_add_explainer()
    with pytest.raises(ValueError, match="Unknown feature"):
        explainer.explain(
            pd.Series({"a": 1.0, "b": 1.0, "c": 1.0}),
            alliance_mode="user",
            user_alliances=[["a", "typo"]],
        )


def test_callable_predict_fn_supported():
    background = pd.DataFrame({"a": [0.0, 0.0], "b": [0.0, 0.0]})

    def predict_fn(X):
        return np.asarray(X).sum(axis=1)

    explainer = DhondtXAI(predict_fn=predict_fn, background_data=background, output_type="custom")
    explanation = explainer.explain(pd.Series({"a": 1.0, "b": 2.0}), allocation_seats=1000)
    assert explanation.delta == pytest.approx(3.0)
    assert abs(sum(explanation.feature_attributions.values()) - explanation.delta) < 1e-9


def test_direct_callable_model_supported():
    background = pd.DataFrame({"a": [0.0, 0.0], "b": [0.0, 0.0]})
    explainer = DhondtXAI(lambda X: np.asarray(X).sum(axis=1), background_data=background, output_type="custom")
    explanation = explainer.explain(pd.Series({"a": 1.0, "b": 2.0}), allocation_seats=1000)
    assert explanation.delta == pytest.approx(3.0)


def test_numpy_input_format_supported():
    background = pd.DataFrame({"a": [0.0, 0.0], "b": [0.0, 0.0]})
    explainer = DhondtXAI(
        predict_fn=lambda X: X.sum(axis=1),
        background_data=background,
        output_type="custom",
        input_format="numpy",
    )
    explanation = explainer.explain(pd.Series({"a": 1.0, "b": 2.0}), allocation_seats=1000)
    assert explanation.delta == pytest.approx(3.0)


def test_binary_decision_class_zero_is_negative_margin():
    background = pd.DataFrame({"a": [0.0, 0.0], "b": [0.0, 0.0]})
    explainer = DhondtXAI(BinaryDecisionModel(), background_data=background, output_type="decision")
    x = pd.Series({"a": 1.0, "b": 2.0})
    explanation_one = explainer.explain(x, class_index=1)
    explanation_zero = explainer.explain(x, class_index=0)
    assert explanation_zero.score == pytest.approx(-explanation_one.score)
    assert explanation_zero.delta == pytest.approx(-explanation_one.delta)


def test_probability_and_logit_scales_keep_positive_direction_for_simple_case():
    background = pd.DataFrame({"a": [0.0, 0.0], "b": [0.0, 0.0]})
    x = pd.Series({"a": 1.0, "b": 2.0})
    probability = DhondtXAI(ProbaModel(), background_data=background, output_type="probability")
    logit = DhondtXAI(ProbaModel(), background_data=background, output_type="logit")
    probability_explanation = probability.explain(x, n_background=2)
    logit_explanation = logit.explain(x, n_background=2)
    assert probability_explanation.delta > 0
    assert logit_explanation.delta > 0
    assert probability_explanation.feature_attributions["a"] > 0
    assert logit_explanation.feature_attributions["a"] > 0


def test_multioutput_prediction_target_index():
    background = pd.DataFrame({"a": [0.0, 0.0], "b": [0.0, 0.0]})
    explainer = DhondtXAI(MultiOutputModel(), background_data=background, output_type="prediction", target_index=1)
    explanation = explainer.explain(pd.Series({"a": 1.0, "b": 2.0}), allocation_seats=1000)
    assert explanation.score == pytest.approx(5.0)
    assert explanation.delta == pytest.approx(5.0)


def test_regression_task_maps_integer_target_to_target_index_with_auto_output():
    background = pd.DataFrame({"a": [0.0, 0.0], "b": [0.0, 0.0]})
    explainer = DhondtXAI(MultiOutputModel(), background_data=background, task="regression", target=1)
    explanation = explainer.explain(pd.Series({"a": 1.0, "b": 2.0}), allocation_seats=1000)
    assert explanation.target_index == 1
    assert explanation.class_index is None
    assert explanation.score == pytest.approx(5.0)


def test_auto_prediction_integer_target_maps_to_target_index_for_multioutput_model():
    background = pd.DataFrame({"a": [0.0, 0.0], "b": [0.0, 0.0]})
    explainer = DhondtXAI(MultiOutputModel(), background_data=background, target=1)
    explanation = explainer.explain(pd.Series({"a": 1.0, "b": 2.0}), allocation_seats=1000)
    assert explanation.resolved_output_type == "prediction"
    assert explanation.target_index == 1
    assert explanation.class_index is None
    assert explanation.score == pytest.approx(5.0)


def test_single_column_2d_prediction_defaults_to_first_target():
    background = pd.DataFrame({"a": [0.0, 0.0], "b": [0.0, 0.0]})
    explainer = DhondtXAI(SingleColumn2DModel(), background_data=background, output_type="prediction")
    explanation = explainer.explain(pd.Series({"a": 1.0, "b": 2.0}), n_background=2)
    assert explanation.score == pytest.approx(3.0)
    assert explanation.delta == pytest.approx(3.0)
    assert "prediction" in explanation.summary()
    assert "class_index" not in explanation.summary()


def test_default_allocation_resolution_keeps_equal_features_nonzero():
    feature_count = 20
    background = pd.DataFrame({f"f{i}": [0.0] * 5 for i in range(feature_count)})
    x = pd.Series({f"f{i}": 1.0 for i in range(feature_count)})
    explainer = DhondtXAI(AddModel(), background_data=background, output_type="prediction")
    explanation = explainer.explain(x, seats=10, n_background=5)
    values = [explanation.feature_attributions[f"f{i}"] for i in range(feature_count)]
    assert explanation.allocation_seat_count >= 5000
    assert min(values) > 0.0
    assert max(values) - min(values) < 1e-9


def test_integer_column_names_are_treated_as_names_before_positions():
    background = pd.DataFrame({0: [0.0, 0.0], 1: [0.0, 0.0]})
    explainer = DhondtXAI(AddModel(), background_data=background, output_type="prediction")
    explanation = explainer.explain(pd.Series({0: 10.0, 1: 0.0}), exclude_features=[1], n_background=2)
    assert explanation.feature_attributions[0] == pytest.approx(10.0)
    assert explanation.feature_attributions["__excluded__"] == pytest.approx(0.0)
    frame = explanation.to_alliance_frame()
    assert not frame.empty


def test_numpy_background_feature_names_work_in_alliance_frame():
    explainer = DhondtXAI(AddModel(), background_data=np.zeros((3, 2)), output_type="prediction")
    explanation = explainer.explain([1.0, 2.0], alliance_mode="user", user_alliances=[[0, 1]], n_background=3)
    frame = explanation.to_alliance_frame()
    assert frame.iloc[0]["members"] == "0, 1"
    assert "0 + 1" in set(map(str, frame["alliance"]))


def test_mixed_feature_names_do_not_break_pair_keys_or_auto_alliance():
    background = pd.DataFrame({0: [0.0, 0.0, 0.0], "a": [0.0, 0.0, 0.0]})
    explainer = DhondtXAI(AddModel(), background_data=background, output_type="prediction")
    explanation = explainer.explain(
        pd.Series({0: 1.0, "a": 2.0}),
        alliance_mode="auto",
        affinity_mode="absolute_interaction",
        lambda_interaction=0.1,
        n_background=3,
    )
    assert abs(sum(explanation.feature_attributions.values()) - explanation.delta) < 1e-9


def test_reserved_feature_display_names_raise():
    with pytest.raises(ValueError, match="reserved"):
        DhondtXAI(AddModel(), background_data=pd.DataFrame({"a + b": [0.0]}), output_type="prediction")
    with pytest.raises(ValueError, match="reserved"):
        DhondtXAI(AddModel(), background_data=pd.DataFrame({"__excluded__": [0.0]}), output_type="prediction")


def test_duplicate_feature_names_raise_clear_error():
    background = pd.DataFrame(np.zeros((2, 2)), columns=["a", "a"])
    with pytest.raises(ValueError, match="unique"):
        DhondtXAI(AddModel(), background_data=background, output_type="prediction")


def test_nan_model_output_raises_clear_error():
    background = pd.DataFrame({"a": [0.0, 1.0], "b": [0.0, 1.0]})
    explainer = DhondtXAI(NaNModel(), background_data=background, output_type="prediction")
    with pytest.raises(ValueError, match="NaN or infinite"):
        explainer.explain(pd.Series({"a": 1.0, "b": 1.0}))


def test_class_index_none_binary_probability_is_normalized():
    background = pd.DataFrame({"a": [0.0, 0.0], "b": [0.0, 0.0]})
    explainer = DhondtXAI(ProbaModel(), background_data=background, output_type="probability", class_index=None)
    explanation = explainer.explain(pd.Series({"a": 1.0, "b": 2.0}), n_background=2)
    assert explanation.class_index == 1
    assert explanation.class_label == 1
    assert np.isfinite(explanation.score)


def test_dhondt_rejects_negative_and_nonfinite_votes():
    background = pd.DataFrame({"a": [0.0], "b": [0.0]})
    explainer = DhondtXAI(AddModel(), background_data=background, output_type="prediction")
    with pytest.raises(ValueError, match="non-negative"):
        explainer.dhondt_method([1.0, -1.0], 10)
    with pytest.raises(ValueError, match="finite"):
        explainer.dhondt_method([1.0, np.nan], 10)


def test_dhondt_scale_invariance_for_tiny_votes():
    background = pd.DataFrame({"a": [0.0], "b": [0.0], "c": [0.0]})
    explainer = DhondtXAI(AddModel(), background_data=background, output_type="prediction")
    base = explainer.dhondt_method([1.0, 2.0, 5.0], 50)
    tiny = explainer.dhondt_method([1e-13, 2e-13, 5e-13], 50)
    assert tiny.tolist() == base.tolist()


def test_conditional_knn_perturbation_uses_local_neighbors():
    background = pd.DataFrame(
        {
            "a": [100.0, 100.0, 0.0, 0.0],
            "c": [10.0, 11.0, 0.0, 0.1],
        }
    )
    x = pd.Series({"a": 5.0, "c": 0.05})
    explainer = DhondtXAI(AddModel(), background_data=background, output_type="prediction", knn_neighbors=2)

    interventional = explainer.explain(x, n_background=4, perturbation="interventional", random_state=0)
    conditional = explainer.explain(x, n_background=4, perturbation="conditional_knn", random_state=0)

    assert conditional.effects["a"] > interventional.effects["a"]


def test_user_sampler_perturbation_supported():
    background = pd.DataFrame({"a": [0.0, 0.0], "b": [0.0, 0.0]})

    def sampler(x, group, background_sample, n):
        rows = pd.DataFrame([x.to_dict()] * n)
        for feature in group:
            rows[feature] = 0.0
        return rows

    explainer = DhondtXAI(
        AddModel(),
        background_data=background,
        output_type="prediction",
        perturbation="user_sampler",
        perturbation_sampler=sampler,
    )
    explanation = explainer.explain(pd.Series({"a": 1.0, "b": 2.0}), n_background=2)
    assert explanation.delta == pytest.approx(3.0)
    assert explanation.feature_attributions["a"] > 0


def test_absolute_interaction_affinity_can_capture_pure_interaction():
    background = pd.DataFrame({"a": [0.0], "b": [0.0]})
    explainer = DhondtXAI(AddModel(), background_data=background, output_type="prediction")
    affinity = explainer._build_affinity(
        ["a", "b"],
        {"a": 0.0, "b": 0.0},
        {("a", "b"): 1.0},
        {("a", "b"): 1.0},
        "absolute_interaction",
    )
    assert affinity[("a", "b")] > 0.9


def test_projection_auto_uses_residual_bucket_when_raw_evidence_is_zero():
    background = pd.DataFrame({"a": [0.0, 2.0], "b": [2.0, 0.0]})
    explainer = DhondtXAI(ProductModel(), background_data=background, output_type="prediction")
    explanation = explainer.explain(
        pd.Series({"a": 1.0, "b": 1.0}),
        n_background=2,
        random_state=0,
        projection_mode="auto",
    )
    assert explanation.feature_attributions["a"] == pytest.approx(0.0)
    assert explanation.feature_attributions["b"] == pytest.approx(0.0)
    assert explanation.feature_attributions["__projection_residual__"] == pytest.approx(explanation.delta)
    assert explanation.projection_residual_ratio == pytest.approx(1.0)


def test_projection_auto_uses_residual_bucket_when_ratio_is_high():
    explainer = make_add_explainer()
    projected, raw_sum, residual, ratio, bucket = explainer._project_values(
        {"a": 0.90, "b": 0.0},
        target=1.0,
        names=["a", "b"],
        mode="auto",
        residual_name="__projection_residual__",
        residual_threshold=0.05,
    )
    assert projected == {"a": pytest.approx(0.90), "b": pytest.approx(0.0)}
    assert raw_sum == pytest.approx(0.90)
    assert residual == pytest.approx(0.10)
    assert ratio == pytest.approx(0.10)
    assert bucket == pytest.approx(0.10)


def test_strict_exclusion_keeps_interaction_with_excluded_in_residual():
    background = pd.DataFrame({"a": [0.0, 0.0], "b": [0.0, 0.0]})
    explainer = DhondtXAI(ProductModel(), background_data=background, output_type="prediction")
    explanation = explainer.explain(
        pd.Series({"a": 1.0, "b": 1.0}),
        exclude_features=["a"],
        n_background=2,
        exclude_mode="strict",
    )
    assert explanation.feature_attributions["b"] == pytest.approx(0.0)
    assert explanation.feature_attributions["__excluded__"] == pytest.approx(1.0)


def test_strict_threshold_keeps_interaction_with_below_threshold_in_residual():
    background = pd.DataFrame({"a": [0.0, 0.0], "b": [0.0, 0.0]})
    explainer = DhondtXAI(ProductModel(), background_data=background, output_type="prediction")
    explanation = explainer.explain(
        pd.Series({"a": 1.0, "b": 1.0}),
        threshold=0.99,
        redistribute=False,
        n_background=2,
        threshold_mode="strict_residual",
    )
    assert explanation.feature_attributions["__below_threshold__"] == pytest.approx(1.0)
    feature_mass = explanation.feature_attributions["a"] + explanation.feature_attributions["b"]
    assert feature_mass == pytest.approx(1.0)
    assert explanation.feature_attributions["__projection_residual__"] == pytest.approx(-1.0)
    assert sum(explanation.feature_attributions.values()) == pytest.approx(explanation.delta)


def test_dhondt_tie_break_modes_are_explicit():
    background = pd.DataFrame({"a": [0.0], "b": [0.0]})
    explainer = DhondtXAI(AddModel(), background_data=background, output_type="prediction", random_state=1)
    stable = explainer.dhondt_method([1.0, 1.0], 1, tie_break="stable")
    random = explainer.dhondt_method([1.0, 1.0], 1, tie_break="random")
    assert stable.tolist() == [1, 0]
    assert random.sum() == 1


def test_explain_global_varies_row_random_states_when_requested():
    explainer = make_add_explainer()
    X = pd.DataFrame(
        [
            {"a": 1.0, "b": 0.0, "c": 0.0},
            {"a": 2.0, "b": 0.0, "c": 0.0},
            {"a": 3.0, "b": 0.0, "c": 0.0},
        ]
    )
    explainer.explain_global(X, random_state=123, reuse_background_sample=False, n_background=5)
    assert len(set(explainer.global_random_states_)) == len(X)

    explainer.explain_global(X, random_state=123, reuse_background_sample=True, n_background=5)
    assert len(set(explainer.global_random_states_)) == 1


def test_categorical_fit_correlation_does_not_fail():
    X = pd.DataFrame({"age": [20, 50, 40, 30], "gender": ["F", "M", "M", "F"]})
    y = pd.Series([0, 1, 1, 0])
    explainer = DhondtXAI(CategoricalProbaModel(), output_type="probability")
    explainer.fit(X, y)
    explanation = explainer.explain(X.iloc[0], n_background=2)
    assert isinstance(explanation.score, float)
    assert explainer.correlation_info["gender"] == 0.0


def test_sklearn_categorical_pipeline_with_preprocessing():
    sklearn = pytest.importorskip("sklearn")
    assert sklearn is not None
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    X = pd.DataFrame(
        {
            "age": [20, 50, 45, 25, 60, 30],
            "gender": ["F", "M", "M", "F", "M", "F"],
        }
    )
    y = pd.Series([0, 1, 1, 0, 1, 0])
    preprocess = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["gender"]),
            ("num", "passthrough", ["age"]),
        ]
    )
    model = Pipeline(
        [
            ("preprocess", preprocess),
            ("clf", LogisticRegression(max_iter=200)),
        ]
    )
    explainer = DhondtXAI(model, output_type="probability")
    explainer.fit(X, y)
    explanation = explainer.explain(X.iloc[0], n_background=3)
    assert isinstance(explanation.score, float)
    assert explanation.resolved_output_type == "probability"


def test_non_numeric_prediction_error_is_clear():
    background = pd.DataFrame({"a": [0.0, 1.0]})
    explainer = DhondtXAI(LabelModel(), background_data=background, output_type="prediction")
    with pytest.raises(ValueError, match="numeric"):
        explainer.explain(pd.Series({"a": 1.0}))


def test_compatibility_checker_reports_success():
    background = pd.DataFrame({"a": [0.0, 0.0], "b": [0.0, 0.0]})
    explainer = DhondtXAI(
        predict_fn=lambda X: np.asarray(X).sum(axis=1),
        background_data=background,
        output_type="custom",
    )
    report = explainer.check_model_compatibility()
    assert report["compatible"] is True
    assert report["numeric"] is True
    assert report["resolved_output_type"] == "custom"
    assert report["model_adapter"] == "callable"


def test_compatibility_checker_works_with_x_sample_without_background():
    X_sample = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    explainer = DhondtXAI(AddModel(), output_type="prediction")
    report = explainer.check_model_compatibility(X_sample=X_sample, deep=True)
    assert report["compatible"] is True
    assert report["selected_output_shape"] == (2,)
    assert report["deep_check"] == "Full mini explanation path succeeded."
    assert explainer.features is None


def test_package_version_exposed():
    assert __version__ == "0.9.5.6"


def test_public_strings_do_not_use_non_ascii_locale_characters():
    forbidden = {chr(code) for code in (305, 287, 252, 351, 246, 231, 304, 286, 220, 350, 214, 199)}
    root = Path(__file__).resolve().parents[1]
    paths = [
        *Path(root, "dhondtxai").rglob("*.py"),
        Path(root, "README.md"),
        Path(root, "setup.py"),
        Path(root, "CITATION.cff"),
        Path(root, "LICENSE"),
    ]
    for path in paths:
        text = path.read_text(encoding="utf-8").lower()
        assert not any(char in text for char in forbidden), str(path)


def test_shap_like_explainer_values_api():
    background = pd.DataFrame({"a": [0.0] * 10, "b": [0.0] * 10, "c": [0.0] * 10})
    explainer = Explainer(AddModel(), background)
    dhondtxai_values = explainer(
        pd.DataFrame([{"a": 1.0, "b": 2.0, "c": 3.0}]),
        n_background=5,
        allocation_seats=1000,
    )
    assert isinstance(dhondtxai_values, DhondtValues)
    assert dhondtxai_values.values.shape == (1, 3)
    assert dhondtxai_values.dhondtxai_values.shape == (1, 3)
    assert dhondtxai_values.feature_names == ["a", "b", "c"]
    assert dhondtxai_values.base_values.shape == (1,)
    assert dhondtxai_values[0].dhondtxai_values.shape == (3,)
    assert dhondtxai_values[0].base_value == pytest.approx(dhondtxai_values.base_values[0])


def test_independent_masker_can_supply_background_data():
    import dhondtxai as dxai

    background = pd.DataFrame({"a": [0.0] * 5, "b": [0.0] * 5})
    masker = dxai.maskers.Independent(background)
    explainer = dxai.Explainer(AddModel(), masker=masker, output_type="prediction")
    explanation = explainer.explain(pd.Series({"a": 1.0, "b": 2.0}), n_background=5)
    assert explanation.delta == pytest.approx(3.0)


def test_masker_max_samples_controls_default_background_rows():
    import dhondtxai as dxai

    background = pd.DataFrame({"a": [0.0] * 10, "b": [0.0] * 10})
    masker = dxai.maskers.Independent(background, max_samples=3)
    explainer = dxai.Explainer(AddModel(), masker=masker, output_type="prediction")
    explanation = explainer.explain(pd.Series({"a": 1.0, "b": 2.0}))
    assert explanation.diagnostics()["cost"]["n_background"] == 3


def test_cost_mode_fast_screens_auto_interaction_pairs():
    feature_count = 30
    background = pd.DataFrame({f"f{i}": [0.0] * 5 for i in range(feature_count)})
    x = pd.Series({f"f{i}": 1.0 for i in range(feature_count)})
    explainer = DhondtXAI(AddModel(), background_data=background, output_type="prediction")
    explanation = explainer.explain(
        x,
        alliance_mode="auto",
        affinity_mode="absolute_interaction",
        cost_mode="fast",
    )
    cost = explanation.diagnostics()["cost"]
    assert cost["resolved_cost_mode"] == "fast"
    assert cost["interaction_pairs_used"] <= 200
    assert cost["interaction_pairs_used"] < cost["interaction_pairs_total"]


def test_user_alliance_interactions_only_evaluate_required_pairs():
    background = pd.DataFrame({name: [0.0] * 5 for name in ["a", "b", "c", "d", "e"]})
    explainer = DhondtXAI(AddModel(), background_data=background, output_type="prediction")
    explanation = explainer.explain(
        pd.Series({"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0, "e": 5.0}),
        alliance_mode="user",
        user_alliances=[["a", "b"]],
        lambda_interaction=0.2,
    )
    cost = explanation.diagnostics()["cost"]
    assert cost["interaction_pairs_used"] == 1
    assert set(explanation.interactions.keys()) == {("a", "b")}


def test_shap_like_single_row_values_are_1d():
    explainer = make_add_explainer()
    dhondtxai_values = explainer.dhondtxai_values(
        pd.Series({"a": 1.0, "b": 2.0, "c": 3.0}),
        n_background=5,
        allocation_seats=1000,
    )
    assert dhondtxai_values.values.shape == (3,)
    assert dhondtxai_values.deltas == pytest.approx(6.0)
    assert np.sum(dhondtxai_values.values) == pytest.approx(dhondtxai_values.deltas)


def test_score_fn_alias_for_custom_models():
    background = pd.DataFrame({"a": [0.0] * 10, "b": [0.0] * 10})

    def score_fn(X):
        return np.asarray(X).sum(axis=1)

    explainer = Explainer.from_score_function(score_fn=score_fn, background_data=background)
    dhondtxai_values = explainer(pd.Series({"a": 1.0, "b": 2.0}), n_background=5)
    assert dhondtxai_values.values.shape == (2,)
    assert dhondtxai_values.deltas == pytest.approx(3.0)


def test_custom_2d_score_respects_class_index():
    background = pd.DataFrame({"a": [0.0] * 5, "b": [0.0] * 5})

    def score_fn(X):
        arr = np.asarray(X)
        return np.column_stack([arr[:, 0], arr[:, 0] + 10.0 * arr[:, 1]])

    x = pd.Series({"a": 0.1, "b": 1.0})
    explainer = Explainer(score_fn=score_fn, background_data=background, class_index=1)
    explanation = explainer.explain(x, n_background=5)
    assert explanation.score == pytest.approx(10.1)


def test_custom_2d_score_respects_predicted_class_index():
    background = pd.DataFrame({"a": [0.0] * 5, "b": [0.0] * 5})

    def score_fn(X):
        arr = np.asarray(X)
        return np.column_stack([arr[:, 0], arr[:, 0] + 10.0 * arr[:, 1]])

    x = pd.Series({"a": 0.1, "b": 1.0})
    explainer = Explainer(score_fn=score_fn, background_data=background, class_index="predicted")
    explanation = explainer.explain(x, n_background=5)
    assert explanation.class_index == 1
    assert explanation.score == pytest.approx(10.1)


def test_single_column_probability_class_zero_is_complement():
    background = pd.DataFrame({"a": [0.0] * 5, "b": [0.0] * 5})
    explainer = DhondtXAI(
        SingleColumnProbabilityModel(),
        background_data=background,
        output_type="probability",
        model_adapter="keras",
    )
    x = pd.Series({"a": 3.0, "b": 2.0})
    explanation_one = explainer.explain(x, class_index=1, n_background=5)
    explanation_zero = explainer.explain(x, class_index=0, n_background=5)
    assert explanation_one.score == pytest.approx(0.5)
    assert explanation_zero.score == pytest.approx(0.5)
    assert explanation_zero.delta == pytest.approx(-explanation_one.delta)


def test_keras_like_classification_auto_resolves_probability():
    background = pd.DataFrame({"a": [0.0] * 5, "b": [0.0] * 5})
    explainer = DhondtXAI(
        SingleColumnProbabilityModel(),
        background_data=background,
        task="classification",
        output_type="auto",
        model_adapter="keras",
    )
    explanation = explainer.explain(pd.Series({"a": 2.0, "b": 1.0}), n_background=5)
    assert explanation.resolved_output_type == "probability"
    assert explanation.score == pytest.approx(0.3)


def test_baseline_mode_sample_removes_sampling_projection_noise_for_additive_model():
    class WeightedAdd:
        def predict(self, X):
            arr = np.asarray(X, dtype=float)
            return 2 * arr[:, 0] + 3 * arr[:, 1]

    background = pd.DataFrame({"a": np.arange(100, dtype=float), "b": np.arange(100, dtype=float) * 2})
    row = pd.Series({"a": 100.0, "b": 200.0})
    explainer = DhondtXAI(WeightedAdd(), background_data=background, output_type="prediction", random_state=0)
    full = explainer.explain(row, n_background=1, allocation_seats=10000, baseline_mode="full")
    sample = explainer.explain(row, n_background=1, allocation_seats=10000, baseline_mode="sample")
    assert full.projection_residual_ratio > 0.1
    assert sample.projection_residual_ratio < 1e-8
    assert sample.diagnostics()["baseline_mode"] == "sample"
    assert sample.full_baseline != pytest.approx(sample.sample_baseline)


def test_lambda_alliance_vote_does_not_inflate_value_mass():
    background = pd.DataFrame({"a": [0.0] * 10, "b": [0.0] * 10})
    explainer = DhondtXAI(ProductModel(), background_data=background, output_type="prediction")
    row = pd.Series({"a": 2.0, "b": 3.0})
    no_bonus = explainer.explain(
        row,
        alliance_mode="user",
        user_alliances=[["a", "b"]],
        lambda_alliance_vote=0.0,
        lambda_member_split=0.0,
        n_background=5,
        allocation_seats=1000,
    )
    bonus = explainer.explain(
        row,
        alliance_mode="user",
        user_alliances=[["a", "b"]],
        lambda_alliance_vote=10.0,
        lambda_member_split=0.0,
        n_background=5,
        allocation_seats=1000,
    )
    assert sum(no_bonus.value_masses.values()) == pytest.approx(sum(bonus.value_masses.values()))
    assert sum(no_bonus.positive_value_masses.values()) == pytest.approx(sum(bonus.positive_value_masses.values()))


def test_top_level_explain_api_returns_local_explanation():
    background = pd.DataFrame({"a": [0.0] * 5, "b": [0.0] * 5})
    explanation = dxai.explain(
        AddModel(),
        X_background=background,
        X=pd.Series({"a": 1.0, "b": 2.0}),
        output_type="prediction",
        n_background=5,
    )
    assert explanation.values.tolist() == pytest.approx([1.0, 2.0], abs=1e-3)
    assert explanation.plot(kind="bar", show=False)[0] is not None


def test_class_label_target_maps_to_class_index():
    background = pd.DataFrame({"a": [0.0, 1.0], "b": [0.0, 1.0]})
    explainer = DhondtXAI(ThreeClassProbaModel(), background_data=background, target="high")
    explanation = explainer.explain(pd.Series({"a": 1.0, "b": 1.0}), n_background=2)
    assert explanation.class_index == 2
    assert explanation.class_label == "high"


def test_multiclass_default_target_warns():
    background = pd.DataFrame({"a": [0.0, 1.0], "b": [0.0, 1.0]})
    explainer = DhondtXAI(ThreeClassProbaModel(), background_data=background)
    with pytest.warns(UserWarning, match="Multiclass output detected"):
        explainer.explain(pd.Series({"a": 1.0, "b": 1.0}), n_background=2)


def test_output_adapter_and_predict_kwargs_are_applied():
    class KwargModel:
        def predict(self, X, scale=1.0):
            return np.asarray(X).sum(axis=1) * scale

    background = pd.DataFrame({"a": [0.0, 0.0], "b": [0.0, 0.0]})
    explainer = DhondtXAI(
        KwargModel(),
        background_data=background,
        output_type="prediction",
        predict_kwargs={"scale": 2.0},
        output_adapter=lambda values: np.asarray(values) + 1.0,
    )
    explanation = explainer.explain(pd.Series({"a": 1.0, "b": 2.0}), n_background=2)
    assert explanation.score == pytest.approx(7.0)


def test_invalid_output_type_raises_at_init():
    with pytest.raises(ValueError, match="output_type"):
        DhondtXAI(AddModel(), background_data=pd.DataFrame({"a": [0.0]}), output_type="labels")


def test_empty_global_input_raises_clear_error():
    explainer = make_add_explainer()
    with pytest.raises(ValueError, match="at least one row"):
        explainer.explain_global(pd.DataFrame(columns=["a", "b", "c"]))


def test_max_model_rows_caps_screened_interactions():
    columns = [f"f{i}" for i in range(20)]
    background = pd.DataFrame(np.zeros((30, len(columns))), columns=columns)
    row = pd.Series(np.ones(len(columns)), index=columns)
    explainer = DhondtXAI(AddModel(), background_data=background, output_type="prediction")
    explanation = explainer.explain(
        row,
        alliance_mode="auto",
        affinity_mode="absolute_interaction",
        lambda_interaction=0.1,
        cost_mode="research",
        n_background=2,
        max_model_rows=100,
    )
    assert explanation.diagnostics()["cost"]["estimated_model_rows"] <= 100


def test_friendly_residual_labels_appear_on_bar_plot():
    explainer = make_add_explainer()
    explanation = explainer.explain(
        pd.Series({"a": 10.0, "b": 0.0, "c": 0.0}),
        exclude_features=["a"],
        n_background=5,
    )
    _, ax = explainer.plot_local_bar(explanation, show=False)
    labels = [tick.get_text() for tick in ax.get_yticklabels()]
    assert "excluded-feature effect" in labels


def test_negative_plot_parliament_seats_raise():
    from dhondtxai import plot_parliament

    with pytest.raises(ValueError, match="non-negative"):
        plot_parliament(10, ["a"], [-1], show=False)


def test_one_dimensional_probability_class_zero_is_complement():
    background = pd.DataFrame({"a": [0.0] * 5, "b": [0.0] * 5})

    def score_fn(X):
        return np.clip(np.asarray(X).sum(axis=1) / 10.0, 0.0, 1.0)

    explainer = Explainer(score_fn=score_fn, background_data=background, output_type="probability")
    x = pd.Series({"a": 2.0, "b": 1.0})
    explanation_one = explainer.explain(x, class_index=1, n_background=5)
    explanation_zero = explainer.explain(x, class_index=0, n_background=5)
    assert explanation_one.score == pytest.approx(0.3)
    assert explanation_zero.score == pytest.approx(0.7)


def test_probability_output_outside_unit_interval_raises():
    background = pd.DataFrame({"a": [0.0, 0.0], "b": [0.0, 0.0]})
    explainer = DhondtXAI(InvalidProbabilityModel(), background_data=background, output_type="probability")
    with pytest.raises(ValueError, match="outside \\[0, 1\\]"):
        explainer.explain(pd.Series({"a": 1.0, "b": 1.0}), n_background=2)


def test_probability_validation_can_be_disabled_for_custom_advanced_use():
    background = pd.DataFrame({"a": [0.0, 0.0], "b": [0.0, 0.0]})
    explainer = DhondtXAI(
        InvalidProbabilityModel(),
        background_data=background,
        output_type="probability",
        validate_probability=False,
    )
    explanation = explainer.explain(pd.Series({"a": 1.0, "b": 1.0}), n_background=2)
    assert explanation.score == pytest.approx(4.0)


def test_xgboost_sklearn_classifier_auto_adapter_if_installed():
    xgb = pytest.importorskip("xgboost")
    X, y = make_adapter_data()
    model = xgb.XGBClassifier(
        n_estimators=5,
        max_depth=2,
        learning_rate=0.5,
        eval_metric="logloss",
        random_state=0,
        n_jobs=1,
    )
    model.fit(X, y)
    explainer = DhondtXAI(model, background_data=X, output_type="auto")
    explanation = explainer.explain(X.iloc[0], n_background=4)
    assert explanation.resolved_output_type == "probability"
    assert np.isfinite(explanation.score)


def test_xgboost_native_booster_auto_adapter_if_installed():
    xgb = pytest.importorskip("xgboost")
    X, y = make_adapter_data()
    dtrain = xgb.DMatrix(X.to_numpy(), label=y.to_numpy(), feature_names=list(X.columns))
    booster = xgb.train(
        {"objective": "binary:logistic", "eval_metric": "logloss", "verbosity": 0, "nthread": 1},
        dtrain,
        num_boost_round=5,
    )
    explainer = DhondtXAI(booster, background_data=X, output_type="prediction")
    report = explainer.check_model_compatibility()
    explanation = explainer.explain(X.iloc[0], n_background=4)
    assert report["model_adapter"] == "xgboost"
    assert 0.0 <= explanation.score <= 1.0


def test_lightgbm_native_booster_auto_adapter_if_installed():
    lgb = pytest.importorskip("lightgbm")
    X, y = make_adapter_data()
    dataset = lgb.Dataset(X, label=y)
    booster = lgb.train(
        {"objective": "binary", "metric": "binary_logloss", "verbosity": -1, "seed": 0},
        dataset,
        num_boost_round=5,
    )
    explainer = DhondtXAI(booster, background_data=X, output_type="prediction")
    report = explainer.check_model_compatibility()
    explanation = explainer.explain(X.iloc[0], n_background=4)
    assert report["model_adapter"] == "lightgbm"
    assert 0.0 <= explanation.score <= 1.0


def test_catboost_classifier_auto_adapter_if_installed():
    cb = pytest.importorskip("catboost")
    X, y = make_adapter_data()
    model = cb.CatBoostClassifier(iterations=5, depth=2, learning_rate=0.5, verbose=False, random_seed=0)
    model.fit(X, y)
    explainer = DhondtXAI(model, background_data=X, output_type="auto")
    report = explainer.check_model_compatibility()
    explanation = explainer.explain(X.iloc[0], n_background=4)
    assert report["model_adapter"] == "catboost"
    assert explanation.resolved_output_type == "probability"


def test_torch_module_auto_adapter_if_installed():
    torch = pytest.importorskip("torch")
    X, _ = make_adapter_data()

    class TorchModel(torch.nn.Module):
        def forward(self, X_tensor):
            return torch.sigmoid(X_tensor.sum(dim=1))

    explainer = DhondtXAI(TorchModel(), background_data=X, output_type="prediction")
    report = explainer.check_model_compatibility()
    explanation = explainer.explain(X.iloc[0], n_background=4)
    assert report["model_adapter"] == "torch"
    assert 0.0 <= explanation.score <= 1.0


def test_keras_like_model_adapter():
    X, _ = make_adapter_data()
    explainer = DhondtXAI(
        KerasLikeModel(),
        background_data=X,
        output_type="prediction",
        model_adapter="keras",
    )
    report = explainer.check_model_compatibility()
    explanation = explainer.explain(X.iloc[0], n_background=4)
    assert report["model_adapter"] == "keras"
    assert np.isfinite(explanation.score)


def test_plot_functions_smoke():
    explainer = make_add_explainer()
    explanation = explainer.explain(pd.Series({"a": 1.0, "b": 2.0, "c": 0.0}), n_background=5)
    fig1, ax1 = explainer.plot_local_bar(explanation, show=False)
    fig2, ax2 = explainer.plot_waterfall(explanation, show=False)
    global_frame = explainer.explain_global(pd.DataFrame([{"a": 1.0, "b": 2.0, "c": 0.0}]), n_background=5)
    fig3, ax3 = explainer.plot_global_importance(global_frame, show=False)
    assert fig1 is not None and ax1 is not None
    assert fig2 is not None and ax2 is not None
    assert fig3 is not None and ax3 is not None


def test_alliance_plot_top_k_is_applied():
    explainer = make_add_explainer()
    explanation = explainer.explain(pd.Series({"a": 1.0, "b": 2.0, "c": 3.0}), n_background=5)
    fig, ax = explainer.plot_explanation(explanation, level="alliance", top_k=2, show=False)
    assert fig is not None
    assert len(ax.patches) == 2


def test_waterfall_warns_when_residuals_are_hidden():
    explainer = make_add_explainer()
    explanation = explainer.explain(
        pd.Series({"a": 10.0, "b": 0.0, "c": 0.0}),
        exclude_features=["a"],
        n_background=5,
    )
    fig, ax = explainer.plot_waterfall(explanation, include_residuals=False, show=False)
    assert fig is not None
    assert any("residuals hidden" in text.get_text() for text in ax.texts)


def test_waterfall_other_features_is_not_labeled_as_residual():
    explainer = make_add_explainer()
    explanation = explainer.explain(pd.Series({"a": 1.0, "b": 2.0, "c": 3.0}), n_background=5)
    _, ax = explainer.plot_waterfall(explanation, top_k=1, show=False)
    tick_labels = "\n".join(label.get_text() for label in ax.get_xticklabels())
    assert "other\nfeatures" in tick_labels or "other features" in tick_labels
    assert "Other features" in " ".join(text.get_text() for text in ax.texts)


def test_signed_parliament_smoke():
    from dhondtxai import plot_signed_parliament

    explainer = make_add_explainer()
    explanation = explainer.explain(pd.Series({"a": 1.0, "b": -2.0, "c": 0.0}), n_background=5)
    fig, ax = plot_signed_parliament(explanation, mode="signed", palette="signed", show=False)
    assert fig is not None and ax is not None


def test_non_english_language_is_not_supported():
    explainer = make_add_explainer()
    explanation = explainer.explain(pd.Series({"a": 1.0, "b": -2.0, "c": 0.0}), n_background=5)
    with pytest.raises(ValueError, match="Only English"):
        explanation.summary(language="xx")


def test_signed_parliament_snaps_awkward_seat_count():
    from dhondtxai import plot_signed_parliament

    explainer = make_add_explainer()
    explanation = explainer.explain(
        pd.Series({"a": 10.0, "b": 3.0, "c": 1.0}),
        seats=257,
        n_background=5,
    )
    fig, ax = plot_signed_parliament(explanation, mode="signed", show=False)
    assert fig is not None
    assert "visualized as 250 seats" in ax.get_title()
