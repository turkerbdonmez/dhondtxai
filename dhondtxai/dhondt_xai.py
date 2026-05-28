from dataclasses import dataclass
import heapq
from itertools import combinations
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


SUPPORTED_LANGUAGES = {"en"}
SUPPORTED_PRESETS = {"fast", "balanced", "accurate", "research"}
SUPPORTED_COST_MODES = {"auto", "fast", "balanced", "accurate", "research"}
SUPPORTED_OUTPUT_TYPES = {"auto", "probability", "logit", "log_odds", "decision", "prediction", "custom"}
SUPPORTED_BASELINE_MODES = {"full", "sample", "auto"}

RESIDUAL_LABELS_EN = {
    "__projection_residual__": "projection correction",
    "__below_threshold__": "below-threshold evidence",
    "__excluded__": "excluded-feature effect",
}


@dataclass(frozen=True)
class CostPolicy:
    name: str
    n_background: int
    allocation_seats: int
    max_interaction_pairs: object
    top_k_interaction_features: object
    max_model_rows: object
    global_max_rows: object
    interaction_screening: str


COST_POLICIES = {
    "fast": CostPolicy("fast", 25, 1000, 200, 20, 50_000, 100, "top_effects"),
    "balanced": CostPolicy("balanced", 100, 5000, 1000, 40, 250_000, 500, "top_effects"),
    "accurate": CostPolicy("accurate", 250, 10000, 5000, 80, 1_000_000, 2000, "top_effects"),
    "research": CostPolicy("research", 500, 20000, None, None, None, None, "all"),
}


def _validate_language(language):
    if language not in SUPPORTED_LANGUAGES:
        raise ValueError("Only English output is supported.")


def _residual_label(name, language="en"):
    return RESIDUAL_LABELS_EN.get(str(name), str(name))


@dataclass
class DhondtExplanation:
    """Container for one local DhondtXAI explanation."""

    score: float
    baseline: float
    delta: float
    active_delta: float
    feature_attributions: dict
    feature_order: list
    represented_alliance_attributions: dict
    source_alliance_attributions: dict
    represented_raw_attributions: dict
    source_raw_attributions: dict
    represented_positive_raw: dict
    represented_negative_raw: dict
    positive_seats: dict
    negative_seats: dict
    allocation_positive_seats: dict
    allocation_negative_seats: dict
    votes: dict
    positive_votes: dict
    negative_votes: dict
    value_masses: dict
    positive_value_masses: dict
    negative_value_masses: dict
    effects: dict
    interactions: dict
    alliance_members: dict
    alliance_sign_conflicts: dict
    feature_source_alliance: dict
    eligible_alliances: list
    below_threshold_alliances: list
    excluded_features: list
    excluded_residual: float
    below_threshold_residual: float
    raw_attribution_sum: float
    projection_target: float
    projection_residual: float
    projection_residual_ratio: float
    threshold: float
    redistribution: bool
    seat_count: int
    allocation_seat_count: int
    class_index: int
    target_index: int
    class_label: object
    output_type: str
    resolved_output_type: str
    perturbation: str
    affinity_mode: str
    tie_break: str
    projection_mode: str
    projection_residual_attribution: float
    projection_residual_threshold: float
    exclude_mode: str
    threshold_mode: str
    baseline_mode: str
    full_baseline: float
    sample_baseline: float
    baseline_sampling_gap: float
    baseline_sampling_gap_ratio: float
    data: object = None
    output_names: object = None
    cost_diagnostics: object = None

    @property
    def feature_names(self):
        order = self.feature_order or list(self.feature_attributions.keys())
        return [
            feature
            for feature in order
            if feature in self.feature_attributions and not str(feature).startswith("__")
        ]

    @property
    def values(self):
        return np.asarray(
            [self.feature_attributions.get(feature, 0.0) for feature in self.feature_names],
            dtype=float,
        )

    @property
    def dhondtxai_values(self):
        return self.values

    @property
    def base_value(self):
        return self.baseline

    @property
    def residual_values(self):
        return {
            feature: value
            for feature, value in self.feature_attributions.items()
            if str(feature).startswith("__")
        }

    def to_feature_frame(self, top_k=None, include_residuals=True):
        rows = []
        for feature, value in self.feature_attributions.items():
            is_residual = str(feature).startswith("__")
            if is_residual and not include_residuals:
                continue
            effect = self.effects.get(feature, np.nan)
            attribution_sign = np.sign(value)
            effect_sign = np.sign(effect) if not pd.isna(effect) else np.nan
            if is_residual or pd.isna(effect_sign) or effect_sign == 0 or attribution_sign == 0:
                sign_consistent = np.nan if is_residual else True
            else:
                sign_consistent = bool(effect_sign == attribution_sign)
            rows.append(
                {
                    "feature": feature,
                    "attribution": value,
                    "abs_attribution": abs(value),
                    "direction": self._direction_label(value),
                    "source_alliance": self.feature_source_alliance.get(feature),
                    "effect": effect,
                    "effect_sign": effect_sign,
                    "attribution_sign": attribution_sign,
                    "sign_consistent": sign_consistent,
                    "is_residual": is_residual,
                }
            )
        frame = pd.DataFrame(rows)
        if frame.empty:
            return frame
        frame = frame.sort_values("abs_attribution", ascending=False).reset_index(drop=True)
        total_abs = frame["abs_attribution"].sum()
        frame["relative_share"] = 0.0 if total_abs == 0 else frame["abs_attribution"] / total_abs
        frame.insert(0, "rank", np.arange(1, len(frame) + 1))
        if top_k is not None:
            frame = frame.head(top_k).reset_index(drop=True)
        return frame

    def to_alliance_frame(self, view="source"):
        if view not in {"source", "represented"}:
            raise ValueError("view must be 'source' or 'represented'.")

        rows = []
        all_names = list(self.alliance_members.keys())
        for name in all_names:
            pos_seats = self.positive_seats.get(name, 0)
            neg_seats = self.negative_seats.get(name, 0)
            attribution = (
                self.source_alliance_attributions.get(name, 0.0)
                if view == "source"
                else self.represented_alliance_attributions.get(name, 0.0)
            )
            rows.append(
                {
                    "alliance": name,
                    "members": ", ".join(map(str, self.alliance_members[name])),
                    "attribution": attribution,
                    "direction": self._direction_label(attribution),
                    "source_attribution": self.source_alliance_attributions.get(name, 0.0),
                    "represented_attribution": self.represented_alliance_attributions.get(name, 0.0),
                    "source_raw_attribution": self.source_raw_attributions.get(name, 0.0),
                    "represented_raw_attribution": self.represented_raw_attributions.get(name, 0.0),
                    "votes": self.votes.get(name, 0.0),
                    "positive_votes": self.positive_votes.get(name, 0.0),
                    "negative_votes": self.negative_votes.get(name, 0.0),
                    "value_mass": self.value_masses.get(name, 0.0),
                    "positive_value_mass": self.positive_value_masses.get(name, 0.0),
                    "negative_value_mass": self.negative_value_masses.get(name, 0.0),
                    "positive_seats": pos_seats,
                    "negative_seats": neg_seats,
                    "allocation_positive_seats": self.allocation_positive_seats.get(name, 0),
                    "allocation_negative_seats": self.allocation_negative_seats.get(name, 0),
                    "total_seats": pos_seats + neg_seats,
                    "eligible": name in self.eligible_alliances,
                    "below_threshold": name in self.below_threshold_alliances,
                }
            )
        frame = pd.DataFrame(rows)
        if frame.empty:
            return frame
        return frame.sort_values(["eligible", "total_seats", "votes"], ascending=False).reset_index(drop=True)

    def diagnostics(self):
        values = np.asarray(list(self.feature_attributions.values()), dtype=float)
        completeness_error = abs(float(values.sum()) - self.delta)
        denominator = abs(self.delta) + 1e-12
        excluded_ratio = abs(self.excluded_residual) / denominator
        below_threshold_ratio = abs(self.below_threshold_residual) / denominator
        projection_bucket_ratio = abs(self.projection_residual_attribution) / denominator
        projection_ratio_vs_delta = abs(self.projection_residual) / denominator
        raw_abs_sum = sum(abs(value) for value in self.source_raw_attributions.values())
        cancellation_ratio = raw_abs_sum / (abs(self.raw_attribution_sum) + 1e-12)
        sign_flip_count = int(
            self.to_feature_frame(include_residuals=False)["sign_consistent"].eq(False).sum()
        )
        mixed_sign_alliance_count = int(sum(bool(value) for value in self.alliance_sign_conflicts.values()))
        if self.projection_residual_ratio < 0.10:
            quality = "high"
        elif self.projection_residual_ratio < 0.50:
            quality = "medium"
        else:
            quality = "caution"

        return {
            "completeness_error": completeness_error,
            "projection_residual_ratio": self.projection_residual_ratio,
            "projection_residual_ratio_vs_delta": projection_ratio_vs_delta,
            "projection_residual": self.projection_residual,
            "raw_attribution_abs_sum": raw_abs_sum,
            "cancellation_ratio": cancellation_ratio,
            "baseline_mode": self.baseline_mode,
            "full_baseline": self.full_baseline,
            "sample_baseline": self.sample_baseline,
            "baseline_sampling_gap": self.baseline_sampling_gap,
            "baseline_sampling_gap_ratio": self.baseline_sampling_gap_ratio,
            "excluded_residual_ratio": excluded_ratio,
            "below_threshold_residual_ratio": below_threshold_ratio,
            "projection_residual_attribution_ratio": projection_bucket_ratio,
            "sign_flip_count": sign_flip_count,
            "mixed_sign_alliance_count": mixed_sign_alliance_count,
            "quality": quality,
            "cost": self.cost_diagnostics or {},
        }

    def _target_label(self):
        if self.resolved_output_type in {"probability", "logit", "decision"}:
            return f"class={self.class_label}" if self.class_label is not None else f"class_index={self.class_index}"
        if self.target_index is None:
            return "prediction"
        return f"target_index={self.target_index}"

    @property
    def pairwise_overlaps(self):
        return self.interactions

    def _export_values(self, include_residuals=True):
        feature_names = list(self.feature_names)
        values = [self.feature_attributions.get(feature, 0.0) for feature in feature_names]
        data = None if self.data is None else np.asarray(self.data, dtype=object)
        if include_residuals:
            for name, value in self.residual_values.items():
                if name not in feature_names:
                    feature_names.append(name)
                    values.append(value)
                    if data is not None:
                        data = np.append(data, np.nan)
        elif self.residual_values:
            warnings.warn(
                "Residual attributions are omitted from the exported values. "
                "Use include_residuals=True to preserve local completeness.",
                UserWarning,
                stacklevel=2,
            )
        return np.asarray(values, dtype=float), feature_names, data

    def to_shap(self, include_residuals=True):
        import shap

        values, feature_names, data = self._export_values(include_residuals=include_residuals)
        return shap.Explanation(
            values=values,
            base_values=self.base_value,
            data=data,
            feature_names=feature_names,
            output_names=self.output_names,
        )

    def _projection_warning(self, language="en"):
        _validate_language(language)
        if self.projection_residual_ratio < 0.10:
            return None
        if self.projection_residual_ratio < 0.50:
            level = "medium"
            detail = "Interpret attributions cautiously."
        else:
            level = "high"
            detail = (
                "Raw D'Hondt representation may weakly match the model difference; "
                "attributions may be driven substantially by projection correction."
            )
        return (
            f"Warning: Projection correction is {level}. "
            f"{detail}"
        )

    def summary(self, top_k=5, language="en", style="standard"):
        _validate_language(language)
        if style not in {"standard", "plain"}:
            raise ValueError("style must be 'standard' or 'plain'.")

        frame = self.to_feature_frame(include_residuals=True)
        residual_frame = frame[frame["is_residual"]].copy()
        frame = frame[~frame["is_residual"]].copy()
        supporting = frame[frame["attribution"] > 0].head(top_k)
        opposing = frame[frame["attribution"] < 0].head(top_k)
        diagnostics = self.diagnostics()
        if style == "plain":
            lines = [
                "DhondtXAI explanation",
                f"Model score: {self.score:.6f}",
                f"Reference baseline: {self.baseline:.6f}",
                f"The selected target score is {self.delta:+.6f} relative to the baseline.",
                "",
                "Largest increases:",
            ]
            lines.extend(self._format_ranked_lines(supporting, empty_text="None"))
            lines.append("")
            lines.append("Largest decreases:")
            lines.extend(self._format_ranked_lines(opposing, empty_text="None"))
            if not residual_frame.empty:
                residual_abs = residual_frame["abs_attribution"].sum()
                total_abs = residual_abs + frame["abs_attribution"].sum()
                share = 0.0 if total_abs <= 0 else residual_abs / total_abs
                lines.extend(
                    [
                        "",
                        f"Note: {share:.0%} of displayed attribution mass is residual/correction, "
                        "not an input feature.",
                    ]
                )
            if diagnostics["projection_residual_ratio"] >= 0.10:
                lines.append(
                    f"Projection correction ratio is {diagnostics['projection_residual_ratio']:.0%}; "
                    "interpret the explanation with caution."
                )
            return "\n".join(lines)
        target = self._target_label()
        output_label = self.output_type
        if self.resolved_output_type != self.output_type:
            output_label = f"{self.output_type} resolved as {self.resolved_output_type}"

        lines = [
            "DhondtXAI Local Explanation Report",
            "----------------------------------",
            f"Explained target: {target}, output_type={output_label}",
            f"Model score: {self.score:.6f}",
            f"Baseline: {self.baseline:.6f}",
            f"Difference: {self.delta:.6f}",
            "",
            f"Main interpretation: the model score is {abs(self.delta):.6f} "
            f"{'higher' if self.delta >= 0 else 'lower'} than the baseline.",
            "",
            "Top supporting features:",
        ]
        lines.extend(self._format_ranked_lines(supporting, empty_text="None"))
        lines.append("")
        lines.append("Top opposing features:")
        lines.extend(self._format_ranked_lines(opposing, empty_text="None"))
        lines.extend(
            [
                "",
                "Diagnostics:",
                f"Completeness error: {diagnostics['completeness_error']:.6g}",
                f"Projection residual ratio: {diagnostics['projection_residual_ratio']:.6f}",
                f"Projection residual ratio vs delta: {diagnostics['projection_residual_ratio_vs_delta']:.6f}",
                f"Cancellation ratio: {diagnostics['cancellation_ratio']:.6f}",
                f"Projection residual: {diagnostics['projection_residual']:.6g}",
                f"Projection residual bucket: {self.projection_residual_attribution:.6g}",
                f"Projection residual threshold: {self.projection_residual_threshold:.6f}",
                f"Baseline mode: {self.baseline_mode}",
                f"Full baseline: {self.full_baseline:.6f}",
                f"Sample baseline: {self.sample_baseline:.6f}",
                f"Baseline sampling gap: {self.baseline_sampling_gap:.6g}",
                f"Threshold: {self.threshold if self.threshold is not None else 'disabled'}",
                f"Redistribution: {'enabled' if self.redistribution else 'disabled'}",
                f"Excluded residual: {self.excluded_residual:.6g}",
                f"Below-threshold residual: {self.below_threshold_residual:.6g}",
                f"Sign flip count: {diagnostics['sign_flip_count']}",
                f"Mixed-sign alliance count: {diagnostics['mixed_sign_alliance_count']}",
                f"Quality: {diagnostics['quality']}",
            ]
        )
        if not residual_frame.empty:
            lines.append("")
            lines.append("Residual and correction terms (not input features):")
            lines.extend(self._format_ranked_lines(residual_frame, empty_text="None", language=language))
        warning = self._projection_warning(language)
        if warning is not None:
            lines.extend(["", warning])
        cost = diagnostics.get("cost") or {}
        if cost.get("resolved_cost_mode") in {"fast", "balanced"} and cost.get("approximation_notes"):
            lines.extend(["", "Runtime note: " + " ".join(cost["approximation_notes"])])
        if diagnostics["mixed_sign_alliance_count"] > 0:
            lines.extend(
                [
                    "",
                    "Warning: Some alliances contain both supporting and opposing member effects. "
                    "Inspect the feature-level table before interpreting alliance seats.",
                ]
            )
        return "\n".join(lines)

    def report(self, top_k=5, language="en", style="standard"):
        return self.summary(top_k=top_k, language=language, style=style)

    def plot(self, kind="bar", **kwargs):
        """Plot this explanation without manually keeping the explainer object."""
        if kind == "parliament":
            from .plot_parliament import plot_signed_parliament

            return plot_signed_parliament(self, **kwargs)
        feature_order = list(self.feature_order or self.feature_names)
        background = pd.DataFrame([{feature: 0.0 for feature in feature_order}])
        helper = DhondtXAI(
            model=lambda X: np.zeros(len(X)),
            background_data=background,
            output_type="prediction",
        )
        helper.last_explanation = self
        if kind in {"bar", "local_bar", "local"}:
            return helper.plot_local_bar(self, **kwargs)
        if kind == "waterfall":
            return helper.plot_waterfall(self, **kwargs)
        raise ValueError("kind must be 'bar', 'waterfall', or 'parliament'.")

    def _direction_label(self, value):
        if value > 0:
            return "increases selected target score"
        if value < 0:
            return "decreases selected target score"
        return "neutral"

    def _format_ranked_lines(self, frame, empty_text, language="en"):
        if frame.empty:
            return [f"- {empty_text}"]
        return [
            f"{idx}. {_residual_label(row.feature, language) if str(row.feature).startswith('__') else row.feature}: {row.attribution:+.6f}"
            for idx, row in enumerate(frame.itertuples(index=False), start=1)
        ]


class DhondtValues:
    """DhondtXAI attribution values container.

    The numerical attributions live in ``values`` and ``dhondtxai_values``.
    Residual categories such as ``__excluded__`` are kept separately in
    ``residual_values`` so the feature matrix stays aligned with the original
    model inputs.
    """

    def __init__(self, explanations, feature_names, data=None, single_output=False):
        self.explanations = list(explanations)
        self.feature_names = list(feature_names)
        self.single_output = bool(single_output)
        matrix = np.asarray(
            [
                [exp.feature_attributions.get(feature, 0.0) for feature in self.feature_names]
                for exp in self.explanations
            ],
            dtype=float,
        )
        base_values = np.asarray([exp.baseline for exp in self.explanations], dtype=float)
        scores = np.asarray([exp.score for exp in self.explanations], dtype=float)
        deltas = np.asarray([exp.delta for exp in self.explanations], dtype=float)
        residual_values = [exp.residual_values for exp in self.explanations]

        self.values = matrix[0] if self.single_output else matrix
        self.dhondtxai_values = self.values
        self.base_values = float(base_values[0]) if self.single_output else base_values
        self.scores = float(scores[0]) if self.single_output else scores
        self.deltas = float(deltas[0]) if self.single_output else deltas
        self.residual_values = residual_values[0] if self.single_output else residual_values
        self.data = data

    @property
    def base_value(self):
        if self.single_output:
            return self.base_values
        return self.base_values[0] if len(self.explanations) == 1 else self.base_values

    @property
    def shape(self):
        return np.asarray(self.values).shape

    @property
    def abs(self):
        return np.abs(self.values)

    def mean(self, axis=0):
        return np.mean(self.values, axis=axis)

    def sum(self, axis=0):
        return np.sum(self.values, axis=axis)

    def _export_matrix(self, include_residuals=True):
        feature_names = list(self.feature_names)
        rows = []
        residual_keys = []
        if include_residuals:
            for residuals in (self.residual_values if not self.single_output else [self.residual_values]):
                for key in residuals:
                    if key not in residual_keys:
                        residual_keys.append(key)
            feature_names += residual_keys
        elif any((self.residual_values if self.single_output else [r for r in self.residual_values])):
            warnings.warn(
                "Residual attributions are omitted from the exported values. "
                "Use include_residuals=True to preserve local completeness.",
                UserWarning,
                stacklevel=2,
            )

        for explanation in self.explanations:
            rows.append([explanation.feature_attributions.get(name, 0.0) for name in feature_names])
        matrix = np.asarray(rows, dtype=float)
        data = self.data
        if include_residuals and residual_keys and data is not None:
            data_array = np.asarray(data, dtype=object)
            if data_array.ndim == 1:
                data_array = data_array.reshape(1, -1)
            residual_data = np.full((data_array.shape[0], len(residual_keys)), np.nan, dtype=object)
            data = np.concatenate([data_array, residual_data], axis=1)
            if self.single_output:
                data = data[0]
        if self.single_output:
            matrix = matrix[0]
        return matrix, feature_names, data

    def to_shap(self, include_residuals=True):
        import shap

        values, feature_names, data = self._export_matrix(include_residuals=include_residuals)
        return shap.Explanation(
            values=values,
            base_values=self.base_values,
            data=data,
            feature_names=feature_names,
        )

    def __array__(self, dtype=None):
        return np.asarray(self.values, dtype=dtype)

    def __len__(self):
        return len(self.explanations)

    def __getitem__(self, index):
        return self.explanations[index]

    def to_frame(self, row=0, top_k=None, include_residuals=True):
        return self.explanations[row].to_feature_frame(top_k=top_k, include_residuals=include_residuals)

    def summary(self, row=0, top_k=5, language="en", style="standard"):
        return self.explanations[row].summary(top_k=top_k, language=language, style=style)

    def diagnostics(self, row=0):
        return self.explanations[row].diagnostics()


class DhondtXAI:
    """D'Hondt-based local attribution without using SHAP values.

    The model explanation is built from background-interventional removal
    effects, optional feature alliances, thresholding, positive/negative
    D'Hondt seat allocation, signed source back-projection, and a conservative
    projection onto the local model difference.
    """

    def __init__(
        self,
        model=None,
        background_data=None,
        *,
        predict_fn=None,
        score_fn=None,
        task="auto",
        output=None,
        output_type="auto",
        target="auto",
        class_index=1,
        target_index=None,
        feature_names=None,
        masker=None,
        input_format="auto",
        input_adapter=None,
        output_adapter=None,
        predict_kwargs=None,
        validate_probability=True,
        probability_tolerance=1e-6,
        model_adapter="auto",
        perturbation="interventional",
        perturbation_sampler=None,
        knn_neighbors=25,
        affinity_mode="same_direction",
        tie_break="stable",
        projection_mode="auto",
        projection_residual_threshold=0.10,
        exclude_mode="strict",
        threshold_mode="strict_residual",
        baseline_mode="full",
        feature_reference="auto",
        cost_mode="auto",
        preset=None,
        max_model_rows=None,
        max_interaction_pairs=None,
        top_k_interaction_features=None,
        interaction_screening="top_effects",
        global_max_rows=None,
        eps=1e-12,
        random_state=42,
        strict_features=True,
    ):
        if output is not None:
            output_type = output
        if output_type == "log_odds":
            output_type = "logit"
        if preset is not None:
            if cost_mode not in {None, "auto", preset}:
                raise ValueError("Use either cost_mode or preset when they differ.")
            cost_mode = preset
        if cost_mode is None:
            cost_mode = "auto"
        integer_target = None
        if target not in (None, "auto"):
            if target == "predicted":
                class_index = "predicted"
            elif isinstance(target, (int, np.integer)):
                integer_target = int(target)
                target_as_output_index = (
                    task in {"regression", "custom"}
                    or output_type in {"prediction", "custom"}
                    or (output_type == "auto" and task in {"regression", "custom"})
                )
                if target_as_output_index:
                    target_index = int(target)
                    class_index = None
                else:
                    class_index = int(target)
            elif isinstance(target, str):
                class_index = target
            else:
                raise ValueError("target must be 'auto', 'predicted', an integer index, or a class label.")

        if score_fn is not None:
            if predict_fn is not None:
                raise ValueError("Use either score_fn or predict_fn, not both.")
            predict_fn = score_fn

        torch_like_model = model is not None and hasattr(model, "forward") and hasattr(model, "eval")
        if predict_fn is None and callable(model) and not torch_like_model and not any(
            hasattr(model, attr) for attr in ("predict", "predict_proba", "decision_function")
        ):
            predict_fn = model
            model = None

        if masker is not None:
            if background_data is None and hasattr(masker, "background_data"):
                background_data = masker.background_data
            self_masker_max_samples = getattr(masker, "max_samples", None)
            masker_mode = getattr(masker, "perturbation", None)
            if masker == "independent" or masker_mode == "interventional":
                perturbation = "interventional"
            elif masker == "conditional_knn" or masker_mode == "conditional_knn":
                perturbation = "conditional_knn"
                if hasattr(masker, "knn_neighbors"):
                    knn_neighbors = masker.knn_neighbors
            elif callable(masker) or masker_mode == "user_sampler":
                perturbation = "user_sampler"
                perturbation_sampler = masker
            else:
                raise ValueError(
                    "masker must be 'independent', 'conditional_knn', a DhondtXAI masker, "
                    "or a callable perturbation sampler."
                )
        else:
            self_masker_max_samples = None

        if model is None and predict_fn is None:
            raise ValueError("Provide either model or score_fn.")
        if task not in {"auto", "classification", "regression", "custom"}:
            raise ValueError("task must be 'auto', 'classification', 'regression', or 'custom'.")
        if output_type not in SUPPORTED_OUTPUT_TYPES:
            raise ValueError("output_type must be auto, probability, logit, log_odds, decision, prediction, or custom.")
        if input_format not in {"auto", "dataframe", "numpy"}:
            raise ValueError("input_format must be 'auto', 'dataframe', or 'numpy'.")
        if model_adapter not in {
            "auto",
            "sklearn",
            "xgboost",
            "lightgbm",
            "catboost",
            "torch",
            "keras",
            "callable",
        }:
            raise ValueError(
                "model_adapter must be auto, sklearn, xgboost, lightgbm, catboost, "
                "torch, keras, or callable."
            )
        if perturbation not in {"interventional", "conditional_knn", "user_sampler"}:
            raise ValueError("perturbation must be 'interventional', 'conditional_knn', or 'user_sampler'.")
        if perturbation == "user_sampler" and perturbation_sampler is None:
            raise ValueError("perturbation='user_sampler' requires perturbation_sampler.")
        if affinity_mode not in {"same_direction", "absolute_interaction"}:
            raise ValueError("affinity_mode must be 'same_direction' or 'absolute_interaction'.")
        if tie_break not in {"stable", "random"}:
            raise ValueError("tie_break must be 'stable' or 'random'.")
        if projection_mode not in {"auto", "redistribute", "residual"}:
            raise ValueError("projection_mode must be 'auto', 'redistribute', or 'residual'.")
        projection_residual_threshold = float(projection_residual_threshold)
        if projection_residual_threshold < 0:
            raise ValueError("projection_residual_threshold must be non-negative.")
        if exclude_mode not in {"strict", "standard"}:
            raise ValueError("exclude_mode must be 'strict' or 'standard'.")
        if threshold_mode not in {"strict_residual", "standard"}:
            raise ValueError("threshold_mode must be 'strict_residual' or 'standard'.")
        if baseline_mode not in SUPPORTED_BASELINE_MODES:
            raise ValueError("baseline_mode must be 'full', 'sample', or 'auto'.")
        if feature_reference not in {"auto", "name", "position"}:
            raise ValueError("feature_reference must be 'auto', 'name', or 'position'.")
        if cost_mode not in SUPPORTED_COST_MODES:
            raise ValueError("cost_mode must be auto, fast, balanced, accurate, or research.")
        if interaction_screening not in {"top_effects", "all"}:
            raise ValueError("interaction_screening must be 'top_effects' or 'all'.")
        if int(knn_neighbors) <= 0:
            raise ValueError("knn_neighbors must be positive.")
        probability_tolerance = float(probability_tolerance)
        if probability_tolerance < 0:
            raise ValueError("probability_tolerance must be non-negative.")

        self.model = model
        self.predict_fn = predict_fn
        self.background_data = None
        self.task = task
        self.output_type = output_type
        self.class_index = class_index
        self.target_index = target_index
        self.explicit_feature_names = None if feature_names is None else list(feature_names)
        self.input_format = input_format
        self.input_adapter = input_adapter
        self.output_adapter = output_adapter
        self.predict_kwargs = {} if predict_kwargs is None else dict(predict_kwargs)
        self.validate_probability = bool(validate_probability)
        self.probability_tolerance = probability_tolerance
        self.model_adapter = model_adapter
        self.perturbation = perturbation
        self.perturbation_sampler = perturbation_sampler
        self.knn_neighbors = int(knn_neighbors)
        self.affinity_mode = affinity_mode
        self.tie_break = tie_break
        self.projection_mode = projection_mode
        self.projection_residual_threshold = projection_residual_threshold
        self.exclude_mode = exclude_mode
        self.threshold_mode = threshold_mode
        self.baseline_mode = baseline_mode
        self.feature_reference = feature_reference
        self.cost_mode = cost_mode
        self.preset = cost_mode if cost_mode != "auto" else "balanced"
        self.max_model_rows = max_model_rows
        self.max_interaction_pairs = max_interaction_pairs
        self.top_k_interaction_features = top_k_interaction_features
        self.interaction_screening = interaction_screening
        self.global_max_rows = global_max_rows
        self.masker_max_samples = self_masker_max_samples
        self.eps = eps
        self.random_state = random_state
        self.strict_features = strict_features

        self.features = None
        self.feature_importances = None
        self.correlation_info = None
        self.last_explanation = None
        self.global_explanations_ = None
        self.global_frame_ = None
        self.global_alliance_matrix_ = None
        self.global_random_states_ = None
        self._baseline_cache = {}
        self._feature_positions = {}
        self._integer_target = integer_target
        self._target_explicit = target not in (None, "auto") or target_index is not None or class_index != 1
        self._warned_implicit_multiclass = False

        if background_data is not None:
            self.background_data = self._ensure_frame(background_data)
            if self.explicit_feature_names is not None:
                if len(self.explicit_feature_names) != len(self.background_data.columns):
                    raise ValueError("feature_names length must match the number of background columns.")
                self.background_data.columns = self.explicit_feature_names
            self.features = list(self.background_data.columns)
            self._validate_feature_names(self.features)
            self._refresh_feature_positions()

    def fit(self, X_train, y_train=None, fit_model=True):
        X_train = self._ensure_frame(X_train)
        if self.explicit_feature_names is not None:
            if len(self.explicit_feature_names) != len(X_train.columns):
                raise ValueError("feature_names length must match the number of training columns.")
            X_train.columns = self.explicit_feature_names
        if fit_model and y_train is not None:
            if self.model is None:
                raise ValueError("Cannot fit a model when model=None. Provide a model or set fit_model=False.")
            self.model.fit(X_train, y_train)

        self.features = list(X_train.columns)
        self._validate_feature_names(self.features)
        self._refresh_feature_positions()
        self.background_data = X_train.reset_index(drop=True)
        self.feature_importances = None
        self.reset_cache()

        if hasattr(self.model, "feature_importances_"):
            self.feature_importances = np.asarray(self.model.feature_importances_, dtype=float)

        if y_train is not None:
            y_series = pd.Series(y_train).reset_index(drop=True)
            if not pd.api.types.is_numeric_dtype(y_series):
                y_series = pd.Series(pd.factorize(y_series)[0])
            numeric_X = X_train.select_dtypes(include=[np.number]).reset_index(drop=True)
            correlations = numeric_X.corrwith(y_series).fillna(0.0).to_dict()
            self.correlation_info = {feature: correlations.get(feature, 0.0) for feature in self.features}
        else:
            self.correlation_info = {feature: 0.0 for feature in self.features}

        return self

    @classmethod
    def from_score_function(cls, score_fn, background_data, feature_names=None, **kwargs):
        """Create an explainer from a row-wise numeric scoring function."""
        return cls(score_fn=score_fn, background_data=background_data, feature_names=feature_names, **kwargs)

    def explain(
        self,
        x,
        class_index=None,
        target_index=None,
        seats=100,
        allocation_seats=None,
        threshold=None,
        threshold_enabled=None,
        redistribute=False,
        alliance_mode="none",
        user_alliances=None,
        exclude_features=None,
        n_background=None,
        lambda_interaction=0.0,
        lambda_alliance_vote=None,
        lambda_member_split=None,
        rho=0.5,
        beta=1.0,
        auto_alliance_method="connected_components",
        perturbation=None,
        perturbation_sampler=None,
        affinity_mode=None,
        tie_break=None,
        projection_mode=None,
        projection_residual_threshold=None,
        exclude_mode=None,
        threshold_mode=None,
        baseline_mode=None,
        cost_mode=None,
        preset=None,
        max_model_rows=None,
        max_interaction_pairs=None,
        top_k_interaction_features=None,
        interaction_screening=None,
        allocation_error_tolerance=None,
        random_state=None,
    ):
        if self.background_data is None or self.features is None:
            raise ValueError("Call fit(...) or provide background_data before explain(...).")

        if seats <= 0:
            raise ValueError("seats must be a positive integer.")
        if beta <= 0:
            raise ValueError("beta must be positive.")
        if lambda_interaction < 0:
            raise ValueError("lambda_interaction must be non-negative.")
        lambda_alliance_vote = lambda_interaction if lambda_alliance_vote is None else lambda_alliance_vote
        lambda_member_split = lambda_interaction if lambda_member_split is None else lambda_member_split
        if lambda_alliance_vote < 0:
            raise ValueError("lambda_alliance_vote must be non-negative.")
        if lambda_member_split < 0:
            raise ValueError("lambda_member_split must be non-negative.")
        if not 0 <= rho <= 1:
            raise ValueError("rho must be in [0, 1].")

        explanation_target_explicit = class_index is not None or target_index is not None
        x_series = self._ensure_series(x)
        class_index = self.class_index if class_index is None else class_index
        target_index = self.target_index if target_index is None else target_index
        perturbation = self.perturbation if perturbation is None else perturbation
        perturbation_sampler = self.perturbation_sampler if perturbation_sampler is None else perturbation_sampler
        affinity_mode = self.affinity_mode if affinity_mode is None else affinity_mode
        tie_break = self.tie_break if tie_break is None else tie_break
        projection_mode = self.projection_mode if projection_mode is None else projection_mode
        projection_residual_threshold = (
            self.projection_residual_threshold
            if projection_residual_threshold is None
            else float(projection_residual_threshold)
        )
        exclude_mode = self.exclude_mode if exclude_mode is None else exclude_mode
        threshold_mode = self.threshold_mode if threshold_mode is None else threshold_mode
        baseline_mode = self.baseline_mode if baseline_mode is None else baseline_mode
        requested_cost_mode = self.cost_mode if cost_mode is None else cost_mode
        if preset is not None:
            if requested_cost_mode not in {None, "auto", preset}:
                raise ValueError("Use either cost_mode or preset when they differ.")
            requested_cost_mode = preset
        if requested_cost_mode is None:
            requested_cost_mode = "auto"
        max_model_rows = self.max_model_rows if max_model_rows is None else max_model_rows
        max_interaction_pairs = (
            self.max_interaction_pairs if max_interaction_pairs is None else max_interaction_pairs
        )
        top_k_interaction_features = (
            self.top_k_interaction_features
            if top_k_interaction_features is None
            else top_k_interaction_features
        )
        interaction_screening = self.interaction_screening if interaction_screening is None else interaction_screening
        if perturbation not in {"interventional", "conditional_knn", "user_sampler"}:
            raise ValueError("perturbation must be 'interventional', 'conditional_knn', or 'user_sampler'.")
        if perturbation == "user_sampler" and perturbation_sampler is None:
            raise ValueError("perturbation='user_sampler' requires perturbation_sampler.")
        if affinity_mode not in {"same_direction", "absolute_interaction"}:
            raise ValueError("affinity_mode must be 'same_direction' or 'absolute_interaction'.")
        if tie_break not in {"stable", "random"}:
            raise ValueError("tie_break must be 'stable' or 'random'.")
        if projection_mode not in {"auto", "redistribute", "residual"}:
            raise ValueError("projection_mode must be 'auto', 'redistribute', or 'residual'.")
        if projection_residual_threshold < 0:
            raise ValueError("projection_residual_threshold must be non-negative.")
        if exclude_mode not in {"strict", "standard"}:
            raise ValueError("exclude_mode must be 'strict' or 'standard'.")
        if threshold_mode not in {"strict_residual", "standard"}:
            raise ValueError("threshold_mode must be 'strict_residual' or 'standard'.")
        if baseline_mode not in SUPPORTED_BASELINE_MODES:
            raise ValueError("baseline_mode must be 'full', 'sample', or 'auto'.")
        if requested_cost_mode not in SUPPORTED_COST_MODES:
            raise ValueError("cost_mode must be auto, fast, balanced, accurate, or research.")
        if interaction_screening not in {"top_effects", "all"}:
            raise ValueError("interaction_screening must be 'top_effects' or 'all'.")
        rng = np.random.default_rng(self.random_state if random_state is None else random_state)

        excluded = self._normalize_feature_list(exclude_features or [], self.features)
        active_features = [feature for feature in self.features if feature not in excluded]
        if not active_features:
            raise ValueError("No active features remain after exclusions.")
        needs_pair_candidates = alliance_mode in {"auto", "hybrid"} or lambda_alliance_vote > 0 or lambda_member_split > 0
        policy = self._resolve_cost_policy(
            requested_cost_mode,
            n_features=len(active_features),
            needs_interactions=needs_pair_candidates,
            requested_background=n_background,
            max_model_rows=max_model_rows,
        )
        if n_background is None:
            if self.masker_max_samples is not None:
                n_background = int(self.masker_max_samples)
            else:
                n_background = policy.n_background
            n_background = min(n_background, len(self.background_data))
        if allocation_seats is None:
            if allocation_error_tolerance is not None:
                tolerance = float(allocation_error_tolerance)
                if tolerance <= 0:
                    raise ValueError("allocation_error_tolerance must be positive.")
                allocation_seats = max(seats, int(np.ceil(1.0 / tolerance)))
            else:
                allocation_seats = max(policy.allocation_seats, 100 * len(active_features), seats)
        if max_model_rows is None:
            max_model_rows = policy.max_model_rows
        if max_interaction_pairs is None:
            max_interaction_pairs = policy.max_interaction_pairs
        if top_k_interaction_features is None:
            top_k_interaction_features = policy.top_k_interaction_features
        if interaction_screening is None:
            interaction_screening = policy.interaction_screening
        if allocation_seats <= 0:
            raise ValueError("allocation_seats must be a positive integer.")
        if max_model_rows is not None:
            max_model_rows = int(max_model_rows)
            if max_model_rows <= 0:
                raise ValueError("max_model_rows must be positive when provided.")
            min_rows = len(active_features)
            if min_rows > max_model_rows:
                raise ValueError(
                    "max_model_rows is too small for the active feature count. "
                    f"Need at least {min_rows} model rows before background replication."
                )
            if n_background * min_rows > max_model_rows:
                n_background = max(1, max_model_rows // min_rows)
        background_sample = self._sample_background(n_background, rng)
        x_frame = pd.DataFrame([x_series[self.features]])
        resolved_output_type = self._resolve_output_type()
        if (
            resolved_output_type in {"prediction", "custom"}
            and target_index is None
            and self._integer_target is not None
            and class_index == self._integer_target
        ):
            target_index = self._integer_target
            class_index = None
        class_index = self._resolve_class_index_for_x(x_frame, class_index, target_index)
        class_index = self._normalize_class_index_for_output(x_frame, class_index, target_index)
        self._warn_if_implicit_multiclass(
            x_frame,
            class_index,
            target_index,
            target_explicit=explanation_target_explicit,
        )

        score = float(
            self._score_frame(
                x_frame,
                class_index=class_index,
                target_index=target_index,
            )[0]
        )
        full_baseline = self._baseline(class_index=class_index, target_index=target_index)
        sample_baseline = self._baseline_from_frame(
            background_sample,
            class_index=class_index,
            target_index=target_index,
        )
        resolved_baseline_mode = "sample" if baseline_mode == "auto" and len(background_sample) < len(self.background_data) else baseline_mode
        baseline = sample_baseline if resolved_baseline_mode == "sample" else full_baseline
        baseline_sampling_gap = sample_baseline - full_baseline
        baseline_sampling_gap_ratio = abs(baseline_sampling_gap) / (max(abs(full_baseline), abs(sample_baseline), self.eps))
        delta = score - baseline
        strict_exclusion = bool(excluded) and exclude_mode == "strict"
        effect_context_features = tuple(active_features) if strict_exclusion else tuple(self.features)
        effect_base_score = (
            self._context_score(
                x_series,
                effect_context_features,
                background_sample,
                class_index,
                target_index,
                perturbation=perturbation,
                perturbation_sampler=perturbation_sampler,
            )
            if strict_exclusion
            else score
        )

        single_effects = self._batch_removal_effects(
            x_series,
            [(feature,) for feature in active_features],
            effect_base_score,
            background_sample,
            class_index,
            target_index,
            perturbation=perturbation,
            perturbation_sampler=perturbation_sampler,
            context_features=effect_context_features,
        )
        feature_effects = {feature: single_effects[(feature,)] for feature in active_features}

        need_interactions = alliance_mode in {"auto", "hybrid"} or lambda_alliance_vote > 0 or lambda_member_split > 0
        pair_effects = {}
        interactions = {}
        total_pair_count = len(active_features) * (len(active_features) - 1) // 2
        pair_selection = {
            "pairs": [],
            "total_pairs": total_pair_count,
            "used_pairs": 0,
            "screening_method": "none",
            "notes": [],
        }
        if need_interactions and len(active_features) > 1:
            pair_selection = self._select_interaction_pairs(
                active_features=active_features,
                feature_effects=feature_effects,
                alliance_mode=alliance_mode,
                user_alliances=user_alliances or [],
                lambda_alliance_vote=lambda_alliance_vote,
                lambda_member_split=lambda_member_split,
                max_interaction_pairs=max_interaction_pairs,
                top_k_interaction_features=top_k_interaction_features,
                interaction_screening=interaction_screening,
            )
            if max_model_rows is not None:
                allowed_pairs = max(0, max_model_rows // int(n_background) - len(active_features))
                if pair_selection["used_pairs"] > allowed_pairs:
                    pair_selection["pairs"] = pair_selection["pairs"][:allowed_pairs]
                    pair_selection["used_pairs"] = len(pair_selection["pairs"])
                    pair_selection["notes"].append("Interaction pairs were capped by max_model_rows.")
            pair_groups = pair_selection["pairs"]
            pair_effects = self._batch_removal_effects(
                x_series,
                pair_groups,
                effect_base_score,
                background_sample,
                class_index,
                target_index,
                perturbation=perturbation,
                perturbation_sampler=perturbation_sampler,
                context_features=effect_context_features,
            ) if pair_groups else {}
            for key in pair_groups:
                left, right = key
                interactions[key] = pair_effects[key] - feature_effects[left] - feature_effects[right]

        affinity = self._build_affinity(active_features, feature_effects, pair_effects, interactions, affinity_mode)
        alliances = self._build_alliances(
            active_features=active_features,
            mode=alliance_mode,
            user_alliances=user_alliances,
            affinity=affinity,
            rho=rho,
            auto_alliance_method=auto_alliance_method,
        )

        alliance_members = {self._alliance_name(group): group for group in alliances}
        feature_source_alliance = {}
        for name, members in alliance_members.items():
            for feature in members:
                feature_source_alliance[feature] = name

        alliance_effects = {}
        chi = {}
        votes = {}
        positive_votes = {}
        negative_votes = {}
        value_masses = {}
        positive_value_masses = {}
        negative_value_masses = {}
        alliance_groups_to_score = []
        for group in alliances:
            if len(group) > 1:
                key = self._group_key(group)
                if key not in pair_effects:
                    alliance_groups_to_score.append(group)
        skipped_alliance_groups = []
        if max_model_rows is not None and alliance_groups_to_score:
            used_groups = len(active_features) + pair_selection.get("used_pairs", 0)
            allowed_group_count = max(0, max_model_rows // int(n_background) - used_groups)
            if len(alliance_groups_to_score) > allowed_group_count:
                skipped_alliance_groups = alliance_groups_to_score[allowed_group_count:]
                alliance_groups_to_score = alliance_groups_to_score[:allowed_group_count]
                pair_selection["notes"].append(
                    "Some multi-feature alliance effects used summed member effects because max_model_rows was reached."
                )
        alliance_group_effects = self._batch_removal_effects(
            x_series,
            alliance_groups_to_score,
            effect_base_score,
            background_sample,
            class_index,
            target_index,
            perturbation=perturbation,
            perturbation_sampler=perturbation_sampler,
            context_features=effect_context_features,
        ) if alliance_groups_to_score else {}
        for group in alliances:
            name = self._alliance_name(group)
            if len(group) == 1:
                effect = feature_effects[group[0]]
            else:
                key = self._group_key(group)
                effect = pair_effects.get(key, alliance_group_effects.get(tuple(group)))
                if effect is None and group in skipped_alliance_groups:
                    effect = sum(feature_effects.get(feature, 0.0) for feature in group)
            alliance_effects[name] = effect
            chi[name] = self._interaction_strength(group, interactions)
            votes[name] = abs(effect) + lambda_alliance_vote * chi[name]
            positive_votes[name] = votes[name] if effect > 0 else 0.0
            negative_votes[name] = votes[name] if effect < 0 else 0.0
            value_masses[name] = abs(effect)
            positive_value_masses[name] = abs(effect) if effect > 0 else 0.0
            negative_value_masses[name] = abs(effect) if effect < 0 else 0.0

        tau, threshold_is_enabled = self._normalize_threshold(threshold, threshold_enabled)
        eligible, below_threshold = self._threshold_alliances(votes, tau, threshold_is_enabled)

        transfer = self._transfer_matrix(
            alliance_members=alliance_members,
            votes=votes,
            eligible=eligible,
            below_threshold=below_threshold,
            affinity=affinity,
            redistribute=redistribute,
        )

        updated_positive = {}
        updated_negative = {}
        updated_positive_value = {}
        updated_negative_value = {}
        for target in eligible:
            updated_positive[target] = 0.0
            updated_negative[target] = 0.0
            updated_positive_value[target] = 0.0
            updated_negative_value[target] = 0.0
            for source in alliance_members:
                weight = transfer.get((source, target), 0.0)
                updated_positive[target] += weight * positive_votes[source]
                updated_negative[target] += weight * negative_votes[source]
                updated_positive_value[target] += weight * positive_value_masses[source]
                updated_negative_value[target] += weight * negative_value_masses[source]

        total_positive_for_seats = sum(updated_positive.values())
        total_negative_for_seats = sum(updated_negative.values())
        total_positive = sum(updated_positive_value.values())
        total_negative = sum(updated_negative_value.values())
        allocation_positive_count, allocation_negative_count = self._split_signed_seats(
            allocation_seats, total_positive_for_seats, total_negative_for_seats, delta
        )
        display_positive_count, display_negative_count = self._split_signed_seats(
            seats, total_positive_for_seats, total_negative_for_seats, delta
        )

        allocation_positive_values = self._dhondt_allocate(
            [updated_positive[name] for name in eligible], allocation_positive_count, tie_break=tie_break, rng=rng
        )
        allocation_negative_values = self._dhondt_allocate(
            [updated_negative[name] for name in eligible], allocation_negative_count, tie_break=tie_break, rng=rng
        )
        positive_seat_values = self._dhondt_allocate(
            [updated_positive[name] for name in eligible], display_positive_count, tie_break=tie_break, rng=rng
        )
        negative_seat_values = self._dhondt_allocate(
            [updated_negative[name] for name in eligible], display_negative_count, tie_break=tie_break, rng=rng
        )

        positive_seats = {name: 0 for name in alliance_members}
        negative_seats = {name: 0 for name in alliance_members}
        allocation_positive_seats = {name: 0 for name in alliance_members}
        allocation_negative_seats = {name: 0 for name in alliance_members}
        for index, name in enumerate(eligible):
            positive_seats[name] = int(positive_seat_values[index])
            negative_seats[name] = int(negative_seat_values[index])
            allocation_positive_seats[name] = int(allocation_positive_values[index])
            allocation_negative_seats[name] = int(allocation_negative_values[index])

        represented_positive_raw = {}
        represented_negative_raw = {}
        represented_raw_values = {}
        for name in eligible:
            positive_share = (
                allocation_positive_seats[name] / allocation_positive_count
                if allocation_positive_count > 0
                else 0.0
            )
            negative_share = (
                allocation_negative_seats[name] / allocation_negative_count
                if allocation_negative_count > 0
                else 0.0
            )
            represented_positive_raw[name] = total_positive * positive_share
            represented_negative_raw[name] = total_negative * negative_share
            represented_raw_values[name] = represented_positive_raw[name] - represented_negative_raw[name]

        if not excluded:
            active_delta = delta
        elif strict_exclusion:
            active_delta = effect_base_score - baseline
        else:
            active_delta = self._removal_effect(
                x_series, active_features, score, background_sample, class_index, target_index,
                perturbation=perturbation, perturbation_sampler=perturbation_sampler
            )
        if threshold_is_enabled and not redistribute:
            eligible_features = self._features_from_alliances(eligible, alliance_members)
            if set(eligible_features) == set(active_features):
                projection_target = active_delta
            elif threshold_mode == "strict_residual":
                projection_target = self._context_score(
                    x_series,
                    eligible_features,
                    background_sample,
                    class_index,
                    target_index,
                    perturbation=perturbation,
                    perturbation_sampler=perturbation_sampler,
                ) - baseline
            else:
                projection_target = self._removal_effect(
                    x_series, eligible_features, score, background_sample, class_index, target_index,
                    perturbation=perturbation, perturbation_sampler=perturbation_sampler
                )
            projected_source_names = eligible
        else:
            projection_target = active_delta
            projected_source_names = list(alliance_members.keys())

        excluded_residual = delta - active_delta
        below_threshold_residual = active_delta - projection_target

        represented_attributions, _, _, _, _ = self._project_values(
            represented_raw_values, projection_target, eligible
        )
        source_raw_values = self._back_project_sources_signed(
            represented_positive_raw=represented_positive_raw,
            represented_negative_raw=represented_negative_raw,
            alliance_members=alliance_members,
            positive_votes=positive_value_masses,
            negative_votes=negative_value_masses,
            eligible=eligible,
            transfer=transfer,
        )
        (
            source_attributions,
            raw_sum,
            projection_residual,
            projection_residual_ratio,
            projection_residual_attribution,
        ) = self._project_values(
            source_raw_values,
            projection_target,
            projected_source_names,
            mode=projection_mode,
            residual_name="__projection_residual__",
            residual_threshold=projection_residual_threshold,
        )
        for name in alliance_members:
            source_attributions.setdefault(name, 0.0)

        feature_attributions = self._distribute_to_features(
            source_attributions=source_attributions,
            alliance_members=alliance_members,
            feature_effects=feature_effects,
            interactions=interactions,
            lambda_interaction=lambda_member_split,
            beta=beta,
        )

        for feature in self.features:
            feature_attributions.setdefault(feature, 0.0)
        if excluded:
            feature_attributions["__excluded__"] = excluded_residual
        if threshold_is_enabled and not redistribute and below_threshold:
            feature_attributions["__below_threshold__"] = below_threshold_residual
        if abs(projection_residual_attribution) > self.eps:
            feature_attributions["__projection_residual__"] = projection_residual_attribution
        alliance_sign_conflicts = self._alliance_sign_conflicts(alliance_members, feature_attributions)
        estimated_model_rows = int(n_background) * (
            len(active_features) + pair_selection.get("used_pairs", 0) + len(alliance_groups_to_score)
        )
        approximation_notes = list(pair_selection.get("notes", []))
        if self.masker_max_samples is not None and n_background == min(int(self.masker_max_samples), len(self.background_data)):
            approximation_notes.append("Background rows were capped by masker.max_samples.")
        if policy.name in {"fast", "balanced"}:
            approximation_notes.append(
                "Use cost_mode='accurate' or cost_mode='research' for a more exhaustive explanation."
            )
        cost_diagnostics = {
            "requested_cost_mode": requested_cost_mode,
            "resolved_cost_mode": policy.name,
            "n_features": len(active_features),
            "n_background": int(n_background),
            "estimated_model_rows": estimated_model_rows,
            "interaction_pairs_total": pair_selection.get("total_pairs", total_pair_count),
            "interaction_pairs_used": pair_selection.get("used_pairs", 0),
            "allocation_seats": int(allocation_seats),
            "screening_method": pair_selection.get("screening_method", interaction_screening),
            "max_model_rows": max_model_rows,
            "approximation_notes": approximation_notes,
        }

        explanation = DhondtExplanation(
            score=score,
            baseline=baseline,
            delta=delta,
            active_delta=active_delta,
            feature_attributions=feature_attributions,
            feature_order=list(self.features),
            represented_alliance_attributions=represented_attributions,
            source_alliance_attributions=source_attributions,
            represented_raw_attributions=represented_raw_values,
            source_raw_attributions=source_raw_values,
            represented_positive_raw=represented_positive_raw,
            represented_negative_raw=represented_negative_raw,
            positive_seats=positive_seats,
            negative_seats=negative_seats,
            allocation_positive_seats=allocation_positive_seats,
            allocation_negative_seats=allocation_negative_seats,
            votes=votes,
            positive_votes=positive_votes,
            negative_votes=negative_votes,
            value_masses=value_masses,
            positive_value_masses=positive_value_masses,
            negative_value_masses=negative_value_masses,
            effects={**feature_effects, **alliance_effects},
            interactions=interactions,
            alliance_members=alliance_members,
            alliance_sign_conflicts=alliance_sign_conflicts,
            feature_source_alliance=feature_source_alliance,
            eligible_alliances=eligible,
            below_threshold_alliances=below_threshold,
            excluded_features=excluded,
            excluded_residual=excluded_residual,
            below_threshold_residual=below_threshold_residual,
            raw_attribution_sum=raw_sum,
            projection_target=projection_target,
            projection_residual=projection_residual,
            projection_residual_ratio=projection_residual_ratio,
            threshold=tau if threshold_is_enabled else None,
            redistribution=redistribute,
            seat_count=seats,
            allocation_seat_count=allocation_seats,
            class_index=class_index,
            target_index=target_index,
            class_label=self._class_label(class_index),
            output_type=self.output_type,
            resolved_output_type=resolved_output_type,
            perturbation=perturbation,
            affinity_mode=affinity_mode,
            tie_break=tie_break,
            projection_mode=projection_mode,
            projection_residual_attribution=projection_residual_attribution,
            projection_residual_threshold=projection_residual_threshold,
            exclude_mode=exclude_mode,
            threshold_mode=threshold_mode,
            baseline_mode=resolved_baseline_mode,
            full_baseline=full_baseline,
            sample_baseline=sample_baseline,
            baseline_sampling_gap=baseline_sampling_gap,
            baseline_sampling_gap_ratio=baseline_sampling_gap_ratio,
            data=x_series[self.features].to_numpy(),
            cost_diagnostics=cost_diagnostics,
        )
        self.last_explanation = explanation
        return explanation

    def __call__(self, X, **kwargs):
        return self.dhondtxai_values(X, **kwargs)

    def dhondtxai_values(
        self,
        X,
        max_rows=None,
        random_state=None,
        reuse_background_sample=False,
        **kwargs,
    ):
        return self.values(
            X,
            max_rows=max_rows,
            random_state=random_state,
            reuse_background_sample=reuse_background_sample,
            **kwargs,
        )

    def values(self, X, max_rows=None, random_state=None, reuse_background_sample=False, **kwargs):
        """Return DhondtXAI attribution values.

        For a single row, ``values`` is a one-dimensional array with one value
        per original feature. For a table, ``values`` is a two-dimensional
        ``n_rows x n_features`` matrix. Detailed local explanations remain
        available through the returned object's ``explanations`` attribute.
        """
        if self.background_data is None or self.features is None:
            raise ValueError("Call fit(...) or provide background_data before values(...).")

        single_output = self._is_single_input(X)
        if single_output:
            explanation = self.explain(X, random_state=random_state, **kwargs)
            return DhondtValues(
                [explanation],
                self.features,
                data=self._single_input_data(X),
                single_output=True,
            )

        X_frame = self.background_data if X is None else self._ensure_frame(X)
        if max_rows is not None:
            X_frame = X_frame.iloc[:max_rows]
        if len(X_frame) == 0:
            raise ValueError("X must contain at least one row.")
        explanations = self.explain_many(
            X_frame,
            random_state=random_state,
            reuse_background_sample=reuse_background_sample,
            **kwargs,
        )
        return DhondtValues(
            explanations,
            self.features,
            data=X_frame[self.features].to_numpy(),
            single_output=False,
        )

    def explain_many(self, X, max_rows=None, random_state=None, reuse_background_sample=False, **kwargs):
        X = self.background_data if X is None else self._ensure_frame(X)
        if max_rows is not None:
            X = X.iloc[:max_rows]
        if len(X) == 0:
            raise ValueError("X must contain at least one row.")

        explanations = []
        row_random_states = []
        rng = np.random.default_rng(self.random_state if random_state is None else random_state)
        for _, row in X.iterrows():
            row_kwargs = dict(kwargs)
            if "random_state" not in row_kwargs:
                if reuse_background_sample:
                    row_seed = self.random_state if random_state is None else random_state
                else:
                    row_seed = int(rng.integers(0, np.iinfo(np.uint32).max))
                row_kwargs["random_state"] = row_seed
                row_random_states.append(row_seed)
            explanations.append(self.explain(row, **row_kwargs))
        self.global_random_states_ = row_random_states
        return explanations

    def explain_global(self, X=None, max_rows=None, random_state=None, reuse_background_sample=False, **kwargs):
        X = self.background_data if X is None else self._ensure_frame(X)
        if len(X) == 0:
            raise ValueError("X must contain at least one row.")
        if max_rows is None:
            requested_mode = kwargs.get("cost_mode", kwargs.get("preset", self.cost_mode))
            if requested_mode == "auto":
                requested_mode = "balanced"
            policy = COST_POLICIES.get(requested_mode, COST_POLICIES["balanced"])
            cap = self.global_max_rows if self.global_max_rows is not None else policy.global_max_rows
            if cap is not None and len(X) > int(cap):
                max_rows = int(cap)
        if max_rows is not None:
            X = X.iloc[:max_rows]

        explanations = self.explain_many(
            X,
            random_state=random_state,
            reuse_background_sample=reuse_background_sample,
            **kwargs,
        )
        self.global_explanations_ = explanations
        self.global_alliance_matrix_ = self._compute_global_alliance_matrix(explanations)

        all_keys = list(self.features)
        for explanation in explanations:
            for key in explanation.feature_attributions:
                if key not in all_keys:
                    all_keys.append(key)

        rows = []
        for feature in all_keys:
            is_residual = str(feature).startswith("__")
            values = np.asarray([exp.feature_attributions.get(feature, 0.0) for exp in explanations], dtype=float)
            rows.append(
                {
                    "feature": feature,
                    "global_abs": float(np.mean(np.abs(values))),
                    "directional": float(np.mean(values)),
                    "positive": float(np.mean(np.maximum(values, 0.0))),
                    "negative": float(np.mean(np.maximum(-values, 0.0))),
                    "threshold_survival": np.nan if is_residual else self._threshold_survival(feature, explanations),
                    "is_residual": is_residual,
                }
            )
        frame = pd.DataFrame(rows).sort_values("global_abs", ascending=False).reset_index(drop=True)
        frame.insert(0, "rank", np.arange(1, len(frame) + 1))
        self.global_frame_ = frame
        return frame

    def set_background(self, background_data):
        self.background_data = self._ensure_frame(background_data).reset_index(drop=True)
        if self.explicit_feature_names is not None:
            if len(self.explicit_feature_names) != len(self.background_data.columns):
                raise ValueError("feature_names length must match the number of background columns.")
            self.background_data.columns = self.explicit_feature_names
        self.features = list(self.background_data.columns)
        self._validate_feature_names(self.features)
        self._refresh_feature_positions()
        self.reset_cache()
        return self

    def reset_cache(self):
        self._baseline_cache = {}
        return self

    def _preset_defaults(self, preset, n_features, seats):
        if preset not in SUPPORTED_PRESETS:
            raise ValueError("preset must be 'fast', 'balanced', 'accurate', or 'research'.")
        policy = COST_POLICIES[preset]
        return {
            "n_background": policy.n_background,
            "allocation_seats": max(policy.allocation_seats, 100 * int(n_features), int(seats)),
        }

    def _resolve_cost_policy(self, cost_mode, n_features, needs_interactions, requested_background, max_model_rows):
        if cost_mode != "auto":
            return COST_POLICIES[cost_mode]

        background_rows = 100 if requested_background is None else int(requested_background)
        pair_count = n_features * (n_features - 1) // 2 if needs_interactions else 0
        estimated_rows = background_rows * (n_features + pair_count)
        cap = max_model_rows
        if cap is not None and estimated_rows > cap:
            return COST_POLICIES["fast"]
        if estimated_rows > 1_000_000:
            return COST_POLICIES["fast"]
        if estimated_rows > 250_000:
            return COST_POLICIES["balanced"]
        return COST_POLICIES["balanced"]

    def _select_interaction_pairs(
        self,
        active_features,
        feature_effects,
        alliance_mode,
        user_alliances,
        lambda_alliance_vote,
        lambda_member_split,
        max_interaction_pairs,
        top_k_interaction_features,
        interaction_screening,
    ):
        all_pairs = [self._pair_key(left, right) for left, right in combinations(active_features, 2)]
        required = []
        user_groups = self._normalize_user_alliances(user_alliances or [], active_features)
        if alliance_mode in {"user", "hybrid"} and (lambda_alliance_vote > 0 or lambda_member_split > 0):
            for group in user_groups:
                for left, right in combinations(group, 2):
                    required.append(self._pair_key(left, right))

        required = list(dict.fromkeys(required))
        if alliance_mode == "none":
            selected = required
            screening_method = "user_alliance_pairs"
        elif interaction_screening == "all" or (max_interaction_pairs is None and top_k_interaction_features is None):
            selected = list(dict.fromkeys(required + all_pairs))
            screening_method = "all"
        else:
            candidates = all_pairs
            if alliance_mode == "user":
                candidates = required
            elif top_k_interaction_features is not None:
                top_k = max(0, int(top_k_interaction_features))
                ranked_features = sorted(active_features, key=lambda f: abs(feature_effects.get(f, 0.0)), reverse=True)
                top_features = set(ranked_features[:top_k])
                candidates = [pair for pair in candidates if pair[0] in top_features and pair[1] in top_features]

            ranked_pairs = sorted(
                candidates,
                key=lambda pair: abs(feature_effects.get(pair[0], 0.0)) + abs(feature_effects.get(pair[1], 0.0)),
                reverse=True,
            )
            if max_interaction_pairs is not None:
                ranked_pairs = ranked_pairs[: max(0, int(max_interaction_pairs))]
            selected = list(dict.fromkeys(required + ranked_pairs))
            screening_method = "top_effects"

        notes = []
        if len(selected) < len(all_pairs) and alliance_mode in {"auto", "hybrid"}:
            notes.append("Only screened interaction pairs were evaluated to reduce runtime.")
        if alliance_mode == "user" and len(selected) < len(all_pairs):
            notes.append("Only interaction pairs inside user-defined alliances were evaluated.")

        return {
            "pairs": selected,
            "total_pairs": len(all_pairs),
            "used_pairs": len(selected),
            "screening_method": screening_method,
            "notes": notes,
        }

    def check_model_compatibility(self, X_sample=None, class_index=None, target_index=None, deep=False):
        if self.background_data is None and X_sample is None:
            return {
                "compatible": False,
                "problem": "No background data or X_sample was provided.",
                "suggestion": "Provide background_data, call fit(...), or pass X_sample.",
            }

        X = self.background_data.head(5) if X_sample is None else self._ensure_frame(X_sample).head(5)
        resolved_output_type = self._resolve_output_type()
        original_features = self.features
        original_positions = dict(self._feature_positions)

        try:
            if self.features is None:
                self.features = list(X.columns)
                self._validate_feature_names(self.features)
                self._refresh_feature_positions()
            X = X[self.features]
            class_index = self.class_index if class_index is None else class_index
            target_index = self.target_index if target_index is None else target_index
            class_index = self._resolve_class_index_for_x(X.head(1), class_index, target_index)
            class_index = self._normalize_class_index_for_output(X.head(1), class_index, target_index)
            model_input = self._prepare_model_input(X)
            raw_output = self._raw_model_output(model_input)
            selected = self._score_frame(X, class_index=class_index, target_index=target_index)
            deep_message = None
            if deep:
                deep_message = self._deep_compatibility_probe(X.head(1), class_index, target_index)
            return {
                "compatible": True,
                "input_format": self.input_format if self.input_adapter is None else "input_adapter",
                "model_adapter": self._resolve_model_adapter(),
                "output_type": self.output_type,
                "resolved_output_type": resolved_output_type,
                "raw_output_shape": tuple(np.asarray(raw_output).shape),
                "selected_output_shape": tuple(np.asarray(selected).shape),
                "numeric": True,
                "class_index": class_index,
                "target_index": target_index,
                "deep_check": deep_message,
                "message": "Model is compatible with DhondtXAI.",
            }
        except Exception as exc:
            return {
                "compatible": False,
                "problem": str(exc),
                "suggestion": (
                    "Check output_type, class_index/target_index, input_format/input_adapter, "
                    "or provide a numeric score_fn."
                ),
            }
        finally:
            self.features = original_features
            self._feature_positions = original_positions

    def _deep_compatibility_probe(self, X, class_index, target_index):
        original_background = self.background_data
        original_features = self.features
        original_positions = dict(self._feature_positions)
        original_last = self.last_explanation
        original_cache = dict(self._baseline_cache)
        try:
            if self.background_data is None:
                self.background_data = X.reset_index(drop=True)
                self.features = list(X.columns)
            self.explain(
                X.iloc[0],
                class_index=class_index,
                target_index=target_index,
                seats=10,
                allocation_seats=100,
                n_background=min(2, len(self.background_data)),
            )
            return "Full mini explanation path succeeded."
        finally:
            self.background_data = original_background
            self.features = original_features
            self._feature_positions = original_positions
            self.last_explanation = original_last
            self._baseline_cache = original_cache

    def select_features(self, feature_names):
        """Interactive CLI helper kept for notebooks and legacy demos.

        Core DhondtXAI usage should pass exclude_features and user_alliances
        programmatically to explain(...). This helper uses input(...) and will
        block in automated jobs.
        """
        warnings.warn(
            "select_features(...) is an interactive legacy helper. Prefer passing "
            "exclude_features and user_alliances directly to explain(...).",
            DeprecationWarning,
            stacklevel=2,
        )
        feature_names = list(feature_names)
        print("Available features:")
        for idx, feature in enumerate(feature_names):
            print(f"{idx + 1}. {feature}")

        exclude_input = input(
            "Enter variables to exclude (e.g., '2, 4' or 'var2, var4') or 'none': "
        )
        exclude_features = []
        if exclude_input.lower() != "none":
            exclude_features = self._normalize_feature_list(
                [part.strip() for part in exclude_input.split(",")], feature_names
            )

        alliances_input = input(
            "Enter alliances (e.g., '2 and 3, 4 and 5' or 'var2 and var3') or 'none': "
        )
        alliances = []
        if alliances_input.lower() != "none":
            for part in alliances_input.split(","):
                alliances.append(
                    self._normalize_feature_list([value.strip() for value in part.split(" and ")], feature_names)
                )

        return alliances, exclude_features

    def apply_dhondt(
        self,
        num_votes,
        num_mps,
        threshold=None,
        alliances=None,
        exclude_features=None,
        return_excluded=True,
    ):
        """Legacy global importance allocation kept for backward compatibility.

        The current D'Hondt-projected local method is explain(...). This method still
        supports the original feature_importances_ based workflow.
        """
        if self.feature_importances is None:
            raise ValueError("The model does not expose feature_importances_. Use explain(...) instead.")

        excluded = set(self._normalize_feature_list(exclude_features or [], self.features))
        features = [feature for feature in self.features if feature not in excluded]
        importance_by_feature = {
            feature: float(self.feature_importances[self.features.index(feature)]) for feature in features
        }

        grouped_features = []
        grouped_importances = []
        used = set()
        for group in self._normalize_user_alliances(alliances or [], features):
            name = self._alliance_name(group)
            grouped_features.append(name)
            grouped_importances.append(sum(importance_by_feature[feature] for feature in group))
            used.update(group)

        for feature in features:
            if feature not in used:
                grouped_features.append(feature)
                grouped_importances.append(importance_by_feature[feature])

        grouped_importances = np.asarray(grouped_importances, dtype=float)
        if grouped_importances.sum() <= self.eps:
            votes = np.zeros_like(grouped_importances)
        else:
            votes = grouped_importances / grouped_importances.sum() * num_votes

        excluded_mask = np.zeros(len(votes), dtype=bool)
        if threshold is not None:
            threshold_votes = (threshold / 100.0) * num_votes if threshold > 1 else threshold * num_votes
            excluded_mask = votes < threshold_votes

        if return_excluded:
            return grouped_features, votes, excluded_mask
        return grouped_features, votes

    def dhondt_method(self, votes, num_mps, excluded_features=None, tie_break=None):
        votes = np.asarray(votes, dtype=float)
        if excluded_features is not None:
            votes = np.where(np.asarray(excluded_features, dtype=bool), 0.0, votes)
        return self._dhondt_allocate(votes, num_mps, tie_break=tie_break)

    def plot_results(self, features, seats, title=None, colors=None, show=True):
        if colors is None:
            colors = []
            for feature in features:
                main_feature = feature.split(" + ")[0].strip()
                correlation = 0.0 if self.correlation_info is None else self.correlation_info.get(main_feature, 0.0)
                colors.append("tab:blue" if correlation >= 0 else "tab:red")

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(features, seats, color=colors)
        ax.set_xlabel("Feature/Alliance")
        ax.set_ylabel("Number of seats")
        ax.set_title(title or "DhondtXAI Parliamentary Seat Allocation")
        ax.tick_params(axis="x", rotation=90)
        fig.tight_layout()
        if show:
            plt.show()
        return fig, ax

    def plot_explanation(
        self,
        explanation=None,
        level="alliance",
        top_k=None,
        include_residuals=True,
        residuals="separate",
        friendly_labels=True,
        caption=True,
        language="en",
        show=True,
    ):
        explanation = explanation or self.last_explanation
        if explanation is None:
            raise ValueError("No explanation is available. Call explain(...) first.")
        _validate_language(language)
        if residuals not in {"show", "hide", "separate"}:
            raise ValueError("residuals must be 'show', 'hide', or 'separate'.")
        if not include_residuals:
            residuals = "hide"

        if level == "feature":
            frame = explanation.to_feature_frame(include_residuals=residuals != "hide")
            if residuals == "separate" and not frame.empty:
                residual_frame = frame[frame["is_residual"]].copy()
                feature_frame = frame[~frame["is_residual"]].copy()
                if top_k is not None:
                    feature_frame = feature_frame.head(top_k)
                frame = pd.concat([feature_frame, residual_frame], ignore_index=True)
            elif top_k is not None:
                frame = frame.head(top_k).reset_index(drop=True)
            names = frame["feature"].tolist()
            values = frame["attribution"].tolist()
            xlabel = "Local attribution"
        elif level == "alliance":
            frame = explanation.to_alliance_frame()
            if top_k is not None:
                frame = frame.head(top_k).reset_index(drop=True)
            names = frame["alliance"].tolist()
            values = frame["source_attribution"].tolist()
            xlabel = "Alliance attribution"
        else:
            raise ValueError("level must be 'feature' or 'alliance'.")

        colors = ["#8c8c8c" if str(name).startswith("__") else ("#0072B2" if value >= 0 else "#D55E00") for name, value in zip(names, values)]
        display_names = [
            self._display_name(name, language=language, friendly=friendly_labels)
            for name in names
        ]
        fig, ax = plt.subplots(figsize=(10, max(4, 0.42 * max(1, len(names)))))
        positions = np.arange(len(names))
        ax.barh(positions, values, color=colors)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_yticks(positions)
        ax.set_yticklabels(display_names)
        ax.invert_yaxis()
        ax.set_xlabel(xlabel)
        ax.set_title("DhondtXAI Local Explanation")
        max_abs = max([abs(v) for v in values] + [1.0])
        pad = max_abs * 0.25
        ax.set_xlim(min(0.0, min(values) if values else 0.0) - pad, max(0.0, max(values) if values else 0.0) + pad)
        for position, value in zip(positions, values):
            offset = 0.03 * max_abs
            ha = "left" if value >= 0 else "right"
            x_text = value + offset if value >= 0 else value - offset
            ax.text(x_text, position, f"{value:+.4g}", va="center", ha=ha, fontsize=9)
        if caption:
            caption_text = (
                "Blue increases the selected target; orange decreases it; "
                "gray rows are residual/correction, not input features."
            )
            ax.text(
                0.01,
                -0.12,
                caption_text,
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=9,
                color="#444444",
            )
        warning = explanation._projection_warning(language)
        if warning is not None and explanation.projection_residual_ratio >= 0.50:
            ax.text(
                0.01,
                0.98,
                "High projection correction",
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=9,
                color="darkred",
                bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "darkred"},
            )
        fig.tight_layout()
        if show:
            plt.show()
        return fig, ax

    def plot_local_bar(
        self,
        explanation=None,
        top_k=15,
        include_residuals=True,
        residuals="separate",
        friendly_labels=True,
        caption=True,
        language="en",
        show=True,
    ):
        return self.plot_explanation(
            explanation=explanation,
            level="feature",
            top_k=top_k,
            include_residuals=include_residuals,
            residuals=residuals,
            friendly_labels=friendly_labels,
            caption=caption,
            language=language,
            show=show,
        )

    def plot_waterfall(
        self,
        explanation=None,
        top_k=10,
        include_residuals=True,
        residuals="separate",
        friendly_labels=True,
        caption=True,
        language="en",
        show=True,
    ):
        explanation = explanation or self.last_explanation
        if explanation is None:
            raise ValueError("No explanation is available. Call explain(...) first.")
        _validate_language(language)
        if residuals not in {"show", "hide", "separate"}:
            raise ValueError("residuals must be 'show', 'hide', or 'separate'.")
        if not include_residuals:
            residuals = "hide"

        hidden_residual = 0.0 if residuals != "hide" else sum(explanation.residual_values.values())
        frame = explanation.to_feature_frame(include_residuals=residuals != "hide")
        if residuals == "separate" and not frame.empty:
            residual_frame = frame[frame["is_residual"]].copy()
            frame = frame[~frame["is_residual"]].copy()
        else:
            residual_frame = pd.DataFrame()
        if top_k is not None and len(frame) > top_k:
            selected = frame.head(top_k).copy()
            other_value = frame.iloc[top_k:]["attribution"].sum()
            if abs(other_value) > self.eps:
                selected = pd.concat(
                    [
                        selected,
                        pd.DataFrame(
                            [
                                {
                                    "feature": "__other__",
                                    "attribution": other_value,
                                    "abs_attribution": abs(other_value),
                                    "direction": (
                                        "increases selected target score"
                                        if other_value > 0
                                        else "decreases selected target score"
                                    ),
                                    "is_residual": False,
                                    "is_aggregate": True,
                                }
                            ]
                        ),
                    ],
                    ignore_index=True,
                )
            frame = selected
        if residuals == "separate" and not residual_frame.empty:
            frame = pd.concat([frame, residual_frame], ignore_index=True)

        labels = ["baseline"] + [
            self._display_name(feature, language=language, friendly=friendly_labels)
            for feature in frame["feature"].tolist()
        ] + ["score"]
        current = explanation.baseline
        bottoms = []
        heights = []
        colors = []
        endpoints = [current]
        for feature, value in zip(frame["feature"], frame["attribution"].to_numpy(dtype=float)):
            next_value = current + value
            bottoms.append(min(current, next_value))
            heights.append(abs(value))
            if str(feature) in explanation.residual_values:
                colors.append("#8c8c8c")
            elif str(feature) == "__other__":
                colors.append("#6f6f6f")
            else:
                colors.append("#0072B2" if value >= 0 else "#D55E00")
            current = next_value
            endpoints.append(current)

        fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.9), 6), constrained_layout=True)
        ax.scatter([0], [explanation.baseline], color="black", zorder=3, label="baseline")
        for index, (bottom, height, color, value) in enumerate(
            zip(bottoms, heights, colors, frame["attribution"].to_numpy(dtype=float)),
            start=1,
        ):
            ax.bar(index, height, bottom=bottom, color=color, width=0.65)
            y_text = bottom + height if value >= 0 else bottom
            va = "bottom" if value >= 0 else "top"
            ax.text(index, y_text, f"{value:+.4g}", ha="center", va=va, fontsize=8)
            if index < len(endpoints):
                ax.plot(
                    [index + 0.325, index + 1 - 0.325],
                    [endpoints[index], endpoints[index]],
                    color="gray",
                    linewidth=0.8,
                    linestyle="--",
                )
        ax.scatter([len(labels) - 1], [explanation.score], color="black", zorder=3, label="score")
        ax.axhline(explanation.baseline, color="gray", linewidth=0.8, linestyle="--")
        ax.axhline(explanation.score, color="black", linewidth=0.8, linestyle=":")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels([self._wrap_label(label) for label in labels], rotation=65, ha="right")
        ax.set_ylabel("Model score")
        ax.set_title(f"DhondtXAI Waterfall - {explanation._target_label()}, {explanation.resolved_output_type}")
        if caption:
            caption_text = (
                "Blue raises the selected target, orange lowers it, "
                "gray is correction/residual and not an input feature. "
                "'Other features' aggregates smaller input features."
            )
            ax.text(0.01, -0.20, caption_text, transform=ax.transAxes, va="top", ha="left", fontsize=9)
        notes = []
        if abs(hidden_residual) > self.eps:
            notes.append("residuals hidden; bars may not sum to score")
        if explanation.projection_residual_ratio >= 0.50:
            notes.append("high projection correction")
        if notes:
            ax.text(
                0.01,
                0.98,
                "\n".join(notes),
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=9,
                color="darkred",
                bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "darkred"},
            )
        if not getattr(fig, "get_constrained_layout", lambda: False)():
            fig.tight_layout()
            fig.subplots_adjust(bottom=0.35)
        if show:
            plt.show()
        return fig, ax

    def plot_global_importance(self, global_frame=None, top_k=20, include_residuals=True, show=True):
        frame = global_frame if global_frame is not None else self.global_frame_
        if frame is None:
            raise ValueError("No global explanation is available. Call explain_global(...) first.")
        frame = frame.copy()
        if not include_residuals and "is_residual" in frame:
            frame = frame[~frame["is_residual"]]
        frame = frame.sort_values("global_abs", ascending=False).head(top_k)
        labels = [
            self._display_name(feature, friendly=True)
            for feature in frame["feature"].tolist()
        ]
        colors = [
            "tab:gray" if bool(row.is_residual) else ("tab:blue" if row.directional >= 0 else "tab:red")
            for row in frame.itertuples(index=False)
        ]

        fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(frame))))
        ax.barh(labels, frame["global_abs"], color=colors)
        ax.invert_yaxis()
        ax.set_xlabel("Mean absolute DhondtXAI attribution")
        ax.set_title("DhondtXAI Global Importance")
        fig.tight_layout()
        if show:
            plt.show()
        return fig, ax

    def plot_global_alliance_heatmap(self, matrix=None, hide_diagonal=True, show=True):
        matrix = matrix if matrix is not None else self.global_alliance_matrix_
        if matrix is None:
            raise ValueError("No global alliance matrix is available. Call explain_global(...) first.")

        plot_matrix = matrix.astype(float).copy()
        if hide_diagonal:
            for feature in plot_matrix.index.intersection(plot_matrix.columns):
                plot_matrix.loc[feature, feature] = np.nan

        fig, ax = plt.subplots(figsize=(max(7, 0.45 * len(matrix.columns)), max(6, 0.45 * len(matrix.index))))
        values = plot_matrix.to_numpy(dtype=float)
        finite_values = values[np.isfinite(values)]
        positive_values = finite_values[finite_values > 0]
        cmap = plt.get_cmap("viridis").copy()
        cmap.set_bad("#f2f2f2")

        if len(positive_values) == 0:
            image = ax.imshow(values, vmin=0, vmax=1, cmap=cmap)
            ax.text(
                0.5,
                0.5,
                "No repeated feature co-occurrence beyond single-feature alliances.\n"
                "Use alliance_mode='auto', 'hybrid', or user_alliances to populate this matrix.",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=11,
                bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "gray"},
            )
        else:
            vmax = max(1.0, float(np.nanmax(values)))
            image = ax.imshow(values, vmin=0, vmax=vmax, cmap=cmap)
        ax.set_xticks(np.arange(len(matrix.columns)))
        ax.set_yticks(np.arange(len(matrix.index)))
        ax.set_xticklabels(matrix.columns, rotation=90)
        ax.set_yticklabels(matrix.index)
        ax.set_title(
            "DhondtXAI Global Alliance Co-occurrence\n"
            "cell = fraction of local explanations where two features share an alliance"
        )
        colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        colorbar.set_label("Co-occurrence frequency")
        fig.tight_layout()
        if show:
            plt.show()
        return fig, ax

    def plot(self, explanation=None, kind="bar", **kwargs):
        """Convenience plot dispatcher for local DhondtXAI explanations."""
        if kind in {"bar", "local_bar", "local"}:
            return self.plot_local_bar(explanation=explanation, **kwargs)
        if kind == "waterfall":
            return self.plot_waterfall(explanation=explanation, **kwargs)
        if kind == "parliament":
            from .plot_parliament import plot_signed_parliament

            explanation = explanation or self.last_explanation
            if explanation is None:
                raise ValueError("No explanation is available. Call explain(...) first.")
            return plot_signed_parliament(explanation, **kwargs)
        raise ValueError("kind must be 'bar', 'waterfall', or 'parliament'.")

    def plot_global(self, kind="importance", **kwargs):
        """Convenience plot dispatcher for global DhondtXAI outputs."""
        if kind == "importance":
            return self.plot_global_importance(**kwargs)
        if kind in {"alliance_heatmap", "heatmap"}:
            return self.plot_global_alliance_heatmap(**kwargs)
        raise ValueError("kind must be 'importance' or 'alliance_heatmap'.")

    def _ensure_frame(self, X):
        if isinstance(X, pd.DataFrame):
            return X.copy()
        if isinstance(X, pd.Series):
            return pd.DataFrame([X.to_dict()])
        frame = pd.DataFrame(X)
        if self.features is not None and len(frame.columns) == len(self.features):
            frame.columns = self.features
        return frame

    def _is_single_input(self, X):
        if isinstance(X, (pd.Series, dict)):
            return True
        if isinstance(X, pd.DataFrame):
            return False
        return np.asarray(X).ndim == 1

    def _single_input_data(self, X):
        series = self._ensure_series(X)
        return series[self.features].to_numpy()

    def _display_name(self, name, language="en", friendly=True):
        if friendly and str(name) == "__other__":
            return "other features"
        if friendly and str(name).startswith("__"):
            return _residual_label(name, language)
        return str(name)

    def _wrap_label(self, label, width=18, max_length=44):
        text = str(label)
        if len(text) > max_length:
            text = text[: max_length - 1] + "..."
        if len(text) <= width:
            return text
        import textwrap

        return "\n".join(textwrap.wrap(text, width=width))

    def _ensure_series(self, x):
        if isinstance(x, pd.Series):
            series = x.copy()
        elif isinstance(x, dict):
            series = pd.Series(x)
        elif isinstance(x, pd.DataFrame):
            if len(x) != 1:
                raise ValueError("x must contain exactly one row.")
            series = x.iloc[0].copy()
        else:
            series = pd.Series(np.asarray(x), index=self.features)

        missing = [feature for feature in self.features if feature not in series.index]
        if missing:
            raise ValueError(f"x is missing features: {missing}")
        return series

    def _baseline(self, class_index=None, target_index=None):
        key = (
            self.output_type,
            self._resolve_output_type(),
            class_index,
            target_index,
            tuple(self.features or []),
            len(self.background_data) if self.background_data is not None else 0,
        )
        if key not in self._baseline_cache:
            self._baseline_cache[key] = float(
                np.mean(
                    self._score_frame(
                        self.background_data[self.features],
                        class_index=class_index,
                        target_index=target_index,
                    )
                )
            )
        return self._baseline_cache[key]

    def _baseline_from_frame(self, frame, class_index=None, target_index=None):
        frame = self._ensure_frame(frame)[self.features]
        return float(np.mean(self._score_frame(frame, class_index=class_index, target_index=target_index)))

    def _score_frame(self, X, class_index=None, target_index=None):
        X = self._ensure_frame(X)[self.features]
        output_type = self._resolve_output_type()
        raw_scores = self._predict_raw_scores(X, output_type)
        if (
            isinstance(raw_scores, (list, tuple))
            and raw_scores
            and any(np.asarray(item).ndim >= 2 for item in raw_scores)
        ):
            raise ValueError(
                "List-of-arrays model outputs are ambiguous. For multi-label models, provide a numeric "
                "score_fn selecting the target label and class explicitly."
            )
        scores = np.asarray(raw_scores)

        if output_type == "logit":
            values = self._select_probability_output(scores, class_index)
            values = np.clip(values, self.eps, 1.0 - self.eps)
            return np.log(values / (1.0 - values))

        if output_type == "decision":
            if scores.ndim == 1:
                classes = self._get_classes()
                if classes is not None and len(classes) == 2:
                    index = self._validate_class_index(class_index if class_index is not None else self.class_index, 2)
                    values = scores if index == 1 else -scores
                    return self._ensure_numeric_output(values, "decision_function")
                return self._ensure_numeric_output(scores, "decision_function")
            return self._select_output(scores, class_index)

        if output_type == "probability":
            return self._select_probability_output(scores, class_index)

        if output_type == "decision":
            return self._select_output(scores, class_index)

        if output_type == "prediction":
            return self._select_output(scores, target_index)

        if output_type == "custom":
            index = target_index if target_index is not None else class_index
            return self._select_output(scores, index)

        raise ValueError("output_type must be auto, probability, logit, decision, prediction, custom.")

    def _raw_model_output(self, model_input):
        return self._predict_prepared_input(model_input, self._resolve_output_type())

    def _predict_raw_scores(self, X, output_type):
        model_input = self._prepare_model_input(X)
        return self._predict_prepared_input(model_input, output_type)

    def _predict_prepared_input(self, model_input, output_type):
        if self.predict_fn is not None:
            return self._apply_output_adapter(self._call_predict(self.predict_fn, model_input))

        if self.model is None:
            raise ValueError("No model or score_fn is available for scoring.")

        adapter = self._resolve_model_adapter()
        if output_type in {"probability", "logit"}:
            if hasattr(self.model, "predict_proba"):
                return self._apply_output_adapter(self._call_predict(self.model.predict_proba, model_input))
            if adapter in {"xgboost", "lightgbm", "catboost", "torch", "keras"}:
                return self._apply_output_adapter(self._adapter_predict(model_input, adapter))
            raise ValueError(
                "output_type='probability' or 'logit' requires a probability-capable model, "
                "a compatible model_adapter, or score_fn."
            )
        if output_type == "decision":
            if not hasattr(self.model, "decision_function"):
                if adapter in {"xgboost", "lightgbm", "catboost", "torch", "keras"}:
                    return self._apply_output_adapter(self._adapter_predict(model_input, adapter))
                raise ValueError("output_type='decision' requires decision_function or a compatible adapter.")
            return self._apply_output_adapter(self._call_predict(self.model.decision_function, model_input))
        if output_type == "prediction":
            if adapter in {"xgboost", "lightgbm", "catboost", "torch", "keras"}:
                return self._apply_output_adapter(self._adapter_predict(model_input, adapter))
            if not hasattr(self.model, "predict"):
                raise ValueError("output_type='prediction' requires predict or a compatible adapter.")
            return self._apply_output_adapter(self._call_predict(self.model.predict, model_input))
        if output_type == "custom":
            if adapter in {"xgboost", "lightgbm", "catboost", "torch", "keras"}:
                return self._apply_output_adapter(self._adapter_predict(model_input, adapter))
            raise ValueError("output_type='custom' requires score_fn or a compatible adapter.")
        raise ValueError("output_type must be auto, probability, logit, decision, prediction, custom.")

    def _resolve_output_type(self):
        if self.output_type != "auto":
            return self.output_type
        if self.predict_fn is not None:
            return "custom"
        adapter = self._resolve_model_adapter()
        if self.model is not None and hasattr(self.model, "predict_proba"):
            return "probability"
        if self.model is not None and hasattr(self.model, "decision_function"):
            return "decision"
        if self.task == "classification" and adapter in {"xgboost", "lightgbm", "catboost", "torch", "keras"}:
            return "probability"
        return "prediction"

    def _prepare_model_input(self, X):
        if self.input_adapter is not None:
            return self.input_adapter(X.copy())
        adapter = self._resolve_model_adapter()
        if self.input_format == "auto":
            if adapter == "xgboost" and self._is_native_xgboost_booster():
                return self._xgboost_dmatrix(X)
            if adapter == "torch":
                return self._torch_tensor(X)
            if adapter == "keras":
                return X.to_numpy(dtype=float)
            return X
        if self.input_format == "numpy":
            return X.to_numpy()
        return X

    def _resolve_model_adapter(self):
        if self.model_adapter != "auto":
            return self.model_adapter
        if self.predict_fn is not None:
            return "callable"
        if self.model is None:
            return "sklearn"

        module = type(self.model).__module__.lower()
        name = type(self.model).__name__.lower()
        if "xgboost" in module and name == "booster":
            return "xgboost"
        if "lightgbm" in module:
            return "lightgbm"
        if "catboost" in module:
            return "catboost"
        if "torch" in module or (hasattr(self.model, "forward") and hasattr(self.model, "eval")):
            return "torch"
        if "keras" in module or "tensorflow" in module:
            return "keras"
        return "sklearn"

    def _is_native_xgboost_booster(self):
        if self.model is None:
            return False
        module = type(self.model).__module__.lower()
        name = type(self.model).__name__.lower()
        return "xgboost" in module and name == "booster"

    def _adapter_predict(self, model_input, adapter):
        if adapter == "xgboost":
            return self._call_predict(self.model.predict, model_input)
        if adapter == "lightgbm":
            return self._call_predict(self.model.predict, model_input)
        if adapter == "catboost":
            if hasattr(self.model, "predict_proba") and self._resolve_output_type() in {"auto", "probability", "logit"}:
                return self._call_predict(self.model.predict_proba, model_input)
            return self._call_predict(self.model.predict, model_input)
        if adapter == "torch":
            return self._torch_predict(model_input)
        if adapter == "keras":
            try:
                return self._call_predict(self.model.predict, model_input, verbose=0)
            except TypeError:
                return self._call_predict(self.model.predict, model_input)
        if hasattr(self.model, "predict"):
            return self._call_predict(self.model.predict, model_input)
        raise ValueError(f"Unsupported model_adapter={adapter!r}. Provide score_fn or input_adapter.")

    def _call_predict(self, function, model_input, **extra_kwargs):
        kwargs = dict(self.predict_kwargs)
        kwargs.update(extra_kwargs)
        if kwargs:
            return function(model_input, **kwargs)
        return function(model_input)

    def _apply_output_adapter(self, output):
        if self.output_adapter is None:
            return output
        return self.output_adapter(output)

    def _xgboost_dmatrix(self, X):
        try:
            import xgboost as xgb
        except ImportError as exc:
            raise ImportError("model_adapter='xgboost' requires the xgboost package.") from exc
        return xgb.DMatrix(X.to_numpy(), feature_names=[str(feature) for feature in X.columns])

    def _torch_tensor(self, X):
        try:
            import torch
        except ImportError as exc:
            raise ImportError("model_adapter='torch' requires the torch package.") from exc
        device = None
        if self.model is not None and hasattr(self.model, "parameters"):
            try:
                first_parameter = next(self.model.parameters())
                device = first_parameter.device
            except StopIteration:
                device = None
        return torch.as_tensor(X.to_numpy(dtype=float), dtype=torch.float32, device=device)

    def _torch_predict(self, model_input):
        try:
            import torch
        except ImportError as exc:
            raise ImportError("model_adapter='torch' requires the torch package.") from exc
        was_training = getattr(self.model, "training", False)
        self.model.eval()
        try:
            with torch.no_grad():
                output = self.model(model_input)
        finally:
            if was_training:
                self.model.train()
        if hasattr(output, "detach"):
            output = output.detach().cpu().numpy()
        return output

    def _select_output(self, scores, index=None):
        scores = np.asarray(scores)
        if scores.ndim == 0:
            scores = scores.reshape(1)
        if scores.ndim == 1:
            return self._ensure_numeric_output(scores, "model output")
        if scores.ndim == 2:
            selected_index = 0 if index is None else index
            selected_index = self._validate_class_index(selected_index, scores.shape[1])
            return self._ensure_numeric_output(scores[:, selected_index], "model output")
        raise ValueError("Model output must be 1D or 2D numeric scores.")

    def _select_probability_output(self, scores, class_index=None):
        scores = np.asarray(scores)
        if scores.ndim == 0:
            scores = scores.reshape(1)

        selected_index = self.class_index if class_index is None else class_index
        if selected_index == "predicted":
            raise ValueError("class_index='predicted' must be resolved before probability output selection.")
        if isinstance(selected_index, str):
            selected_index = self._class_index_from_label(selected_index)

        if scores.ndim == 1:
            probabilities = self._ensure_numeric_output(scores, "probability output")
            self._validate_probability_values(probabilities)
            if selected_index in (None, 1):
                return probabilities
            if selected_index == 0:
                return 1.0 - probabilities
            raise ValueError("1D binary probability output supports class_index 0 or 1.")

        if scores.ndim == 2 and scores.shape[1] == 1:
            probabilities = self._ensure_numeric_output(scores[:, 0], "probability output")
            self._validate_probability_values(probabilities)
            if selected_index in (None, 1):
                return probabilities
            if selected_index == 0:
                return 1.0 - probabilities
            raise ValueError("Single-column binary probability output supports class_index 0 or 1.")

        if scores.ndim == 2:
            if selected_index is None and scores.shape[1] == 2:
                selected_index = 1
            probabilities = self._select_output(scores, selected_index)
            self._validate_probability_values(probabilities)
            return probabilities

        raise ValueError("Probability output must be 1D or 2D numeric scores.")

    def _validate_probability_values(self, values):
        if not self.validate_probability:
            return
        values = np.asarray(values, dtype=float)
        tolerance = self.probability_tolerance
        if np.any(values < -tolerance) or np.any(values > 1.0 + tolerance):
            raise ValueError(
                "output_type='probability' was selected, but the model returned values outside [0, 1]. "
                "The model may be returning logits or raw margins. Use an output_adapter that applies "
                "sigmoid/softmax, or choose output_type='prediction' or output_type='decision'."
            )

    def _ensure_numeric_output(self, values, source):
        try:
            output = np.asarray(values, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{source} must return numeric scores. For classifiers, use output_type='probability', "
                "'logit', 'decision', or provide a numeric score_fn."
            ) from exc
        if not np.all(np.isfinite(output)):
            raise ValueError(f"{source} returned NaN or infinite scores.")
        return output

    def _validate_class_index(self, class_index, class_count):
        if class_index == "predicted":
            raise ValueError("class_index='predicted' must be resolved before output selection.")
        if isinstance(class_index, str):
            class_index = self._class_index_from_label(class_index)
        if class_index < 0 or class_index >= class_count:
            raise ValueError(f"class_index={class_index} is outside the valid range [0, {class_count - 1}].")
        return int(class_index)

    def _class_index_from_label(self, label):
        classes = self._get_classes()
        if classes is None:
            raise ValueError("Class-label targets require model.classes_ or a pipeline final step with classes_.")
        classes = list(classes)
        if label not in classes:
            raise ValueError(f"Unknown class label {label!r}. Available labels: {classes}")
        return int(classes.index(label))

    def _resolve_class_index_for_x(self, X, class_index, target_index=None):
        if class_index != "predicted":
            return class_index

        output_type = self._resolve_output_type()
        X = self._ensure_frame(X)[self.features]
        model_input = self._prepare_model_input(X)

        if self.predict_fn is not None:
            scores = np.asarray(self._predict_prepared_input(model_input, output_type))
        elif output_type in {"probability", "logit"}:
            if not hasattr(self.model, "predict_proba"):
                adapter = self._resolve_model_adapter()
                if adapter in {"xgboost", "lightgbm", "catboost", "torch", "keras"}:
                    scores = np.asarray(self._predict_prepared_input(model_input, output_type))
                else:
                    raise ValueError(
                        "class_index='predicted' with probability/logit requires a probability-capable model."
                    )
            else:
                scores = np.asarray(self._predict_prepared_input(model_input, output_type))
        elif output_type == "decision":
            if not hasattr(self.model, "decision_function"):
                adapter = self._resolve_model_adapter()
                if adapter in {"xgboost", "lightgbm", "catboost", "torch", "keras"}:
                    scores = np.asarray(self._predict_prepared_input(model_input, output_type))
                else:
                    raise ValueError("class_index='predicted' with decision output requires decision_function.")
            else:
                scores = np.asarray(self._predict_prepared_input(model_input, output_type))
        elif output_type == "custom" and self.predict_fn is not None:
            scores = np.asarray(self._predict_prepared_input(model_input, output_type))
        else:
            raise ValueError(
                "class_index='predicted' requires probability, logit, decision, or a 2D custom score_fn output."
            )

        if output_type in {"probability", "logit"} and scores.ndim == 2 and scores.shape[1] == 1:
            return 1 if float(scores[0, 0]) >= 0.5 else 0
        if scores.ndim == 2:
            return int(np.argmax(scores[0]))
        if output_type in {"probability", "logit"} and scores.ndim == 1:
            return 1 if float(scores[0]) >= 0.5 else 0
        if scores.ndim == 1:
            classes = self._get_classes()
            if classes is not None and len(classes) == 2:
                return 1 if float(scores[0]) >= 0 else 0
        raise ValueError("class_index='predicted' requires a 2D score matrix or binary decision scores.")

    def _normalize_class_index_for_output(self, X, class_index, target_index=None):
        if isinstance(class_index, str) and class_index != "predicted":
            return self._class_index_from_label(class_index)
        if class_index is not None:
            return class_index

        output_type = self._resolve_output_type()
        if output_type in {"prediction", "custom"} and target_index is not None:
            return None
        if output_type not in {"probability", "logit", "decision"}:
            return None

        X = self._ensure_frame(X)[self.features]
        try:
            scores = np.asarray(self._predict_raw_scores(X.head(1), output_type))
        except Exception:
            return class_index

        if output_type in {"probability", "logit"}:
            if scores.ndim == 1 or (scores.ndim == 2 and scores.shape[1] <= 2):
                return 1
            if scores.ndim == 2:
                return 0
        if output_type == "decision":
            classes = self._get_classes()
            if scores.ndim == 1 and classes is not None and len(classes) == 2:
                return 1
            if scores.ndim == 2:
                return 0
        return class_index

    def _warn_if_implicit_multiclass(self, X, class_index, target_index=None, target_explicit=False):
        if self._target_explicit or target_explicit or self._warned_implicit_multiclass:
            return
        output_type = self._resolve_output_type()
        if output_type not in {"probability", "logit"}:
            return
        try:
            scores = np.asarray(self._predict_raw_scores(self._ensure_frame(X).head(1), output_type))
        except Exception:
            return
        if scores.ndim == 2 and scores.shape[1] > 2:
            warnings.warn(
                "Multiclass output detected; default class_index=1 is being explained. "
                "Pass target='predicted', target=<class_label>, or class_index=... explicitly.",
                UserWarning,
                stacklevel=3,
            )
            self._warned_implicit_multiclass = True

    def _class_label(self, class_index):
        if class_index is None or class_index == "predicted":
            return None
        if isinstance(class_index, str):
            return class_index
        classes = self._get_classes()
        if classes is not None:
            classes = list(classes)
            if 0 <= class_index < len(classes):
                return classes[class_index]
        return None

    def _get_classes(self):
        if self.model is None:
            return None
        if hasattr(self.model, "classes_"):
            return self.model.classes_
        if hasattr(self.model, "steps") and self.model.steps:
            last_step = self.model.steps[-1][1]
            if hasattr(last_step, "classes_"):
                return last_step.classes_
        return None

    def _removal_effect(
        self,
        x_series,
        group,
        original_score,
        background,
        class_index,
        target_index=None,
        perturbation=None,
        perturbation_sampler=None,
    ):
        effects = self._batch_removal_effects(
            x_series,
            [tuple(group)],
            original_score,
            background,
            class_index,
            target_index,
            perturbation=perturbation,
            perturbation_sampler=perturbation_sampler,
        )
        return effects[tuple(group)]

    def _batch_removal_effects(
        self,
        x_series,
        groups,
        original_score,
        background,
        class_index,
        target_index=None,
        perturbation=None,
        perturbation_sampler=None,
        context_features=None,
    ):
        perturbation = self.perturbation if perturbation is None else perturbation
        perturbation_sampler = self.perturbation_sampler if perturbation_sampler is None else perturbation_sampler
        groups = [tuple(group) for group in groups]
        if not groups:
            return {}
        context_features = tuple(self.features if context_features is None else context_features)
        context_set = set(context_features)
        omitted_features = tuple(feature for feature in self.features if feature not in context_set)
        removal_groups = [
            tuple(dict.fromkeys(list(omitted_features) + list(group)))
            for group in groups
        ]

        if self._can_use_numeric_replacement_fast_path(x_series, background, perturbation, perturbation_sampler):
            batch, sizes = self._numeric_replacement_batch(x_series, removal_groups, background)
            scores = self._score_frame(batch, class_index=class_index, target_index=target_index)
            effects = {}
            start = 0
            for group, size in zip(groups, sizes):
                removed_score = float(np.mean(scores[start:start + size]))
                effects[group] = original_score - removed_score
                start += size
            return effects

        blocks = []
        sizes = []
        for removal_group in removal_groups:
            rows = self._replacement_rows(
                x_series,
                removal_group,
                background,
                perturbation,
                perturbation_sampler=perturbation_sampler,
            )
            blocks.append(rows)
            sizes.append(len(rows))

        batch = pd.concat(blocks, ignore_index=True)
        scores = self._score_frame(batch, class_index=class_index, target_index=target_index)

        effects = {}
        start = 0
        for group, size in zip(groups, sizes):
            removed_score = float(np.mean(scores[start:start + size]))
            effects[group] = original_score - removed_score
            start += size
        return effects

    def _context_score(
        self,
        x_series,
        kept_features,
        background,
        class_index,
        target_index=None,
        perturbation=None,
        perturbation_sampler=None,
    ):
        kept = set(kept_features)
        omitted = tuple(feature for feature in self.features if feature not in kept)
        if not omitted:
            return float(
                self._score_frame(
                    pd.DataFrame([x_series[self.features]]),
                    class_index=class_index,
                    target_index=target_index,
                )[0]
            )
        rows = self._replacement_rows(
            x_series,
            omitted,
            background,
            self.perturbation if perturbation is None else perturbation,
            perturbation_sampler=perturbation_sampler,
        )
        return float(np.mean(self._score_frame(rows, class_index=class_index, target_index=target_index)))

    def _can_use_numeric_replacement_fast_path(self, x_series, background, perturbation, perturbation_sampler):
        if perturbation != "interventional" or perturbation_sampler is not None:
            return False
        try:
            background[self.features].to_numpy(dtype=float)
            x_series[self.features].to_numpy(dtype=float)
        except (TypeError, ValueError):
            return False
        return True

    def _numeric_replacement_batch(self, x_series, groups, background):
        background_values = background[self.features].to_numpy(dtype=float)
        x_values = x_series[self.features].to_numpy(dtype=float)
        row_count = len(background)
        group_count = len(groups)
        batch_values = np.tile(x_values, (row_count * group_count, 1))
        positions = self._feature_positions
        for group_index, group in enumerate(groups):
            start = group_index * row_count
            end = start + row_count
            for feature in group:
                batch_values[start:end, positions[feature]] = background_values[:, positions[feature]]
        return pd.DataFrame(batch_values, columns=self.features), [row_count] * group_count

    def _replacement_rows(self, x_series, group, background, perturbation, perturbation_sampler=None):
        group = tuple(group)
        replacement = self._replacement_background(
            x_series,
            group,
            background,
            perturbation,
            perturbation_sampler=perturbation_sampler,
        )
        rows = pd.DataFrame([x_series[self.features].to_dict()] * len(replacement), columns=self.features)
        for feature in group:
            rows[feature] = replacement[feature].to_numpy()
        return rows

    def _replacement_background(self, x_series, group, background, perturbation, perturbation_sampler=None):
        if perturbation == "interventional":
            return background.reset_index(drop=True)
        if perturbation == "user_sampler":
            sampler = self.perturbation_sampler if perturbation_sampler is None else perturbation_sampler
            if sampler is None:
                raise ValueError("perturbation='user_sampler' requires perturbation_sampler.")
            replacement = sampler(x_series.copy(), tuple(group), background.copy(), len(background))
            replacement = self._ensure_frame(replacement).reset_index(drop=True)
            if len(replacement) != len(background):
                raise ValueError("perturbation_sampler must return exactly n replacement rows.")
            missing = [feature for feature in self.features if feature not in replacement.columns]
            if missing:
                raise ValueError(f"perturbation_sampler output is missing features: {missing}")
            return replacement[self.features]
        if perturbation != "conditional_knn":
            raise ValueError("perturbation must be 'interventional', 'conditional_knn', or 'user_sampler'.")

        complement = [feature for feature in self.features if feature not in set(group)]
        if not complement:
            return background.reset_index(drop=True)

        pool = self.background_data.reset_index(drop=True)
        distances = self._mixed_distance(pool[complement], x_series[complement])
        neighbor_count = min(self.knn_neighbors, len(pool))
        nearest = np.argsort(distances)[:neighbor_count]
        if len(nearest) == 0:
            return background.reset_index(drop=True)

        repeats = int(np.ceil(len(background) / len(nearest)))
        indices = np.tile(nearest, repeats)[: len(background)]
        return pool.iloc[indices].reset_index(drop=True)

    def _mixed_distance(self, frame, row):
        distances = np.zeros(len(frame), dtype=float)
        for column in frame.columns:
            values = frame[column]
            value = row[column]
            if pd.api.types.is_numeric_dtype(values):
                numeric = pd.to_numeric(values, errors="coerce")
                scale = float(numeric.std(ddof=0))
                if not np.isfinite(scale) or scale <= self.eps:
                    scale = 1.0
                distances += ((numeric.to_numpy(dtype=float) - float(value)) / scale) ** 2
            else:
                distances += (values.astype(str).to_numpy() != str(value)).astype(float)
        return distances

    def _sample_background(self, n_background, rng):
        n_background = int(n_background)
        if n_background <= 0:
            raise ValueError("n_background must be positive.")
        size = len(self.background_data)
        replace = n_background > size
        indices = rng.choice(size, size=n_background, replace=replace)
        return self.background_data.iloc[indices].reset_index(drop=True)

    def _validate_feature_names(self, feature_names):
        index = pd.Index(feature_names)
        if not index.is_unique:
            duplicates = index[index.duplicated()].tolist()
            raise ValueError(f"Feature names must be unique. Duplicate feature name(s): {duplicates}")
        reserved = []
        for feature in feature_names:
            text = str(feature)
            if text.startswith("__") or " + " in text:
                reserved.append(feature)
        if reserved:
            raise ValueError(
                "Feature names starting with '__' or containing ' + ' are reserved "
                f"for DhondtXAI residual and alliance display keys: {reserved}"
            )

    def _refresh_feature_positions(self):
        self._feature_positions = {feature: index for index, feature in enumerate(self.features or [])}

    def _normalize_feature_list(self, values, feature_names, strict=None):
        strict = self.strict_features if strict is None else strict
        feature_names = list(feature_names)
        normalized = []
        unknown = []
        for value in values:
            if value is None:
                continue
            if value in feature_names:
                normalized.append(value)
                continue
            if isinstance(value, str):
                value = value.strip()
                if not value:
                    continue
                if value in feature_names:
                    normalized.append(value)
                elif value.isdigit() and self.feature_reference != "name":
                    index = int(value) - 1
                    if 0 <= index < len(feature_names):
                        normalized.append(feature_names[index])
                    else:
                        unknown.append(value)
                else:
                    unknown.append(value)
            elif isinstance(value, int):
                index = value - 1 if value > 0 else value
                if self.feature_reference != "name" and 0 <= index < len(feature_names):
                    normalized.append(feature_names[index])
                else:
                    unknown.append(value)
            else:
                if value in feature_names:
                    normalized.append(value)
                else:
                    unknown.append(value)
        if strict and unknown:
            raise ValueError(f"Unknown feature(s): {unknown}")
        return list(dict.fromkeys(normalized))

    def _normalize_user_alliances(self, alliances, active_features):
        normalized = []
        used = set()
        for group in alliances:
            if isinstance(group, str):
                parts = [part.strip() for part in group.replace("+", " and ").split(" and ")]
            else:
                parts = list(group)
            features = tuple(self._normalize_feature_list(parts, active_features))
            if not features:
                continue
            overlap = used.intersection(features)
            if overlap:
                raise ValueError(f"User alliances must be disjoint. Overlap: {sorted(overlap, key=str)}")
            used.update(features)
            normalized.append(features)
        return normalized

    def _build_alliances(self, active_features, mode, user_alliances, affinity, rho, auto_alliance_method):
        if mode not in {"none", "user", "auto", "hybrid"}:
            raise ValueError("alliance_mode must be none, user, auto, or hybrid.")

        user_groups = self._normalize_user_alliances(user_alliances or [], active_features)
        if mode == "none":
            return [(feature,) for feature in active_features]

        if mode == "user":
            used = {feature for group in user_groups for feature in group}
            return user_groups + [(feature,) for feature in active_features if feature not in used]

        if mode == "auto":
            return self._auto_alliances(active_features, affinity, rho, auto_alliance_method)

        used = {feature for group in user_groups for feature in group}
        remaining = [feature for feature in active_features if feature not in used]
        return user_groups + self._auto_alliances(remaining, affinity, rho, auto_alliance_method)

    def _build_affinity(self, active_features, feature_effects, pair_effects, interactions, affinity_mode):
        affinity = {}
        for left, right in combinations(active_features, 2):
            key = self._pair_key(left, right)
            effect_left = feature_effects[left]
            effect_right = feature_effects[right]
            pair_effect = pair_effects.get(key, 0.0)
            interaction = interactions.get(key, 0.0)
            if affinity_mode == "absolute_interaction":
                denominator = abs(pair_effect) + abs(effect_left) + abs(effect_right) + self.eps
                value = abs(interaction) / denominator
            elif affinity_mode == "same_direction":
                same_direction = np.sign(effect_left) == np.sign(effect_right) and abs(effect_left) > self.eps
                same_direction = same_direction and abs(effect_right) > self.eps
                value = 0.0
                if same_direction:
                    denominator = abs(pair_effect) + abs(effect_left) + abs(effect_right) + self.eps
                    value = abs(interaction) / denominator
            else:
                raise ValueError("affinity_mode must be 'same_direction' or 'absolute_interaction'.")
            affinity[key] = float(np.clip(value, 0.0, 1.0))
        return affinity

    def _connected_components(self, features, affinity, rho):
        if not features:
            return []

        parent = {feature: feature for feature in features}

        def find(value):
            while parent[value] != value:
                parent[value] = parent[parent[value]]
                value = parent[value]
            return value

        def union(left, right):
            root_left = find(left)
            root_right = find(right)
            if root_left != root_right:
                parent[root_right] = root_left

        for left, right in combinations(features, 2):
            key = self._pair_key(left, right)
            if affinity.get(key, 0.0) >= rho:
                union(left, right)

        groups = {}
        for feature in features:
            groups.setdefault(find(feature), []).append(feature)
        return [tuple(group) for group in groups.values()]

    def _auto_alliances(self, features, affinity, rho, method):
        if method == "connected_components":
            return self._connected_components(features, affinity, rho)
        if method == "complete_linkage":
            return self._complete_linkage_components(features, affinity, rho)
        raise ValueError("auto_alliance_method must be connected_components or complete_linkage.")

    def _complete_linkage_components(self, features, affinity, rho):
        clusters = [(feature,) for feature in features]
        edges = sorted(
            (
                (affinity.get(self._pair_key(left, right), 0.0), left, right)
                for left, right in combinations(features, 2)
            ),
            key=lambda item: item[0],
            reverse=True,
        )

        for value, left, right in edges:
            if value < rho:
                break
            left_index = next(index for index, group in enumerate(clusters) if left in group)
            right_index = next(index for index, group in enumerate(clusters) if right in group)
            if left_index == right_index:
                continue
            merged = tuple(clusters[left_index] + clusters[right_index])
            if self._complete_linkage_ok(merged, affinity, rho):
                first, second = sorted((left_index, right_index), reverse=True)
                clusters.pop(first)
                clusters.pop(second)
                clusters.append(merged)
        return clusters

    def _complete_linkage_ok(self, group, affinity, rho):
        for left, right in combinations(group, 2):
            if affinity.get(self._pair_key(left, right), 0.0) < rho:
                return False
        return True

    def _interaction_strength(self, group, interactions):
        if len(group) < 2:
            return 0.0
        value = 0.0
        for left, right in combinations(group, 2):
            value += abs(interactions.get(self._pair_key(left, right), 0.0))
        return value

    def _normalize_threshold(self, threshold, threshold_enabled):
        if threshold_enabled is None:
            threshold_enabled = threshold is not None
        if not threshold_enabled:
            return None, False
        tau = 0.0 if threshold is None else float(threshold)
        if tau > 1:
            tau = tau / 100.0
        if not 0 <= tau < 1:
            raise ValueError("threshold must be in [0, 1) or a percentage in [0, 100).")
        return tau, True

    def _threshold_alliances(self, votes, tau, threshold_enabled):
        names = list(votes.keys())
        if not threshold_enabled:
            return names, []

        total_votes = sum(votes.values())
        shares = {name: (votes[name] / total_votes if total_votes > 0 else 0.0) for name in names}
        eligible = [name for name in names if shares[name] >= tau]
        if not eligible:
            eligible = [max(names, key=lambda name: votes[name])]
        below = [name for name in names if name not in eligible]
        return eligible, below

    def _transfer_matrix(self, alliance_members, votes, eligible, below_threshold, affinity, redistribute):
        transfer = {}
        for source in alliance_members:
            for target in eligible:
                transfer[(source, target)] = 0.0

        for source in eligible:
            transfer[(source, source)] = 1.0

        if not redistribute:
            return transfer

        for source in below_threshold:
            scores = []
            for target in eligible:
                scores.append(self._group_affinity(alliance_members[source], alliance_members[target], affinity))
            scores = np.asarray(scores, dtype=float) + self.eps
            scores = scores / scores.sum()
            for target, weight in zip(eligible, scores):
                transfer[(source, target)] = float(weight)

        return transfer

    def _group_affinity(self, left_group, right_group, affinity):
        values = []
        for left in left_group:
            for right in right_group:
                if left == right:
                    values.append(1.0)
                else:
                    values.append(affinity.get(self._pair_key(left, right), 0.0))
        return float(np.mean(values)) if values else 0.0

    def _dhondt_allocate(self, votes, seats, tie_break=None, rng=None):
        tie_break = self.tie_break if tie_break is None else tie_break
        if tie_break not in {"stable", "random"}:
            raise ValueError("tie_break must be 'stable' or 'random'.")
        votes = np.asarray(votes, dtype=float)
        if not np.all(np.isfinite(votes)):
            raise ValueError("D'Hondt votes must be finite.")
        if np.any(votes < 0):
            raise ValueError("D'Hondt votes must be non-negative.")
        allocation = np.zeros(len(votes), dtype=int)
        if seats <= 0 or len(votes) == 0 or not np.any(votes > 0):
            return allocation
        rng = np.random.default_rng(self.random_state) if rng is None else rng
        tie_keys = np.arange(len(votes), dtype=float) if tie_break == "stable" else rng.random(len(votes))
        heap = []
        for index, vote in enumerate(votes):
            heapq.heappush(heap, (-(vote / 1.0), float(tie_keys[index]), index))
        for _ in range(int(seats)):
            _, tie_key, index = heapq.heappop(heap)
            allocation[index] += 1
            next_quotient = votes[index] / (allocation[index] + 1)
            heapq.heappush(heap, (-next_quotient, tie_key, index))
        return allocation

    def _split_signed_seats(self, seats, total_positive, total_negative, delta):
        total = total_positive + total_negative
        seats = int(seats)
        if total > 0:
            positive_count = int(np.floor(seats * total_positive / total + 0.5))
        else:
            positive_count = seats if delta >= 0 else 0
        return positive_count, seats - positive_count

    def _features_from_alliances(self, alliance_names, alliance_members):
        features = []
        for name in alliance_names:
            for feature in alliance_members[name]:
                if feature not in features:
                    features.append(feature)
        return tuple(features)

    def _alliance_sign_conflicts(self, alliance_members, feature_attributions):
        conflicts = {}
        for name, members in alliance_members.items():
            values = [feature_attributions.get(feature, 0.0) for feature in members]
            has_positive = any(value > self.eps for value in values)
            has_negative = any(value < -self.eps for value in values)
            conflicts[name] = bool(has_positive and has_negative)
        return conflicts

    def _project_values(
        self,
        raw_values,
        target,
        names,
        mode="redistribute",
        residual_name=None,
        residual_threshold=0.10,
    ):
        if mode not in {"auto", "redistribute", "residual"}:
            raise ValueError("projection mode must be 'auto', 'redistribute', or 'residual'.")
        residual_threshold = float(residual_threshold)
        if residual_threshold < 0:
            raise ValueError("residual_threshold must be non-negative.")
        projected = {name: 0.0 for name in raw_values}
        names = list(names)
        if not names:
            return projected, 0.0, target, 0.0, target if residual_name else 0.0

        raw_sum = sum(raw_values.get(name, 0.0) for name in names)
        residual = target - raw_sum
        raw_abs_sum = sum(abs(raw_values.get(name, 0.0)) for name in names)
        denominator = max(abs(target), raw_abs_sum, self.eps)
        ratio = abs(residual) / denominator
        use_residual_bucket = mode == "residual" or (
            mode == "auto"
            and (
                (raw_abs_sum <= self.eps and abs(target) > self.eps)
                or ratio >= residual_threshold
            )
        )
        if use_residual_bucket:
            for name in names:
                projected[name] = raw_values.get(name, 0.0)
            return projected, raw_sum, residual, ratio, residual

        magnitudes = np.asarray([abs(raw_values.get(name, 0.0)) + self.eps for name in names], dtype=float)
        weights = magnitudes / magnitudes.sum()
        for name, weight in zip(names, weights):
            projected[name] = raw_values.get(name, 0.0) + float(weight) * residual

        return projected, raw_sum, residual, ratio, 0.0

    def _back_project_sources_signed(
        self,
        represented_positive_raw,
        represented_negative_raw,
        alliance_members,
        positive_votes,
        negative_votes,
        eligible,
        transfer,
    ):
        source_attributions = {name: 0.0 for name in alliance_members}
        for target in eligible:
            positive_mass = []
            positive_sources = []
            negative_mass = []
            negative_sources = []

            for source in alliance_members:
                transfer_weight = transfer.get((source, target), 0.0)
                if transfer_weight <= 0:
                    continue
                pos_weight = transfer_weight * positive_votes[source]
                neg_weight = transfer_weight * negative_votes[source]
                if pos_weight > self.eps:
                    positive_sources.append(source)
                    positive_mass.append(pos_weight)
                if neg_weight > self.eps:
                    negative_sources.append(source)
                    negative_mass.append(neg_weight)

            if positive_mass:
                weights = np.asarray(positive_mass, dtype=float)
                weights = weights / weights.sum()
                for source, weight in zip(positive_sources, weights):
                    source_attributions[source] += float(weight) * represented_positive_raw.get(target, 0.0)

            if negative_mass:
                weights = np.asarray(negative_mass, dtype=float)
                weights = weights / weights.sum()
                for source, weight in zip(negative_sources, weights):
                    source_attributions[source] -= float(weight) * represented_negative_raw.get(target, 0.0)

        return source_attributions

    def _distribute_to_features(
        self,
        source_attributions,
        alliance_members,
        feature_effects,
        interactions,
        lambda_interaction,
        beta,
    ):
        feature_attributions = {}
        for alliance, members in alliance_members.items():
            value = source_attributions.get(alliance, 0.0)
            if len(members) == 1:
                feature_attributions[members[0]] = feature_attributions.get(members[0], 0.0) + value
                continue

            raw_values = {}
            for feature in members:
                interaction_sum = 0.0
                for other in members:
                    if other == feature:
                        continue
                    interaction_sum += abs(interactions.get(self._pair_key(feature, other), 0.0))
                effect = feature_effects.get(feature, 0.0)
                sign = np.sign(effect)
                if sign == 0 and abs(value) > self.eps:
                    sign = np.sign(value)
                raw_values[feature] = effect + lambda_interaction * sign * interaction_sum

            projected_values = self._project_with_custom_weights(raw_values, value, beta)
            for feature, projected in projected_values.items():
                feature_attributions[feature] = feature_attributions.get(feature, 0.0) + projected
        return feature_attributions

    def _project_with_custom_weights(self, raw_values, target, beta):
        names = list(raw_values.keys())
        raw_sum = sum(raw_values.values())
        residual = target - raw_sum
        weights = np.asarray([(abs(raw_values[name]) + self.eps) ** beta for name in names], dtype=float)
        weights = weights / weights.sum()
        return {
            name: raw_values[name] + float(weight) * residual
            for name, weight in zip(names, weights)
        }

    def _threshold_survival(self, feature, explanations):
        if not explanations:
            return 0.0
        count = 0
        for explanation in explanations:
            source = explanation.feature_source_alliance.get(feature)
            if source in explanation.eligible_alliances:
                count += 1
        return count / len(explanations)

    def _compute_global_alliance_matrix(self, explanations):
        matrix = pd.DataFrame(0.0, index=self.features, columns=self.features)
        if not explanations:
            return matrix

        for explanation in explanations:
            for members in explanation.alliance_members.values():
                for left in members:
                    for right in members:
                        if left in matrix.index and right in matrix.columns:
                            matrix.loc[left, right] += 1.0
        return matrix / len(explanations)

    def _alliance_name(self, group):
        return group[0] if len(group) == 1 else " + ".join(map(str, group))

    def _feature_order_key(self, feature):
        if self.features is None:
            return (0, str(type(feature)), str(feature))
        positions = getattr(self, "_feature_positions", {})
        if feature in positions:
            return (0, positions[feature], "")
        return (1, str(type(feature)), str(feature))

    def _pair_key(self, left, right):
        return tuple(sorted((left, right), key=self._feature_order_key))

    def _group_key(self, group):
        return tuple(sorted(tuple(group), key=self._feature_order_key))
