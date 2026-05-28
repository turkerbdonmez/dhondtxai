from dataclasses import dataclass
import heapq
from itertools import combinations
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


@dataclass
class DhondtExplanation:
    """Container for one local DhondtXAI explanation."""

    score: float
    baseline: float
    delta: float
    active_delta: float
    feature_attributions: dict
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
    effects: dict
    interactions: dict
    alliance_members: dict
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
        sign_flip_count = int(
            self.to_feature_frame(include_residuals=False)["sign_consistent"].eq(False).sum()
        )
        if self.projection_residual_ratio < 0.10:
            quality = "high"
        elif self.projection_residual_ratio < 0.50:
            quality = "medium"
        else:
            quality = "caution"

        return {
            "completeness_error": completeness_error,
            "projection_residual_ratio": self.projection_residual_ratio,
            "projection_residual": self.projection_residual,
            "excluded_residual_ratio": excluded_ratio,
            "below_threshold_residual_ratio": below_threshold_ratio,
            "sign_flip_count": sign_flip_count,
            "quality": quality,
        }

    def _target_label(self):
        if self.resolved_output_type in {"probability", "logit", "decision"}:
            return f"class={self.class_label}" if self.class_label is not None else f"class_index={self.class_index}"
        if self.target_index is None:
            return "prediction"
        return f"target_index={self.target_index}"

    def _projection_warning(self, language):
        if self.projection_residual_ratio < 0.10:
            return None
        if language == "tr":
            level = "orta" if self.projection_residual_ratio < 0.50 else "yüksek"
            return (
                f"Uyarı: Projeksiyon düzeltmesi {level} düzeyde. "
                "Atıfları temkinli yorumlayın."
            )
        level = "medium" if self.projection_residual_ratio < 0.50 else "high"
        return (
            f"Warning: Projection correction is {level}. "
            "Interpret attributions cautiously."
        )

    def summary(self, top_k=5, language="en"):
        if language not in {"en", "tr"}:
            raise ValueError("language must be 'en' or 'tr'.")

        frame = self.to_feature_frame(include_residuals=True)
        supporting = frame[frame["attribution"] > 0].head(top_k)
        opposing = frame[frame["attribution"] < 0].head(top_k)
        diagnostics = self.diagnostics()
        target = self._target_label()
        output_label = self.output_type
        if self.resolved_output_type != self.output_type:
            output_label = f"{self.output_type} resolved as {self.resolved_output_type}"

        if language == "tr":
            lines = [
                "DhondtXAI Yerel Açıklama Raporu",
                "--------------------------------",
                f"Açıklanan hedef: {target}, output_type={output_label}",
                f"Model skoru: {self.score:.6f}",
                f"Baseline: {self.baseline:.6f}",
                f"Fark: {self.delta:.6f}",
                "",
                f"Ana yorum: model skoru baseline değerinden {abs(self.delta):.6f} "
                f"{'yüksek' if self.delta >= 0 else 'düşük'}.",
                "",
                "En güçlü destekleyen özellikler:",
            ]
            lines.extend(self._format_ranked_lines(supporting, empty_text="Yok"))
            lines.append("")
            lines.append("En güçlü karşı kanıtlar:")
            lines.extend(self._format_ranked_lines(opposing, empty_text="Yok"))
            lines.extend(
                [
                    "",
                    "Diagnostics:",
                    f"Completeness error: {diagnostics['completeness_error']:.6g}",
                    f"Projection residual ratio: {diagnostics['projection_residual_ratio']:.6f}",
                    f"Projection residual: {diagnostics['projection_residual']:.6g}",
                    f"Threshold: {self.threshold if self.threshold is not None else 'disabled'}",
                    f"Redistribution: {'enabled' if self.redistribution else 'disabled'}",
                    f"Excluded residual: {self.excluded_residual:.6g}",
                    f"Below-threshold residual: {self.below_threshold_residual:.6g}",
                    f"Sign flip count: {diagnostics['sign_flip_count']}",
                    f"Quality: {diagnostics['quality']}",
                ]
            )
            warning = self._projection_warning(language)
            if warning is not None:
                lines.extend(["", warning])
            return "\n".join(lines)

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
                f"Projection residual: {diagnostics['projection_residual']:.6g}",
                f"Threshold: {self.threshold if self.threshold is not None else 'disabled'}",
                f"Redistribution: {'enabled' if self.redistribution else 'disabled'}",
                f"Excluded residual: {self.excluded_residual:.6g}",
                f"Below-threshold residual: {self.below_threshold_residual:.6g}",
                f"Sign flip count: {diagnostics['sign_flip_count']}",
                f"Quality: {diagnostics['quality']}",
            ]
        )
        warning = self._projection_warning(language)
        if warning is not None:
            lines.extend(["", warning])
        return "\n".join(lines)

    def report(self, top_k=5, language="en"):
        return self.summary(top_k=top_k, language=language)

    def _direction_label(self, value):
        if value > 0:
            return "supports prediction"
        if value < 0:
            return "opposes prediction"
        return "neutral"

    def _format_ranked_lines(self, frame, empty_text):
        if frame.empty:
            return [f"- {empty_text}"]
        return [
            f"{idx}. {row.feature}: {row.attribution:+.6f}"
            for idx, row in enumerate(frame.itertuples(index=False), start=1)
        ]


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
        predict_fn=None,
        background_data=None,
        output_type="auto",
        class_index=1,
        target_index=None,
        input_format="dataframe",
        input_adapter=None,
        perturbation="interventional",
        perturbation_sampler=None,
        knn_neighbors=25,
        affinity_mode="same_direction",
        tie_break="stable",
        feature_reference="auto",
        eps=1e-12,
        random_state=42,
        strict_features=True,
    ):
        if predict_fn is None and callable(model) and not any(
            hasattr(model, attr) for attr in ("predict", "predict_proba", "decision_function")
        ):
            predict_fn = model
            model = None

        if model is None and predict_fn is None:
            raise ValueError("Provide either model or predict_fn.")
        if input_format not in {"dataframe", "numpy"}:
            raise ValueError("input_format must be 'dataframe' or 'numpy'.")
        if perturbation not in {"interventional", "conditional_knn", "user_sampler"}:
            raise ValueError("perturbation must be 'interventional', 'conditional_knn', or 'user_sampler'.")
        if perturbation == "user_sampler" and perturbation_sampler is None:
            raise ValueError("perturbation='user_sampler' requires perturbation_sampler.")
        if affinity_mode not in {"same_direction", "absolute_interaction"}:
            raise ValueError("affinity_mode must be 'same_direction' or 'absolute_interaction'.")
        if tie_break not in {"stable", "random"}:
            raise ValueError("tie_break must be 'stable' or 'random'.")
        if feature_reference not in {"auto", "name", "position"}:
            raise ValueError("feature_reference must be 'auto', 'name', or 'position'.")
        if int(knn_neighbors) <= 0:
            raise ValueError("knn_neighbors must be positive.")

        self.model = model
        self.predict_fn = predict_fn
        self.background_data = None
        self.output_type = output_type
        self.class_index = class_index
        self.target_index = target_index
        self.input_format = input_format
        self.input_adapter = input_adapter
        self.perturbation = perturbation
        self.perturbation_sampler = perturbation_sampler
        self.knn_neighbors = int(knn_neighbors)
        self.affinity_mode = affinity_mode
        self.tie_break = tie_break
        self.feature_reference = feature_reference
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

        if background_data is not None:
            self.background_data = self._ensure_frame(background_data)
            self.features = list(self.background_data.columns)
            self._validate_feature_names(self.features)

    def fit(self, X_train, y_train=None, fit_model=True):
        X_train = self._ensure_frame(X_train)
        if fit_model and y_train is not None:
            if self.model is None:
                raise ValueError("Cannot fit a model when model=None. Provide a model or set fit_model=False.")
            self.model.fit(X_train, y_train)

        self.features = list(X_train.columns)
        self._validate_feature_names(self.features)
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
        n_background=100,
        lambda_interaction=0.0,
        rho=0.5,
        beta=1.0,
        auto_alliance_method="connected_components",
        perturbation=None,
        perturbation_sampler=None,
        affinity_mode=None,
        tie_break=None,
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
        if not 0 <= rho <= 1:
            raise ValueError("rho must be in [0, 1].")

        x_series = self._ensure_series(x)
        class_index = self.class_index if class_index is None else class_index
        target_index = self.target_index if target_index is None else target_index
        perturbation = self.perturbation if perturbation is None else perturbation
        perturbation_sampler = self.perturbation_sampler if perturbation_sampler is None else perturbation_sampler
        affinity_mode = self.affinity_mode if affinity_mode is None else affinity_mode
        tie_break = self.tie_break if tie_break is None else tie_break
        if perturbation not in {"interventional", "conditional_knn", "user_sampler"}:
            raise ValueError("perturbation must be 'interventional', 'conditional_knn', or 'user_sampler'.")
        if perturbation == "user_sampler" and perturbation_sampler is None:
            raise ValueError("perturbation='user_sampler' requires perturbation_sampler.")
        if affinity_mode not in {"same_direction", "absolute_interaction"}:
            raise ValueError("affinity_mode must be 'same_direction' or 'absolute_interaction'.")
        if tie_break not in {"stable", "random"}:
            raise ValueError("tie_break must be 'stable' or 'random'.")
        rng = np.random.default_rng(self.random_state if random_state is None else random_state)

        excluded = self._normalize_feature_list(exclude_features or [], self.features)
        active_features = [feature for feature in self.features if feature not in excluded]
        if not active_features:
            raise ValueError("No active features remain after exclusions.")
        allocation_seats = (
            max(5000, 100 * len(active_features), seats)
            if allocation_seats is None
            else allocation_seats
        )
        if allocation_seats <= 0:
            raise ValueError("allocation_seats must be a positive integer.")
        background_sample = self._sample_background(n_background, rng)
        x_frame = pd.DataFrame([x_series[self.features]])
        class_index = self._resolve_class_index_for_x(x_frame, class_index, target_index)

        score = float(
            self._score_frame(
                x_frame,
                class_index=class_index,
                target_index=target_index,
            )[0]
        )
        resolved_output_type = self._resolve_output_type()
        baseline = self._baseline(class_index=class_index, target_index=target_index)
        delta = score - baseline

        single_effects = self._batch_removal_effects(
            x_series,
            [(feature,) for feature in active_features],
            score,
            background_sample,
            class_index,
            target_index,
            perturbation=perturbation,
            perturbation_sampler=perturbation_sampler,
        )
        feature_effects = {feature: single_effects[(feature,)] for feature in active_features}

        need_interactions = alliance_mode in {"auto", "hybrid"} or lambda_interaction > 0
        pair_effects = {}
        interactions = {}
        if need_interactions and len(active_features) > 1:
            pair_groups = [self._pair_key(left, right) for left, right in combinations(active_features, 2)]
            pair_effects = self._batch_removal_effects(
                x_series,
                pair_groups,
                score,
                background_sample,
                class_index,
                target_index,
                perturbation=perturbation,
                perturbation_sampler=perturbation_sampler,
            )
            for left, right in combinations(active_features, 2):
                key = self._pair_key(left, right)
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
        alliance_groups_to_score = []
        for group in alliances:
            if len(group) > 1:
                key = self._group_key(group)
                if key not in pair_effects:
                    alliance_groups_to_score.append(group)
        alliance_group_effects = self._batch_removal_effects(
            x_series,
            alliance_groups_to_score,
            score,
            background_sample,
            class_index,
            target_index,
            perturbation=perturbation,
            perturbation_sampler=perturbation_sampler,
        ) if alliance_groups_to_score else {}
        for group in alliances:
            name = self._alliance_name(group)
            if len(group) == 1:
                effect = feature_effects[group[0]]
            else:
                key = self._group_key(group)
                effect = pair_effects.get(key, alliance_group_effects.get(tuple(group)))
            alliance_effects[name] = effect
            chi[name] = self._interaction_strength(group, interactions)
            votes[name] = abs(effect) + lambda_interaction * chi[name]
            positive_votes[name] = votes[name] if effect > 0 else 0.0
            negative_votes[name] = votes[name] if effect < 0 else 0.0

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
        for target in eligible:
            updated_positive[target] = 0.0
            updated_negative[target] = 0.0
            for source in alliance_members:
                weight = transfer.get((source, target), 0.0)
                updated_positive[target] += weight * positive_votes[source]
                updated_negative[target] += weight * negative_votes[source]

        total_positive = sum(updated_positive.values())
        total_negative = sum(updated_negative.values())
        allocation_positive_count, allocation_negative_count = self._split_signed_seats(
            allocation_seats, total_positive, total_negative, delta
        )
        display_positive_count, display_negative_count = self._split_signed_seats(
            seats, total_positive, total_negative, delta
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

        active_delta = (
            delta
            if not excluded
            else self._removal_effect(
                x_series, active_features, score, background_sample, class_index, target_index,
                perturbation=perturbation, perturbation_sampler=perturbation_sampler
            )
        )
        if threshold_is_enabled and not redistribute:
            eligible_features = self._features_from_alliances(eligible, alliance_members)
            projection_target = (
                active_delta
                if set(eligible_features) == set(active_features)
                else self._removal_effect(
                    x_series, eligible_features, score, background_sample, class_index, target_index,
                    perturbation=perturbation, perturbation_sampler=perturbation_sampler
                )
            )
            projected_source_names = eligible
        else:
            projection_target = active_delta
            projected_source_names = list(alliance_members.keys())

        excluded_residual = delta - active_delta
        below_threshold_residual = active_delta - projection_target

        represented_attributions, _, _, _ = self._project_values(
            represented_raw_values, projection_target, eligible
        )
        source_raw_values = self._back_project_sources_signed(
            represented_positive_raw=represented_positive_raw,
            represented_negative_raw=represented_negative_raw,
            alliance_members=alliance_members,
            positive_votes=positive_votes,
            negative_votes=negative_votes,
            eligible=eligible,
            transfer=transfer,
        )
        source_attributions, raw_sum, projection_residual, projection_residual_ratio = self._project_values(
            source_raw_values, projection_target, projected_source_names
        )
        for name in alliance_members:
            source_attributions.setdefault(name, 0.0)

        feature_attributions = self._distribute_to_features(
            source_attributions=source_attributions,
            alliance_members=alliance_members,
            feature_effects=feature_effects,
            interactions=interactions,
            lambda_interaction=lambda_interaction,
            beta=beta,
        )

        for feature in self.features:
            feature_attributions.setdefault(feature, 0.0)
        if excluded:
            feature_attributions["__excluded__"] = excluded_residual
        if threshold_is_enabled and not redistribute and below_threshold:
            feature_attributions["__below_threshold__"] = below_threshold_residual

        explanation = DhondtExplanation(
            score=score,
            baseline=baseline,
            delta=delta,
            active_delta=active_delta,
            feature_attributions=feature_attributions,
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
            effects={**feature_effects, **alliance_effects},
            interactions=interactions,
            alliance_members=alliance_members,
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
        )
        self.last_explanation = explanation
        return explanation

    def explain_many(self, X, max_rows=None, random_state=None, reuse_background_sample=False, **kwargs):
        X = self.background_data if X is None else self._ensure_frame(X)
        if max_rows is not None:
            X = X.iloc[:max_rows]

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
        self.features = list(self.background_data.columns)
        self._validate_feature_names(self.features)
        self.reset_cache()
        return self

    def reset_cache(self):
        self._baseline_cache = {}
        return self

    def check_model_compatibility(self, X_sample=None, class_index=None, target_index=None):
        if self.background_data is None and X_sample is None:
            return {
                "compatible": False,
                "problem": "No background data or X_sample was provided.",
                "suggestion": "Provide background_data, call fit(...), or pass X_sample.",
            }

        X = self.background_data.head(5) if X_sample is None else self._ensure_frame(X_sample).head(5)
        resolved_output_type = self._resolve_output_type()
        original_features = self.features

        try:
            if self.features is None:
                self.features = list(X.columns)
                self._validate_feature_names(self.features)
            X = X[self.features]
            class_index = self.class_index if class_index is None else class_index
            target_index = self.target_index if target_index is None else target_index
            class_index = self._resolve_class_index_for_x(X.head(1), class_index, target_index)
            model_input = self._prepare_model_input(X)
            raw_output = self._raw_model_output(model_input)
            selected = self._score_frame(X, class_index=class_index, target_index=target_index)
            return {
                "compatible": True,
                "input_format": self.input_format if self.input_adapter is None else "input_adapter",
                "output_type": self.output_type,
                "resolved_output_type": resolved_output_type,
                "raw_output_shape": tuple(np.asarray(raw_output).shape),
                "selected_output_shape": tuple(np.asarray(selected).shape),
                "numeric": True,
                "class_index": class_index,
                "target_index": target_index,
                "message": "Model is compatible with DhondtXAI.",
            }
        except Exception as exc:
            return {
                "compatible": False,
                "problem": str(exc),
                "suggestion": (
                    "Check output_type, class_index/target_index, input_format/input_adapter, "
                    "or provide a numeric predict_fn."
                ),
            }
        finally:
            self.features = original_features

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

        The new SHAP-independent local method is explain(...). This method still
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

    def plot_explanation(self, explanation=None, level="alliance", top_k=None, include_residuals=True, show=True):
        explanation = explanation or self.last_explanation
        if explanation is None:
            raise ValueError("No explanation is available. Call explain(...) first.")

        if level == "feature":
            frame = explanation.to_feature_frame(top_k=top_k, include_residuals=include_residuals)
            names = frame["feature"].tolist()
            values = frame["attribution"].tolist()
            ylabel = "Local attribution"
        elif level == "alliance":
            frame = explanation.to_alliance_frame()
            names = frame["alliance"].tolist()
            values = frame["source_attribution"].tolist()
            ylabel = "Alliance attribution"
        else:
            raise ValueError("level must be 'feature' or 'alliance'.")

        colors = ["tab:blue" if value >= 0 else "tab:red" for value in values]
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(names, values, color=colors)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_ylabel(ylabel)
        ax.set_title("DhondtXAI Local Explanation")
        ax.tick_params(axis="x", rotation=90)
        fig.tight_layout()
        if show:
            plt.show()
        return fig, ax

    def plot_local_bar(self, explanation=None, top_k=15, include_residuals=True, show=True):
        return self.plot_explanation(
            explanation=explanation,
            level="feature",
            top_k=top_k,
            include_residuals=include_residuals,
            show=show,
        )

    def plot_waterfall(self, explanation=None, top_k=10, include_residuals=True, show=True):
        explanation = explanation or self.last_explanation
        if explanation is None:
            raise ValueError("No explanation is available. Call explain(...) first.")

        frame = explanation.to_feature_frame(include_residuals=include_residuals)
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
                                    "direction": "supports prediction" if other_value > 0 else "opposes prediction",
                                    "is_residual": True,
                                }
                            ]
                        ),
                    ],
                    ignore_index=True,
                )
            frame = selected

        labels = ["baseline"] + frame["feature"].tolist() + ["score"]
        current = explanation.baseline
        bottoms = []
        heights = []
        colors = []
        for value in frame["attribution"].to_numpy(dtype=float):
            next_value = current + value
            bottoms.append(min(current, next_value))
            heights.append(abs(value))
            colors.append("tab:blue" if value >= 0 else "tab:red")
            current = next_value

        fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.8), 6))
        ax.scatter([0], [explanation.baseline], color="black", zorder=3, label="baseline")
        for index, (bottom, height, color) in enumerate(zip(bottoms, heights, colors), start=1):
            ax.bar(index, height, bottom=bottom, color=color, width=0.65)
        ax.scatter([len(labels) - 1], [explanation.score], color="black", zorder=3, label="score")
        ax.axhline(explanation.baseline, color="gray", linewidth=0.8, linestyle="--")
        ax.axhline(explanation.score, color="black", linewidth=0.8, linestyle=":")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=75, ha="right")
        ax.set_ylabel("Model score")
        ax.set_title("DhondtXAI Waterfall Explanation")
        fig.tight_layout()
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
        colors = [
            "tab:gray" if bool(row.is_residual) else ("tab:blue" if row.directional >= 0 else "tab:red")
            for row in frame.itertuples(index=False)
        ]

        fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(frame))))
        ax.barh(frame["feature"], frame["global_abs"], color=colors)
        ax.invert_yaxis()
        ax.set_xlabel("Mean absolute DhondtXAI attribution")
        ax.set_title("DhondtXAI Global Importance")
        fig.tight_layout()
        if show:
            plt.show()
        return fig, ax

    def plot_global_alliance_heatmap(self, matrix=None, show=True):
        matrix = matrix if matrix is not None else self.global_alliance_matrix_
        if matrix is None:
            raise ValueError("No global alliance matrix is available. Call explain_global(...) first.")

        fig, ax = plt.subplots(figsize=(max(7, 0.45 * len(matrix.columns)), max(6, 0.45 * len(matrix.index))))
        image = ax.imshow(matrix.to_numpy(dtype=float), vmin=0, vmax=1, cmap="Blues")
        ax.set_xticks(np.arange(len(matrix.columns)))
        ax.set_yticks(np.arange(len(matrix.index)))
        ax.set_xticklabels(matrix.columns, rotation=90)
        ax.set_yticklabels(matrix.index)
        ax.set_title("DhondtXAI Global Alliance Co-occurrence")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        if show:
            plt.show()
        return fig, ax

    def _ensure_frame(self, X):
        if isinstance(X, pd.DataFrame):
            return X.copy()
        if isinstance(X, pd.Series):
            return pd.DataFrame([X.to_dict()])
        frame = pd.DataFrame(X)
        if self.features is not None and len(frame.columns) == len(self.features):
            frame.columns = self.features
        return frame

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

    def _score_frame(self, X, class_index=None, target_index=None):
        X = self._ensure_frame(X)[self.features]
        output_type = self._resolve_output_type()
        model_input = self._prepare_model_input(X)

        if self.predict_fn is not None:
            raw_scores = np.asarray(self.predict_fn(model_input))
            if output_type == "custom":
                return self._select_output(raw_scores, target_index)
            if output_type == "logit":
                values = self._select_output(raw_scores, class_index)
                values = np.clip(values, self.eps, 1.0 - self.eps)
                return np.log(values / (1.0 - values))
            return self._select_output(
                raw_scores,
                class_index if output_type in {"probability", "decision"} else target_index,
            )

        if self.model is None:
            raise ValueError("No model or predict_fn is available for scoring.")

        if output_type in {"probability", "logit"}:
            if not hasattr(self.model, "predict_proba"):
                raise ValueError("output_type='probability' or 'logit' requires predict_proba.")
            scores = np.asarray(self.model.predict_proba(model_input))
            values = self._select_output(scores, class_index)
            if output_type == "logit":
                values = np.clip(values, self.eps, 1.0 - self.eps)
                values = np.log(values / (1.0 - values))
            return np.asarray(values, dtype=float)

        if output_type == "decision":
            if not hasattr(self.model, "decision_function"):
                raise ValueError("output_type='decision' requires decision_function.")
            scores = np.asarray(self.model.decision_function(model_input))
            if scores.ndim == 1:
                classes = self._get_classes()
                if classes is not None and len(classes) == 2:
                    index = self._validate_class_index(class_index if class_index is not None else self.class_index, 2)
                    values = scores if index == 1 else -scores
                    return np.asarray(values, dtype=float)
                return self._ensure_numeric_output(scores, "decision_function")
            return self._select_output(scores, class_index)

        if output_type == "prediction":
            scores = np.asarray(self.model.predict(model_input))
            return self._select_output(scores, target_index)

        if output_type == "custom":
            raise ValueError("output_type='custom' requires predict_fn.")

        raise ValueError("output_type must be auto, probability, logit, decision, prediction, custom.")

    def _raw_model_output(self, model_input):
        output_type = self._resolve_output_type()
        if self.predict_fn is not None:
            return self.predict_fn(model_input)

        if self.model is None:
            raise ValueError("No model or predict_fn is available for scoring.")

        if output_type in {"probability", "logit"}:
            if not hasattr(self.model, "predict_proba"):
                raise ValueError("output_type='probability' or 'logit' requires predict_proba.")
            return self.model.predict_proba(model_input)
        if output_type == "decision":
            if not hasattr(self.model, "decision_function"):
                raise ValueError("output_type='decision' requires decision_function.")
            return self.model.decision_function(model_input)
        if output_type == "prediction":
            return self.model.predict(model_input)
        raise ValueError("output_type='custom' requires predict_fn.")

    def _resolve_output_type(self):
        if self.output_type != "auto":
            return self.output_type
        if self.predict_fn is not None:
            return "custom"
        if self.model is not None and hasattr(self.model, "predict_proba"):
            return "probability"
        if self.model is not None and hasattr(self.model, "decision_function"):
            return "decision"
        return "prediction"

    def _prepare_model_input(self, X):
        if self.input_adapter is not None:
            return self.input_adapter(X.copy())
        if self.input_format == "numpy":
            return X.to_numpy()
        return X

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

    def _ensure_numeric_output(self, values, source):
        try:
            return np.asarray(values, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{source} must return numeric scores. For classifiers, use output_type='probability', "
                "'logit', 'decision', or provide a numeric predict_fn."
            ) from exc

    def _validate_class_index(self, class_index, class_count):
        if class_index == "predicted":
            raise ValueError("class_index='predicted' must be resolved before output selection.")
        if class_index < 0 or class_index >= class_count:
            raise ValueError(f"class_index={class_index} is outside the valid range [0, {class_count - 1}].")
        return int(class_index)

    def _resolve_class_index_for_x(self, X, class_index, target_index=None):
        if class_index != "predicted":
            return class_index

        output_type = self._resolve_output_type()
        X = self._ensure_frame(X)[self.features]
        model_input = self._prepare_model_input(X)

        if self.predict_fn is not None:
            scores = np.asarray(self.predict_fn(model_input))
        elif output_type in {"probability", "logit"}:
            if not hasattr(self.model, "predict_proba"):
                raise ValueError("class_index='predicted' with probability/logit requires predict_proba.")
            scores = np.asarray(self.model.predict_proba(model_input))
        elif output_type == "decision":
            if not hasattr(self.model, "decision_function"):
                raise ValueError("class_index='predicted' with decision output requires decision_function.")
            scores = np.asarray(self.model.decision_function(model_input))
        else:
            raise ValueError(
                "class_index='predicted' requires probability, logit, decision, or a 2D custom predict_fn output."
            )

        if scores.ndim == 2:
            return int(np.argmax(scores[0]))
        if scores.ndim == 1:
            classes = self._get_classes()
            if classes is not None and len(classes) == 2:
                return 1 if float(scores[0]) >= 0 else 0
        raise ValueError("class_index='predicted' requires a 2D score matrix or binary decision scores.")

    def _class_label(self, class_index):
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
    ):
        perturbation = self.perturbation if perturbation is None else perturbation
        perturbation_sampler = self.perturbation_sampler if perturbation_sampler is None else perturbation_sampler
        groups = [tuple(group) for group in groups]
        if not groups:
            return {}

        blocks = []
        sizes = []
        for group in groups:
            rows = self._replacement_rows(
                x_series,
                group,
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
        shares = {name: votes[name] / (total_votes + self.eps) for name in names}
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
        allocation = np.zeros(len(votes), dtype=int)
        if seats <= 0 or len(votes) == 0 or votes.sum() <= self.eps:
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
        if total > self.eps:
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

    def _project_values(self, raw_values, target, names):
        projected = {name: 0.0 for name in raw_values}
        names = list(names)
        if not names:
            return projected, 0.0, target, 0.0

        raw_sum = sum(raw_values.get(name, 0.0) for name in names)
        residual = target - raw_sum
        magnitudes = np.asarray([abs(raw_values.get(name, 0.0)) + self.eps for name in names], dtype=float)
        weights = magnitudes / magnitudes.sum()
        for name, weight in zip(names, weights):
            projected[name] = raw_values.get(name, 0.0) + float(weight) * residual

        denominator = max(abs(target), sum(abs(raw_values.get(name, 0.0)) for name in names), self.eps)
        ratio = abs(residual) / denominator
        return projected, raw_sum, residual, ratio

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
        positions = {name: index for index, name in enumerate(self.features)}
        if feature in positions:
            return (0, positions[feature], "")
        return (1, str(type(feature)), str(feature))

    def _pair_key(self, left, right):
        return tuple(sorted((left, right), key=self._feature_order_key))

    def _group_key(self, group):
        return tuple(sorted(tuple(group), key=self._feature_order_key))
