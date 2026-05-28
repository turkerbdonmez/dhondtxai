
from .dhondt_xai import DhondtExplanation, DhondtValues, DhondtXAI
from .plot_parliament import plot_parliament, plot_signed_parliament
from . import maskers

__version__ = "0.9.5.6"
Explainer = DhondtXAI


_EXPLAIN_KWARGS = {
    "class_index",
    "target_index",
    "seats",
    "allocation_seats",
    "threshold",
    "threshold_enabled",
    "redistribute",
    "alliance_mode",
    "user_alliances",
    "exclude_features",
    "n_background",
    "lambda_interaction",
    "lambda_alliance_vote",
    "lambda_member_split",
    "rho",
    "beta",
    "auto_alliance_method",
    "perturbation",
    "perturbation_sampler",
    "affinity_mode",
    "tie_break",
    "projection_mode",
    "projection_residual_threshold",
    "exclude_mode",
    "threshold_mode",
    "baseline_mode",
    "cost_mode",
    "preset",
    "max_model_rows",
    "max_interaction_pairs",
    "top_k_interaction_features",
    "interaction_screening",
    "allocation_error_tolerance",
    "random_state",
}


def explain(model=None, X_background=None, X=None, **kwargs):
    """One-call convenience API for a local DhondtXAI explanation."""
    if X is None:
        if "x" in kwargs:
            X = kwargs.pop("x")
        else:
            raise ValueError("Provide X or x for the row to explain.")
    if X_background is None:
        if "background_data" in kwargs:
            X_background = kwargs.pop("background_data")
        else:
            raise ValueError("Provide X_background or background_data.")

    explain_kwargs = {key: kwargs.pop(key) for key in list(kwargs.keys()) if key in _EXPLAIN_KWARGS}
    explainer = DhondtXAI(model=model, background_data=X_background, **kwargs)
    return explainer.explain(X, **explain_kwargs)

__all__ = [
    "DhondtExplanation",
    "DhondtValues",
    "DhondtXAI",
    "Explainer",
    "explain",
    "maskers",
    "plot_parliament",
    "plot_signed_parliament",
]
