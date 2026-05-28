"""Optional export helpers for users who want SHAP Explanation objects.

DhondtXAI does not compute SHAP values. These helpers only wrap already
computed DhondtXAI attributions in the external ``shap.Explanation`` container.
"""


def to_shap_explanation(explanation_or_values, include_residuals=True):
    """Return a ``shap.Explanation`` for a DhondtXAI explanation/value object."""
    return explanation_or_values.to_shap(include_residuals=include_residuals)

