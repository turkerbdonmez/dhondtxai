"""Lightweight masker helpers for DhondtXAI tabular explanations."""

import pandas as pd


class Independent:
    """Use independent background replacement for hidden features."""

    perturbation = "interventional"

    def __init__(self, background_data, max_samples=None):
        self.background_data = _ensure_frame(background_data)
        self.max_samples = _validate_max_samples(max_samples)


class ConditionalKNN:
    """Use nearest-neighbor conditional replacement for hidden features."""

    perturbation = "conditional_knn"

    def __init__(self, background_data, knn_neighbors=25, max_samples=None):
        self.background_data = _ensure_frame(background_data)
        self.knn_neighbors = int(knn_neighbors)
        self.max_samples = _validate_max_samples(max_samples)


class UserDefined:
    """Wrap a custom perturbation sampler.

    The sampler must accept ``(x, group, background_sample, n)`` and return a
    replacement table with the same feature columns as the background data.
    """

    perturbation = "user_sampler"

    def __init__(self, sampler, background_data=None):
        if not callable(sampler):
            raise ValueError("sampler must be callable.")
        self.sampler = sampler
        self.background_data = None if background_data is None else _ensure_frame(background_data)

    def __call__(self, x, group, background_sample, n):
        return self.sampler(x, group, background_sample, n)


def _ensure_frame(data):
    if isinstance(data, pd.DataFrame):
        return data.copy()
    return pd.DataFrame(data)


def _validate_max_samples(max_samples):
    if max_samples is None:
        return None
    max_samples = int(max_samples)
    if max_samples <= 0:
        raise ValueError("max_samples must be positive.")
    return max_samples
