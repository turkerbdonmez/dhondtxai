"""Synthetic additive ground-truth benchmark for DhondtXAI.

Run:
    python benchmarks/synthetic_additive.py

Output:
    benchmarks/results/synthetic_additive.csv
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dhondtxai import DhondtXAI


class LinearModel:
    def __init__(self, weights):
        self.weights = np.asarray(weights, dtype=float)

    def predict(self, X):
        return np.asarray(X, dtype=float) @ self.weights


def main():
    rng = np.random.default_rng(42)
    feature_count = 8
    rows = 200
    weights = np.linspace(0.5, 2.0, feature_count)
    columns = [f"x{i}" for i in range(feature_count)]
    background = pd.DataFrame(rng.normal(0.0, 1.0, size=(rows, feature_count)), columns=columns)
    x = pd.Series(rng.normal(0.0, 1.0, size=feature_count), index=columns)

    model = LinearModel(weights)
    explainer = DhondtXAI(model, background_data=background, output_type="prediction", random_state=42)
    explanation = explainer.explain(x, n_background=len(background), allocation_seats=10000, random_state=42)

    expected = weights * (x.to_numpy(dtype=float) - background.mean(axis=0).to_numpy(dtype=float))
    frame = pd.DataFrame(
        {
            "feature": columns,
            "expected": expected,
            "dhondtxai": [explanation.feature_attributions[name] for name in columns],
        }
    )
    frame["absolute_error"] = (frame["dhondtxai"] - frame["expected"]).abs()
    frame["completeness_error"] = abs(sum(explanation.feature_attributions.values()) - explanation.delta)
    frame["projection_residual_ratio"] = explanation.projection_residual_ratio

    output_dir = Path(__file__).resolve().parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "synthetic_additive.csv"
    frame.to_csv(output_path, index=False)
    print(output_path)


if __name__ == "__main__":
    main()
