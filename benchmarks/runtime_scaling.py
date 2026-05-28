"""Small runtime scaling benchmark for DhondtXAI.

Run:
    python benchmarks/runtime_scaling.py

Output:
    benchmarks/results/runtime_scaling.csv
"""

from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dhondtxai import DhondtXAI


class SumModel:
    def predict(self, X):
        return np.asarray(X, dtype=float).sum(axis=1)


def main():
    rng = np.random.default_rng(123)
    rows = []
    for feature_count in [5, 10, 20, 40]:
        columns = [f"x{i}" for i in range(feature_count)]
        background = pd.DataFrame(rng.normal(size=(100, feature_count)), columns=columns)
        x = pd.Series(rng.normal(size=feature_count), index=columns)
        explainer = DhondtXAI(SumModel(), background_data=background, output_type="prediction")

        start = perf_counter()
        explanation = explainer.explain(x, n_background=50, allocation_seats=5000, random_state=123)
        elapsed = perf_counter() - start
        rows.append(
            {
                "feature_count": feature_count,
                "elapsed_seconds": elapsed,
                "projection_residual_ratio": explanation.projection_residual_ratio,
                "completeness_error": abs(sum(explanation.feature_attributions.values()) - explanation.delta),
            }
        )

    output_dir = Path(__file__).resolve().parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "runtime_scaling.csv"
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(output_path)


if __name__ == "__main__":
    main()
