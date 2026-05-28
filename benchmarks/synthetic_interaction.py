"""Synthetic interaction benchmark for automatic DhondtXAI alliances.

Run:
    python benchmarks/synthetic_interaction.py

Output:
    benchmarks/results/synthetic_interaction.csv
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dhondtxai import DhondtXAI


class InteractionModel:
    def predict(self, X):
        arr = np.asarray(X, dtype=float)
        return 2.0 * arr[:, 0] * arr[:, 1] + arr[:, 2]


def main():
    rng = np.random.default_rng(7)
    columns = ["x0", "x1", "x2"]
    background = pd.DataFrame(rng.normal(0.0, 0.2, size=(100, 3)), columns=columns)
    x = pd.Series({"x0": 2.0, "x1": 2.0, "x2": 0.5})

    explainer = DhondtXAI(
        InteractionModel(),
        background_data=background,
        output_type="prediction",
        affinity_mode="absolute_interaction",
        random_state=7,
    )
    explanation = explainer.explain(
        x,
        alliance_mode="auto",
        affinity_mode="absolute_interaction",
        rho=0.1,
        lambda_interaction=0.5,
        n_background=100,
        allocation_seats=10000,
        random_state=7,
    )

    rows = []
    for key, value in explanation.interactions.items():
        rows.append(
            {
                "pair": " + ".join(map(str, key)),
                "interaction": value,
                "same_alliance": explanation.feature_source_alliance.get(key[0])
                == explanation.feature_source_alliance.get(key[1]),
            }
        )
    frame = pd.DataFrame(rows)
    frame["projection_residual_ratio"] = explanation.projection_residual_ratio

    output_dir = Path(__file__).resolve().parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "synthetic_interaction.csv"
    frame.to_csv(output_path, index=False)
    print(output_path)


if __name__ == "__main__":
    main()
