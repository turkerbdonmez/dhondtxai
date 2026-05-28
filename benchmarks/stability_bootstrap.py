"""Bootstrap stability benchmark for DhondtXAI global importance.

Run:
    python benchmarks/stability_bootstrap.py

Output:
    benchmarks/results/stability_bootstrap.csv
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dhondtxai import DhondtXAI


class WeightedModel:
    def __init__(self, weights):
        self.weights = np.asarray(weights, dtype=float)

    def predict(self, X):
        return np.asarray(X, dtype=float) @ self.weights


def main():
    rng = np.random.default_rng(11)
    feature_count = 6
    columns = [f"x{i}" for i in range(feature_count)]
    background = pd.DataFrame(rng.normal(size=(150, feature_count)), columns=columns)
    X_eval = pd.DataFrame(rng.normal(size=(20, feature_count)), columns=columns)
    model = WeightedModel(np.linspace(1.0, 2.0, feature_count))

    frames = []
    for seed in [101, 202]:
        explainer = DhondtXAI(model, background_data=background, output_type="prediction", random_state=seed)
        frame = explainer.explain_global(X_eval, max_rows=20, n_background=50, random_state=seed)
        frames.append(frame.set_index("feature")["global_abs"].rename(f"seed_{seed}"))

    result = pd.concat(frames, axis=1).fillna(0.0)
    result["rank_correlation"] = result.iloc[:, 0].corr(result.iloc[:, 1], method="spearman")

    output_dir = Path(__file__).resolve().parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "stability_bootstrap.csv"
    result.reset_index().to_csv(output_path, index=False)
    print(output_path)


if __name__ == "__main__":
    main()
