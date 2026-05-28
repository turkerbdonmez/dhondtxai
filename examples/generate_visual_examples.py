from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "exampleimages"
sys.path.insert(0, str(ROOT))

from dhondtxai import Explainer, plot_signed_parliament


def save(fig, *filenames):
    path = None
    for filename in filenames:
        path = OUTPUT_DIR / filename
        fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dataset = load_breast_cancer()
    X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    y = pd.Series(dataset.target)
    X_train, X_test, y_train, _ = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    explainer = Explainer(model, X_train, random_state=42)
    explanation = explainer.explain(
        X_test.iloc[0],
        seats=200,
        allocation_seats=5000,
        threshold=0.05,
        redistribute=True,
        alliance_mode="user",
        user_alliances=[
            ["mean concavity", "mean concave points"],
            ["mean radius", "mean perimeter", "mean area"],
        ],
        n_background=50,
        lambda_interaction=0.2,
    )

    fig, _ = explainer.plot_local_bar(explanation, top_k=15, show=False)
    save(fig, "local_bar.png", "barplotview.png")

    fig, _ = explainer.plot_waterfall(explanation, top_k=12, show=False)
    save(fig, "waterfall.png")

    fig, _ = plot_signed_parliament(explanation, mode="signed", seat_count=600, show=False)
    save(fig, "signed_parliament.png", "parliamentview.png")

    fig, _ = plot_signed_parliament(explanation, mode="positive", seat_count=600, show=False)
    save(fig, "positive_parliament.png")

    fig, _ = plot_signed_parliament(explanation, mode="negative", seat_count=600, show=False)
    save(fig, "negative_parliament.png")

    global_frame = explainer.explain_global(
        X_test.head(20),
        max_rows=20,
        random_state=42,
        n_background=30,
        seats=100,
        alliance_mode="user",
        user_alliances=[
            ["mean concavity", "mean concave points"],
            ["worst concavity", "worst concave points"],
            ["mean radius", "mean perimeter", "mean area"],
            ["worst radius", "worst perimeter", "worst area"],
        ],
    )

    fig, _ = explainer.plot_global_importance(global_frame, top_k=15, show=False)
    save(fig, "global_importance.png")

    fig, _ = explainer.plot_global_alliance_heatmap(show=False)
    save(fig, "global_alliance_heatmap.png")


if __name__ == "__main__":
    main()
