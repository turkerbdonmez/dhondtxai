import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from dhondtxai import DhondtXAI, plot_signed_parliament


def main():
    dataset = load_breast_cancer()
    X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    y = pd.Series(dataset.target)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    explainer = DhondtXAI(model, output_type="probability", class_index=1, random_state=42)
    explainer.fit(X_train, y_train)

    explanation = explainer.explain(
        X_test.iloc[0],
        seats=120,
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
        beta=1.0,
        perturbation="interventional",
        tie_break="stable",
    )

    print(f"Model score: {explanation.score:.4f}")
    print(f"Baseline: {explanation.baseline:.4f}")
    print(f"Delta: {explanation.delta:.4f}")
    print(f"Projection residual ratio: {explanation.projection_residual_ratio:.4f}")
    print(f"Excluded residual: {explanation.excluded_residual:.4f}")
    print(f"Below-threshold residual: {explanation.below_threshold_residual:.4f}")
    print("\nExplanation summary")
    print(explanation.summary(top_k=5))
    print("\nFeature attributions")
    print(explanation.to_feature_frame())
    print("\nAlliance seats")
    print(explanation.to_alliance_frame())

    explainer.plot_local_bar(explanation, top_k=10, show=False)
    explainer.plot_waterfall(explanation, top_k=10, show=False)
    plot_signed_parliament(explanation, mode="signed", show=False)


if __name__ == "__main__":
    main()
