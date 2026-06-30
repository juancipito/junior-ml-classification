from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "sample_synthetic_data.csv"


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    target = "sla_risk"
    features = [
        "tickets_last_7d",
        "avg_aht_seconds",
        "adherence_pct",
        "schedule_variance_min",
        "tenure_days",
        "training_completed",
        "language_group",
    ]
    X_train, X_test, y_train, y_test = train_test_split(
        df[features], df[target], test_size=0.25, random_state=42, stratify=df[target]
    )
    numeric = [c for c in features if c != "language_group"]
    categorical = ["language_group"]
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ]
    )
    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", LogisticRegression(max_iter=1000)),
        ]
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    print("Confusion matrix")
    print(confusion_matrix(y_test, predictions))
    print("\nClassification report")
    print(classification_report(y_test, predictions, digits=3))
    print(f"ROC AUC: {roc_auc_score(y_test, probabilities):.3f}")


if __name__ == "__main__":
    main()
