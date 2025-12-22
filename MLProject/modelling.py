import os
import pandas as pd
import numpy as np

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Preprocessor otomatis:
    - numeric: impute median + scale
    - categorical: impute most_frequent + onehot(handle_unknown)
    """
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    numeric_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, num_cols),
            ("cat", categorical_pipe, cat_cols),
        ],
        remainder="drop"
    )
    return preprocessor


def main():
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("Telco-CI-Retrain")

    DATA_PATH = "telco_preprocessed/telco_preprocessed.csv"
    df = pd.read_csv(DATA_PATH)
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    target_candidates = ["Churn", "churn", "label", "target"]
    target_col = None
    for c in target_candidates:
        if c in df.columns:
            target_col = c
            break

    if target_col is None:
        raise ValueError(
            f"Target column not found. Add one of {target_candidates} or set target_col manually."
        )

    y = df[target_col]
    X = df.drop(columns=[target_col])

    if y.dtype == "object":
        y = y.astype(str).str.lower().map({"yes": 1, "no": 0, "true": 1, "false": 0})
    y = y.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = build_preprocessor(X_train)

    clf = LogisticRegression(max_iter=2000)

    pipeline = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", clf),
    ])

    with mlflow.start_run():
        pipeline.fit(X_train, y_train)

        proba = pipeline.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)

        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
        mlflow.log_metric("accuracy", float(accuracy_score(y_test, pred)))
        mlflow.log_metric("f1", float(f1_score(y_test, pred)))
        mlflow.log_metric("roc_auc", float(roc_auc_score(y_test, proba)))

        signature = infer_signature(X_train, pipeline.predict_proba(X_train))

        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            signature=signature,
            input_example=X_train.head(2),
        )

        print("Training done + pipeline logged to MLflow as artifact 'model'")


if __name__ == "__main__":
    main()
