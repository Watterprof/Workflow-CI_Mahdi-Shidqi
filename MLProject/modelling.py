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
    Menyiapkan data:
    - Angka (Numeric): Isi data kosong dengan median, lalu standarisasi (scaling).
    - Teks (Categorical): Isi data kosong dengan yang paling sering muncul, lalu ubah jadi angka (one-hot).
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

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, num_cols),
            ("cat", categorical_pipe, cat_cols),
        ],
        remainder="drop"
    )

def main():    

    mlflow.set_experiment("Telco-CI-Retrain")

    DATA_PATH = "telco_preprocessed/telco_preprocessed.csv"
    
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"File dataset tidak ketemu di: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    print(f"Data terbaca: {len(df)} baris, {len(df.columns)} kolom.")

    target_candidates = ["Churn", "churn", "label", "target"]
    target_col = next((c for c in target_candidates if c in df.columns), None)

    if target_col is None:
        raise ValueError(f"Kolom target tidak ada! Kolom tersedia: {df.columns.tolist()}.")

    y = df[target_col]
    X = df.drop(columns=[target_col])

    if y.dtype == "object" or y.dtype == "bool":
        y = y.astype(str).str.lower().map({"yes": 1, "no": 0, "true": 1, "false": 0, "1": 1, "0": 0})
    
    y = y.fillna(0).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = Pipeline(steps=[
        ("preprocess", build_preprocessor(X_train)),
        ("model", LogisticRegression(max_iter=2000)),
    ])

    with mlflow.start_run():
        pipeline.fit(X_train, y_train)

        proba = pipeline.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)

        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
        metrics = {
            "accuracy": accuracy_score(y_test, pred),
            "f1": f1_score(y_test, pred),
            "roc_auc": roc_auc_score(y_test, proba)
        }

        mlflow.log_metrics(metrics)
        print(f"Hasil Training -> Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}")

        signature = infer_signature(X_train, pipeline.predict(X_train))
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            signature=signature,
            input_example=X_train.head(1),
        )

        print("Selesai! Model sudah tersimpan di MLflow.")

if __name__ == "__main__":
    main()