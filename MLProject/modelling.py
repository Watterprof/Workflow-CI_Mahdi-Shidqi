from pathlib import Path
import os
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parents[1]  
DATA_PATH = BASE_DIR / "MLProject" / "telco_preprocessed" / "telco_preprocessed.csv"

TMP_DIR = BASE_DIR / "MLProject" / "tmp_artifacts"
TMP_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)
df.columns = df.columns.astype(str).str.strip()

TARGET_COL = "target"
if TARGET_COL not in df.columns:
    raise ValueError(
        f"Kolom target '{TARGET_COL}' tidak ditemukan. Kolom yang ada (tail): {df.columns.tolist()[-10:]}"
    )

y = df[TARGET_COL]
X = df.drop(columns=[TARGET_COL])

if y.dtype == "object":
    y = y.astype(str).str.strip().str.lower().map({"yes": 1, "no": 0, "1": 1, "0": 0})

y = pd.to_numeric(y, errors="coerce")
if y.isna().any():
    raise ValueError("Kolom target berisi nilai non-numerik/invalid. Pastikan target hanya 0/1.")

uniq = set(y.unique().tolist())
if not uniq.issubset({0, 1}):
    raise ValueError(f"Target harus biner 0/1. Nilai unik yang ditemukan: {sorted(list(uniq))}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
if tracking_uri:
    mlflow.set_tracking_uri(tracking_uri)

mlflow.set_experiment("Eksperimen_SML_K2")
mlflow.sklearn.autolog(log_models=True)

started_here = False
if mlflow.active_run() is None:
    mlflow.start_run()
    started_here = True

try:
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    mlflow.log_metric("accuracy_manual", accuracy_score(y_test, pred))
    mlflow.log_metric("f1_manual", f1_score(y_test, pred))
    mlflow.log_metric("roc_auc_manual", roc_auc_score(y_test, proba))

    cm_path = TMP_DIR / "confusion_matrix.png"
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_predictions(y_test, pred, ax=ax)
    plt.tight_layout()
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)
    mlflow.log_artifact(str(cm_path))

    est_path = TMP_DIR / "estimator.html"
    try:
        html = model._repr_html_()
        est_path.write_text(html, encoding="utf-8")
        mlflow.log_artifact(str(est_path))
    except Exception:
        pass

finally:
    if started_here:
        mlflow.end_run()

print("Training selesai. MLflow tracking URI:", mlflow.get_tracking_uri())
