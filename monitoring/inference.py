import time
from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"  
model = joblib.load(MODEL_PATH)

app = FastAPI(title="Telco Churn API + Metrics")

REQ_COUNT = Counter("api_requests_total", "Total requests", ["endpoint", "method", "status"])
PRED_COUNT = Counter("predictions_total", "Total predictions", ["label"])
LATENCY = Histogram("inference_latency_seconds", "Inference latency (seconds)")
INPROGRESS = Gauge("inprogress_requests", "In-progress requests")
LAST_PRED_PROBA = Gauge("last_prediction_proba", "Last prediction probability (churn=1)")

ERROR_COUNT = Counter("api_errors_total", "Total errors")
FEATURE_COUNT = Gauge("feature_count", "Number of features received")
BATCH_SIZE = Gauge("batch_size", "Batch size for prediction")
MODEL_UP = Gauge("model_up", "Model loaded (1=up)")
MODEL_UP.set(1)

EXPECTED_FEATURES = None
if hasattr(model, "feature_names_in_"):
    EXPECTED_FEATURES = [str(c) for c in model.feature_names_in_]

class Payload(BaseModel):
    data: dict

@app.get("/")
def home():
    return {"msg": "API jalan. Buka /docs untuk Swagger."}

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": True}

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

def _prepare_df(data_dict: dict) -> pd.DataFrame:
    """
    Buat DataFrame 1 baris, lalu:
    - kalau model punya EXPECTED_FEATURES: auto isi yang kurang = 0, drop yang ekstra, urutkan kolom
    - convert semua ke numeric kalau memungkinkan
    """
    df = pd.DataFrame([data_dict])

    df.columns = [str(c) for c in df.columns]

    if EXPECTED_FEATURES is not None:
        missing = [c for c in EXPECTED_FEATURES if c not in df.columns]
        for c in missing:
            df[c] = 0

        extra = [c for c in df.columns if c not in EXPECTED_FEATURES]
        if extra:
            df = df.drop(columns=extra)

        df = df[EXPECTED_FEATURES]

    df = df.apply(pd.to_numeric, errors="coerce").fillna(0)

    return df

@app.get("/predict")
def predict_get():
    return {
        "message": "Endpoint /predict itu untuk POST (inference). Kalau kamu buka via browser itu GET, jadi aku sediain pesan ini biar tidak error.",
        "how_to_use": "Buka /docs lalu coba POST /predict → Try it out → Execute.",
        "example_body": {
            "data": {
                "0": 1,
                "1": 0,
                "2": 0,
                "7087": 1
            }
        }
    }

@app.post("/predict")
def predict(payload: Payload):
    INPROGRESS.inc()
    try:
        df = _prepare_df(payload.data)

        FEATURE_COUNT.set(df.shape[1])
        BATCH_SIZE.set(len(df))

        with LATENCY.time():
            if hasattr(model, "predict_proba"):
                proba = float(model.predict_proba(df)[:, 1][0])
            else:
                pred_val = int(model.predict(df)[0])
                proba = float(pred_val)

        pred = 1 if proba >= 0.5 else 0

        LAST_PRED_PROBA.set(proba)
        PRED_COUNT.labels(label=str(pred)).inc()
        REQ_COUNT.labels(endpoint="/predict", method="POST", status="200").inc()

        return {"prediction": pred, "proba_churn": proba}

    except Exception as e:
        ERROR_COUNT.inc()
        REQ_COUNT.labels(endpoint="/predict", method="POST", status="500").inc()
        raise HTTPException(status_code=500, detail=f"Predict failed: {type(e).__name__}: {e}")

    finally:
        INPROGRESS.dec()
