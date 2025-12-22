FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    fastapi "uvicorn[standard]" pandas numpy scikit-learn joblib \
    prometheus-client

COPY inference.py /app/inference.py
COPY model.pkl /app/model.pkl

EXPOSE 8000
CMD ["python","-m","uvicorn","inference:app","--host","0.0.0.0","--port","8000"]
