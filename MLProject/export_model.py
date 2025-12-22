import os
import joblib
import mlflow
from mlflow.tracking import MlflowClient

OUT_PATH = "../monitoring/model.pkl"

def main():
    client = MlflowClient()
    exps = client.search_experiments()
    if not exps:
        raise SystemExit("No experiments found")

    newest = None
    for e in exps:
        runs = client.search_runs(
            experiment_ids=[e.experiment_id],
            order_by=["attributes.start_time DESC"],
            max_results=1
        )
        if runs:
            r = runs[0]
            if newest is None or r.info.start_time > newest.info.start_time:
                newest = r

    if newest is None:
        raise SystemExit("No runs found")

    run_id = newest.info.run_id
    model_uri = f"runs:/{run_id}/model"
    print("Using model:", model_uri)

    model = mlflow.sklearn.load_model(model_uri)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    joblib.dump(model, OUT_PATH)
    print(" Saved:", OUT_PATH)

if __name__ == "__main__":
    main()
