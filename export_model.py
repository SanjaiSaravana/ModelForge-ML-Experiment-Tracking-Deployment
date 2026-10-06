"""
Model Export Script
Exports the trained model from MLflow tracking artifacts to an independent model.pkl file
using joblib, and verifies that the exported pickle can be loaded and used standalone.
"""

import os
import sys
import joblib
import pandas as pd
import mlflow
import mlflow.xgboost


def export_model_to_pickle(run_id=None):
    print("=" * 60)
    print("Exporting MLflow XGBoost Model to Standalone model.pkl")
    print("=" * 60)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_pkl_path = os.path.join(script_dir, "model.pkl")

    # 1. Resolve Run ID
    if not run_id:
        if len(sys.argv) > 1 and sys.argv[1].strip():
            run_id = sys.argv[1].strip()
        else:
            run_id_file = os.path.join(script_dir, "latest_run_id.txt")
            if os.path.exists(run_id_file):
                with open(run_id_file, "r", encoding="utf-8") as f:
                    run_id = f.read().strip()
            else:
                experiment = mlflow.get_experiment_by_name("Demo_Local_Run")
                if experiment:
                    runs = mlflow.search_runs(
                        experiment_ids=[experiment.experiment_id],
                        order_by=["start_time DESC"],
                        max_results=1
                    )
                    if not runs.empty:
                        run_id = runs.iloc[0]["run_id"]

    if not run_id:
        raise ValueError(
            "Could not find an MLflow run to export. Please run 'python mlflow_demo.py' first."
        )

    model_uri = f"runs:/{run_id}/xgboost_model"
    print(f"Loading trained model from MLflow artifact URI: {model_uri}")
    model = mlflow.xgboost.load_model(model_uri)

    # 2. Export model using joblib
    print(f"Saving model using joblib to: {model_pkl_path}")
    joblib.dump(model, model_pkl_path)

    # 3. Verification: Ensure file exists and is not empty
    if not os.path.exists(model_pkl_path):
        raise FileNotFoundError(f"Export failed! {model_pkl_path} does not exist.")

    file_size = os.path.getsize(model_pkl_path)
    if file_size == 0:
        raise ValueError("Export failed! model.pkl is empty (0 bytes).")

    print(f"[SUCCESS] model.pkl created successfully. Size: {file_size} bytes.")

    # 4. Standalone loading verification (independent of MLflow)
    print("\nVerifying standalone loading with joblib.load()...")
    loaded_standalone_model = joblib.load(model_pkl_path)

    # Test prediction
    test_input = pd.DataFrame([[4, 10]], columns=["feature1", "feature2"])
    pred = loaded_standalone_model.predict(test_input)
    print(f"Standalone Model Test Prediction for [4, 10]: Class {int(pred[0])}")
    print("[SUCCESS] Standalone model verification passed!")
    print("=" * 60)

    return model_pkl_path


if __name__ == "__main__":
    export_model_to_pickle()
