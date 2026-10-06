"""
Model Loading Script
Demonstrates loading the trained XGBoost model directly from MLflow tracking artifacts
and generating predictions.
"""

import sys
import os
import pandas as pd
import mlflow
import mlflow.xgboost


def load_and_predict(run_id=None):
    print("=" * 60)
    print("MLflow Model Loading and Prediction Test")
    print("=" * 60)

    # 1. Determine Run ID dynamically
    if not run_id:
        if len(sys.argv) > 1 and sys.argv[1].strip():
            run_id = sys.argv[1].strip()
            print(f"Using Run ID provided via command-line argument: {run_id}")
        else:
            # Check for saved latest_run_id.txt
            script_dir = os.path.dirname(os.path.abspath(__file__))
            run_id_file = os.path.join(script_dir, "latest_run_id.txt")
            if os.path.exists(run_id_file):
                with open(run_id_file, "r", encoding="utf-8") as f:
                    run_id = f.read().strip()
                print(f"Loaded Run ID from '{run_id_file}': {run_id}")
            else:
                # Query MLflow client directly
                print("Querying MLflow tracking for the latest run in 'Demo_Local_Run'...")
                experiment = mlflow.get_experiment_by_name("Demo_Local_Run")
                if experiment:
                    runs = mlflow.search_runs(
                        experiment_ids=[experiment.experiment_id],
                        order_by=["start_time DESC"],
                        max_results=1
                    )
                    if not runs.empty:
                        run_id = runs.iloc[0]["run_id"]
                        print(f"Discovered latest Run ID from MLflow: {run_id}")

    if not run_id:
        raise ValueError(
            "No MLflow run ID found! Please run 'python mlflow_demo.py' first or pass a run ID as an argument."
        )

    # 2. Build model URI and load model from MLflow
    model_uri = f"runs:/{run_id}/xgboost_model"
    print(f"Attempting to load model from MLflow URI: {model_uri}")

    try:
        model = mlflow.xgboost.load_model(model_uri)
    except Exception as e:
        print(f"Direct mlflow.xgboost load failed: {e}. Trying mlflow.pyfunc fallback...")
        model = mlflow.pyfunc.load_model(model_uri)

    print("\n[SUCCESS] Model loaded successfully from MLflow!")
    print(f"Loaded Model Object Type: {type(model)}")

    # 3. Create sample input matching training features: Feature 1 = 4, Feature 2 = 10
    test_feature1 = 4
    test_feature2 = 10
    sample_input = pd.DataFrame(
        [[test_feature1, test_feature2]],
        columns=["feature1", "feature2"]
    )

    print("\n" + "-" * 40)
    print("Test Input:")
    print(f"  Feature 1: {test_feature1}")
    print(f"  Feature 2: {test_feature2}")
    print("-" * 40)

    # 4. Make prediction
    prediction = model.predict(sample_input)
    predicted_class = int(prediction[0])

    print(f"Prediction Output: {prediction}")
    print(f"Predicted Class:   {predicted_class}")
    print("-" * 40)

    return predicted_class


if __name__ == "__main__":
    load_and_predict()
