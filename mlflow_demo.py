"""
MLflow Demo Training Script
Tracks XGBoost model training, logs parameters, metrics, and models using MLflow.
"""

import os
import joblib
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.xgboost


def run_training():
    print("=" * 60)
    print("Starting MLflow XGBoost Training Workflow")
    print("=" * 60)

    # 1. Prepare sample dataset
    # Feature 1: 1 through 10
    # Feature 2: 11 through 20
    # Target: 0, 0, 0, 0, 1, 1, 1, 1, 1, 1
    data = {
        "feature1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "feature2": [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        "target": [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    }
    df = pd.DataFrame(data)
    print(f"Sample dataset created with {len(df)} records:")
    print(df)

    X = df[["feature1", "feature2"]]
    y = df["target"]

    # 2. Train-test split
    test_size = 0.2
    random_state = 42
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"\nData split: {len(X_train)} training samples, {len(X_test)} testing samples.")

    # 3. Configure MLflow experiment
    experiment_name = "Demo_Local_Run"
    mlflow.set_experiment(experiment_name)
    tracking_uri = mlflow.get_tracking_uri()
    print(f"MLflow Tracking URI: {tracking_uri}")
    print(f"MLflow Experiment: {experiment_name}")

    # 4. Start MLflow run
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"\nStarted MLflow Run ID: {run_id}")

        # Train XGBoost Classifier
        model_type = "XGBoost"
        model = xgb.XGBClassifier(
            eval_metric="logloss",
            random_state=random_state
        )
        print("Training XGBoost Classifier...")
        model.fit(X_train, y_train)

        # Make predictions on test set
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"Model Training Complete. Test Accuracy: {acc:.4f}")

        # 5. Log parameters
        print("\nLogging parameters to MLflow...")
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("random_state", random_state)

        # 6. Log metrics
        print("Logging metrics to MLflow...")
        mlflow.log_metric("accuracy", acc)

        # 7. Log model artifact
        artifact_path = "xgboost_model"
        print(f"Logging model artifact under '{artifact_path}'...")
        mlflow.xgboost.log_model(model, artifact_path=artifact_path)
        model_uri = f"runs:/{run_id}/{artifact_path}"
        print(f"Model successfully logged to MLflow at URI: {model_uri}")

        # 8. Export model.pkl for independent deployment
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_pkl_path = os.path.join(script_dir, "model.pkl")
        joblib.dump(model, model_pkl_path)
        print(f"Exported model to: {model_pkl_path}")

        # Save run ID to a metadata file for easy reference
        run_id_path = os.path.join(script_dir, "latest_run_id.txt")
        with open(run_id_path, "w", encoding="utf-8") as f:
            f.write(run_id)

        print("\n" + "=" * 60)
        print("MLFLOW RUN SUMMARY")
        print("=" * 60)
        print(f"Experiment Name:       {experiment_name}")
        print(f"Run ID:                {run_id}")
        print(f"Accuracy:              {acc:.4f}")
        print(f"Model Artifact:        {artifact_path}")
        print(f"Model URI:             {model_uri}")
        print(f"Tracking Data Path:    {os.path.abspath('mlruns')}")
        print(f"Exported Pickle File:  {model_pkl_path}")
        print("=" * 60)

        return run_id, acc, model_uri


if __name__ == "__main__":
    run_training()
