"""
MODEL TRAINING
==============
What: Trains XGBoost classifier to predict Sales Method
Why:  This is the core ML step - model learns patterns from features
Input:  Processed CSV from feature_engineering.py
Output: Trained model (model.pkl) + all metrics logged to MLflow server
"""

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_sample_weight
import joblib
import json
import yaml
import os
import time
import mlflow
from datetime import datetime


def load_config():
    """Load configuration from YAML file"""
    with open("configs/config.yaml", "r") as f:
        return yaml.safe_load(f)


def run_training():
    """Main training function"""

    print("=" * 60)
    print("STAGE 4: MODEL TRAINING (XGBoost)")
    print("=" * 60)

    # Load config
    config = load_config()
    input_path = config["data"]["processed_local"]
    model_params = config["model"]["params"]
    model_version = config["model"]["version"]

    # -------------------------------------------
    # STEP 1: Connect to MLflow server
    # -------------------------------------------
    print("\n[Step 1] Connecting to MLflow server...")

    mlflow_uri = config["mlflow"]["tracking_uri"]
    experiment_name = config["mlflow"]["experiment_name"]

    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(experiment_name)
    print(f"         Server: {mlflow_uri}")
    print(f"         Experiment: {experiment_name}")

    # -------------------------------------------
    # STEP 2: Load features and target
    # -------------------------------------------
    print("\n[Step 2] Loading processed data...")
    df = pd.read_csv(input_path)

    # Load feature config to know which columns to use
    with open("model/feature_config.json") as f:
        feature_config = json.load(f)

    feature_columns = feature_config["feature_columns"]
    target_column = feature_config["target_column"]

    X = df[feature_columns]
    y = df[target_column]

    print(f"         Features: {X.shape} (rows, columns)")
    print(f"         Target: {y.shape}")
    print(f"         Classes: {feature_config['target_mapping']}")

    # -------------------------------------------
    # STEP 3: Train/Test split
    # -------------------------------------------
    print("\n[Step 3] Splitting data (80% train, 20% test)...")

    test_size = config["evaluation"]["test_size"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    print(f"         Train: {len(X_train)} samples")
    print(f"         Test:  {len(X_test)} samples")

    # -------------------------------------------
    # STEP 4: Handle class imbalance
    # -------------------------------------------
    print("\n[Step 4] Computing sample weights (handling class imbalance)...")

    sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)
    print(f"         Weight range: {sample_weights.min():.3f} to {sample_weights.max():.3f}")

    # -------------------------------------------
    # STEP 5: Train XGBoost + Log to MLflow
    # -------------------------------------------
    print(f"\n[Step 5] Training XGBoost...")
    print(f"         n_estimators={model_params['n_estimators']}, "
            f"max_depth={model_params['max_depth']}, lr={model_params['learning_rate']}")

    with mlflow.start_run(run_name=f"xgboost_{model_version}") as run:

        # Log parameters
        mlflow.log_param("algorithm", "XGBClassifier")
        mlflow.log_param("n_estimators", model_params["n_estimators"])
        mlflow.log_param("max_depth", model_params["max_depth"])
        mlflow.log_param("learning_rate", model_params["learning_rate"])
        mlflow.log_param("n_features", len(feature_columns))
        mlflow.log_param("train_samples", len(X_train))
        mlflow.log_param("test_samples", len(X_test))
        mlflow.log_param("class_weight", "balanced")
        mlflow.log_param("model_version", model_version)

        # Create model
        model = XGBClassifier(
            n_estimators=model_params["n_estimators"],
            max_depth=model_params["max_depth"],
            learning_rate=model_params["learning_rate"],
            objective='multi:softprob',
            num_class=3,
            eval_metric='mlogloss',
            random_state=model_params["random_state"],
        )

        # Train
        start_time = time.time()
        model.fit(
            X_train, y_train,
            sample_weight=sample_weights,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
        training_time = time.time() - start_time
        print(f"         Training completed in {training_time:.2f} seconds")

        # -------------------------------------------
        # STEP 6: Evaluate
        # -------------------------------------------
        print("\n[Step 6] Evaluating model...")

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')

        print(f"         Accuracy:      {accuracy:.4f} ({accuracy*100:.1f}%)")
        print(f"         F1 (weighted): {f1:.4f}")

        # Log metrics to MLflow
        mlflow.log_metric("accuracy", round(accuracy, 4))
        mlflow.log_metric("f1_weighted", round(f1, 4))
        mlflow.log_metric("training_time_seconds", round(training_time, 2))

        # -------------------------------------------
        # STEP 7: Cross-validation
        # -------------------------------------------
        print("\n[Step 7] Cross-validation (5-fold)...")

        cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        print(f"         CV Accuracy: {cv_mean:.4f} (+/- {cv_std:.4f})")

        mlflow.log_metric("cv_accuracy_mean", round(cv_mean, 4))
        mlflow.log_metric("cv_accuracy_std", round(cv_std, 4))

        # -------------------------------------------
        # STEP 8: Feature importance
        # -------------------------------------------
        print("\n[Step 8] Feature importance (top 5):")

        importances = dict(zip(feature_columns, model.feature_importances_.tolist()))
        sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)

        for feat, imp in sorted_imp[:5]:
            bar = "█" * int(imp * 40)
            print(f"         {feat:20s} {imp:.4f} {bar}")

        # -------------------------------------------
        # STEP 9: Save model locally
        # -------------------------------------------
        print("\n[Step 9] Saving model artifacts...")

        model_file = "model/model.pkl"
        joblib.dump(model, model_file)
        print(f"         Model saved: {model_file}")

        # Save test data for evaluation script
        X_test.to_csv("model/X_test.csv", index=False)
        y_test.to_csv("model/y_test.csv", index=False)
        print(f"         Test data saved for evaluation stage")

        # -------------------------------------------
        # STEP 10: Log model to MLflow (stores in S3)
        # -------------------------------------------
        print("\n[Step 10] Logging model to MLflow (S3 artifacts)...")

        mlflow.xgboost.log_model(model, "model")
        mlflow.log_artifact("model/feature_config.json")
        mlflow.log_artifact("model/encoders.pkl")
        mlflow.log_artifact("model/drift_baseline.json")

        run_id = run.info.run_id
        print(f"         MLflow Run ID: {run_id}")
        print(f"         View at: {mlflow_uri}/#/experiments")

        # -------------------------------------------
        # STEP 11: Save training report
        # -------------------------------------------
        report = {
            "stage": "training",
            "timestamp": datetime.now().isoformat(),
            "model_version": model_version,
            "mlflow_run_id": run_id,
            "mlflow_uri": mlflow_uri,
            "metrics": {
                "accuracy": round(accuracy, 4),
                "f1_weighted": round(f1, 4),
                "cv_accuracy_mean": round(cv_mean, 4),
                "cv_accuracy_std": round(cv_std, 4)
            },
            "parameters": model_params,
            "feature_importance_top5": {k: round(v, 4) for k, v in sorted_imp[:5]},
            "training_time_seconds": round(training_time, 2),
            "status": "SUCCESS"
        }

        with open("artifacts/training_report.json", "w") as f:
            json.dump(report, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"TRAINING COMPLETE")
    print(f"  Accuracy: {accuracy:.4f} | F1: {f1:.4f} | CV: {cv_mean:.4f}")
    print(f"  MLflow Run: {run_id}")
    print(f"  View results: {mlflow_uri}")
    print(f"{'=' * 60}")

    return model


if __name__ == "__main__":
    run_training()