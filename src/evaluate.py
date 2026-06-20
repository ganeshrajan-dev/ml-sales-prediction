"""
MODEL EVALUATION
================
What: Detailed evaluation with per-class metrics + quality gate
Why:  We need to know if model is good ENOUGH for production
Input:  Trained model + test data from train.py
Output: Evaluation report + PASS/FAIL decision
Gate:   If accuracy < 80% or F1 < 75% → pipeline STOPS
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import joblib
import json
import yaml
import os
import sys
import mlflow
from datetime import datetime


def load_config():
    """Load configuration from YAML file"""
    with open("configs/config.yaml", "r") as f:
        return yaml.safe_load(f)


def run_evaluation():
    """Main evaluation function"""

    print("=" * 60)
    print("STAGE 5: MODEL EVALUATION")
    print("=" * 60)

    config = load_config()

    # -------------------------------------------
    # STEP 1: Load model and test data
    # -------------------------------------------
    print("\n[Step 1] Loading model and test data...")

    model = joblib.load("model/model.pkl")
    X_test = pd.read_csv("model/X_test.csv")
    y_test = pd.read_csv("model/y_test.csv").squeeze()

    with open("model/feature_config.json") as f:
        feature_config = json.load(f)

    target_mapping = feature_config["target_mapping"]
    class_names = list(target_mapping.keys())

    print(f"         Model loaded: model/model.pkl")
    print(f"         Test samples: {len(X_test)}")
    print(f"         Classes: {class_names}")

    # -------------------------------------------
    # STEP 2: Generate predictions
    # -------------------------------------------
    print("\n[Step 2] Generating predictions...")

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    print(f"         Predictions generated for {len(y_pred)} samples")

    # -------------------------------------------
    # STEP 3: Calculate metrics
    # -------------------------------------------
    print("\n[Step 3] Calculating metrics...")

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    print(f"\n         {'Metric':<25} {'Score':<10}")
    print(f"         {'-'*35}")
    print(f"         {'Accuracy':<25} {accuracy:.4f}")
    print(f"         {'Precision (weighted)':<25} {precision:.4f}")
    print(f"         {'Recall (weighted)':<25} {recall:.4f}")
    print(f"         {'F1 Score (weighted)':<25} {f1:.4f}")

    # -------------------------------------------
    # STEP 4: Per-class performance
    #
    # Why? Overall accuracy can hide problems.
    # Model might be 95% accurate overall but 0% on In-store.
    # Per-class metrics reveal this.
    # -------------------------------------------
    print("\n[Step 4] Per-class performance:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    # -------------------------------------------
    # STEP 5: Confusion Matrix
    #
    # Shows exactly WHERE the model makes mistakes:
    # - How many Online transactions did it wrongly call Outlet?
    # - How many In-store did it miss?
    # -------------------------------------------
    print("[Step 5] Confusion Matrix:")

    cm = confusion_matrix(y_test, y_pred)
    print(f"\n         {'':>12}", end="")
    for name in class_names:
        print(f"{name:>12}", end="")
    print("   ← Predicted")
    print(f"         {'':>12}{'-'*36}")
    for i, name in enumerate(class_names):
        print(f"         {name:>12}|", end="")
        for j in range(len(class_names)):
            print(f"{cm[i][j]:>12}", end="")
        print()
    print(f"         {'↑ Actual':>12}")

    # -------------------------------------------
    # STEP 6: Quality Gate
    #
    # This is the MOST IMPORTANT part for production:
    # Does the model meet minimum standards?
    # If NO → pipeline stops, old model stays in production
    # If YES → proceed to register and deploy
    # -------------------------------------------
    print("\n[Step 6] Quality Gate Check...")

    min_accuracy = config["evaluation"]["min_accuracy"]
    min_f1 = config["evaluation"]["min_f1_score"]

    accuracy_pass = accuracy >= min_accuracy
    f1_pass = f1 >= min_f1

    print(f"         Accuracy: {accuracy:.4f} >= {min_accuracy} → {'PASS' if accuracy_pass else 'FAIL'}")
    print(f"         F1 Score: {f1:.4f} >= {min_f1} → {'PASS' if f1_pass else 'FAIL'}")

    gate_passed = accuracy_pass and f1_pass

    # -------------------------------------------
    # STEP 7: Log to MLflow
    # -------------------------------------------
    print("\n[Step 7] Logging evaluation to MLflow...")

    mlflow_uri = config["mlflow"]["tracking_uri"]
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    # Get the latest run and log evaluation metrics to it
    with open("artifacts/training_report.json") as f:
        training_report = json.load(f)

    run_id = training_report["mlflow_run_id"]

    with mlflow.start_run(run_id=run_id):
        mlflow.log_metric("eval_accuracy", round(accuracy, 4))
        mlflow.log_metric("eval_precision", round(precision, 4))
        mlflow.log_metric("eval_recall", round(recall, 4))
        mlflow.log_metric("eval_f1_weighted", round(f1, 4))
        mlflow.log_metric("gate_passed", int(gate_passed))

    print(f"         Logged to MLflow run: {run_id}")

    # -------------------------------------------
    # STEP 8: Save evaluation report
    # -------------------------------------------
    report = {
        "stage": "evaluation",
        "timestamp": datetime.now().isoformat(),
        "model_version": config["model"]["version"],
        "metrics": {
            "accuracy": round(accuracy, 4),
            "precision_weighted": round(precision, 4),
            "recall_weighted": round(recall, 4),
            "f1_weighted": round(f1, 4)
        },
        "per_class": classification_report(
            y_test, y_pred, target_names=class_names, output_dict=True
        ),
        "confusion_matrix": cm.tolist(),
        "class_names": class_names,
        "quality_gate": {
            "min_accuracy": min_accuracy,
            "min_f1_score": min_f1,
            "accuracy_passed": accuracy_pass,
            "f1_passed": f1_pass,
            "overall": "PASSED" if gate_passed else "FAILED"
        },
        "status": "PASSED" if gate_passed else "FAILED"
    }

    with open("artifacts/evaluation_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"         Report: artifacts/evaluation_report.json")

    # -------------------------------------------
    # QUALITY GATE — Stop pipeline if failed
    # -------------------------------------------
    if not gate_passed:
        print(f"\n{'=' * 60}")
        print(f"*** QUALITY GATE FAILED ***")
        print(f"*** Model does NOT meet production standards ***")
        print(f"*** Pipeline STOPPED — old model stays in production ***")
        print(f"{'=' * 60}")
        sys.exit(1)

    print(f"\n{'=' * 60}")
    print(f"EVALUATION PASSED — Model approved for production")
    print(f"  Accuracy: {accuracy:.4f} | F1: {f1:.4f}")
    print(f"  Gate: PASSED (min accuracy={min_accuracy}, min f1={min_f1})")
    print(f"{'=' * 60}")

    return report


if __name__ == "__main__":
    run_evaluation()