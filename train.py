# ==============================================================================
# | Package and Reproduce an AI Model with Docker & MLflow
# File: train.py
# Description: Trains a RandomForest classifier on Iris dataset, tracks with MLflow,
#              and saves the model in MLflow format (ready for Docker deployment).
# ==============================================================================

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os
import json

# ─────────────────────────────────────────────
# STEP 1: Configure MLflow Tracking
# ─────────────────────────────────────────────
# MLflow will save all runs in a local folder called "mlruns"
import os
mlflow.set_tracking_uri(os.path.join(os.getcwd(), "mlruns"))          # local folder-based tracking
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("iris-classification")  # experiment name in UI

print("=" * 60)
print("| MLflow + Docker Demo")
print("=" * 60)

# ─────────────────────────────────────────────
# STEP 2: Load Dataset
# ─────────────────────────────────────────────
print("\n[1/5] Loading Iris Dataset...")
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name="species")

print(f"      Dataset shape : {X.shape}")
print(f"      Classes       : {list(iris.target_names)}")
print(f"      Features      : {list(iris.feature_names)}")

# ─────────────────────────────────────────────
# STEP 3: Split Data
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n[2/5] Data Split → Train: {len(X_train)} | Test: {len(X_test)}")

# ─────────────────────────────────────────────
# STEP 4: Train Model inside MLflow Run
# ─────────────────────────────────────────────
# Hyperparameters — easy to tune for demo
N_ESTIMATORS = 100
MAX_DEPTH     = 4
RANDOM_STATE  = 42

print("\n[3/5] Starting MLflow Run & Training Model...")

with mlflow.start_run(run_name="rf-iris-v1") as run:

    # ── Train ──────────────────────────────────
    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        random_state=RANDOM_STATE
    )
    model.fit(X_train, y_train)

    # ── Evaluate ───────────────────────────────
    y_pred   = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report   = classification_report(y_test, y_pred,
                                     target_names=iris.target_names,
                                     output_dict=True)

    print(f"\n[4/5] Evaluation Results:")
    print(f"      Accuracy : {accuracy:.4f}  ({accuracy*100:.2f}%)")
    print(f"\n      Classification Report:")
    print(classification_report(y_test, y_pred, target_names=iris.target_names))

    # ── Log Params to MLflow ───────────────────
    mlflow.log_param("n_estimators", N_ESTIMATORS)
    mlflow.log_param("max_depth",    MAX_DEPTH)
    mlflow.log_param("random_state", RANDOM_STATE)
    mlflow.log_param("test_size",    0.2)

    # ── Log Metrics to MLflow ──────────────────
    mlflow.log_metric("accuracy",          accuracy)
    mlflow.log_metric("precision_macro",   report["macro avg"]["precision"])
    mlflow.log_metric("recall_macro",      report["macro avg"]["recall"])
    mlflow.log_metric("f1_macro",          report["macro avg"]["f1-score"])

    # ── Save Report as Artifact ────────────────
    os.makedirs("artifacts", exist_ok=True)
    report_path = "artifacts/classification_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    mlflow.log_artifact(report_path)

    # ── Log the Model (MLflow format) ──────────
    # This saves the model + conda.yaml + requirements.txt automatically
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="iris-model",
        registered_model_name="IrisClassifier"
    )

    # Save Run ID for reference
    run_id = run.info.run_id
    print(f"\n[5/5] MLflow Run Complete!")
    print(f"      Run ID   : {run_id}")
    print(f"      Accuracy : {accuracy:.4f}")

    # Also save run_id to file (used by app.py)
    with open("run_id.txt", "w") as f:
        f.write(run_id)

print("\n" + "=" * 60)
print("  Training complete! Model saved via MLflow.")
print("  ▸ Open MLflow UI : mlflow ui  (then visit http://localhost:5000)")
print("  ▸ Next step      : run app.py to serve predictions")
print("=" * 60 + "\n")
