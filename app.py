# ==============================================================================
# Package and Reproduce an AI Model with Docker & MLflow
# File: app.py
# Description: FastAPI server that loads the MLflow-logged model and serves
#              predictions via REST API. Works both locally and inside Docker.
# ==============================================================================

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import mlflow.sklearn
import mlflow
import numpy as np
import os

# ─────────────────────────────────────────────
# App Setup
# ─────────────────────────────────────────────
app = FastAPI(
    title="Iris Classifier API",
    description=(
        "RTAI-242P Practical 7 — AI Model packaged with MLflow & Docker\n\n"
        "Send flower measurements and get the predicted Iris species."
    ),
    version="1.0.0"
)

# ─────────────────────────────────────────────
# Load Model at Startup
# ─────────────────────────────────────────────
CLASS_NAMES = ["setosa", "versicolor", "virginica"]

# Allow override via environment variable (used inside Docker)
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "mlruns")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Try loading the registered model; fall back to latest run
try:
    model_uri = "models:/IrisClassifier/latest"
    model = mlflow.sklearn.load_model(model_uri)
    print(f"Model loaded from registry: {model_uri}")
except Exception:
    # Fallback: load from run_id.txt written during training
    try:
        with open("run_id.txt") as f:
            run_id = f.read().strip()
        model_uri = f"runs:/{run_id}/iris-model"
        model = mlflow.sklearn.load_model(model_uri)
        print(f"Model loaded from run: {model_uri}")
    except Exception as e:
        print(f"Could not load model: {e}")
        model = None

# ─────────────────────────────────────────────
# Request / Response Schemas
# ─────────────────────────────────────────────
class IrisFeatures(BaseModel):
    """Input: four Iris flower measurements (in cm)."""
    sepal_length: float = Field(..., example=5.1, description="Sepal length (cm)")
    sepal_width:  float = Field(..., example=3.5, description="Sepal width (cm)")
    petal_length: float = Field(..., example=1.4, description="Petal length (cm)")
    petal_width:  float = Field(..., example=0.2, description="Petal width (cm)")

class PredictionResponse(BaseModel):
    predicted_class:  str
    class_index:      int
    confidence:       float
    all_probabilities: dict

# ─────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────
@app.get("/", summary="Health Check")
def root():
    """Simple health-check endpoint."""
    return {
        "status": "running",
        "service": "Iris Classifier — RTAI-242P Practical 7",
        "model_loaded": model is not None,
        "docs": "/docs"
    }

@app.post("/predict", response_model=PredictionResponse, summary="Predict Iris Species")
def predict(features: IrisFeatures):
    """
    Predict the Iris species given four flower measurements.

    **Example inputs:**
    - Setosa:     sepal_length=5.1, sepal_width=3.5, petal_length=1.4, petal_width=0.2
    - Versicolor: sepal_length=6.4, sepal_width=3.2, petal_length=4.5, petal_width=1.5
    - Virginica:  sepal_length=6.3, sepal_width=3.3, petal_length=6.0, petal_width=2.5
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run train.py first.")

    # Build feature array
    X = np.array([[
        features.sepal_length,
        features.sepal_width,
        features.petal_length,
        features.petal_width
    ]])

    # Predict
    class_idx   = int(model.predict(X)[0])
    proba       = model.predict_proba(X)[0]
    confidence  = float(proba[class_idx])

    return PredictionResponse(
        predicted_class=CLASS_NAMES[class_idx],
        class_index=class_idx,
        confidence=round(confidence, 4),
        all_probabilities={
            name: round(float(p), 4)
            for name, p in zip(CLASS_NAMES, proba)
        }
    )

@app.get("/model-info", summary="Model Info")
def model_info():
    """Returns information about the loaded model."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    return {
        "model_type": type(model).__name__,
        "n_estimators": getattr(model, "n_estimators", "N/A"),
        "max_depth": getattr(model, "max_depth", "N/A"),
        "classes": CLASS_NAMES,
        "features": [
            "sepal length (cm)", "sepal width (cm)",
            "petal length (cm)", "petal width (cm)"
        ]
    }

# ─────────────────────────────────────────────
# Run directly (local dev)
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
