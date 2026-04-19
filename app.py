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
        "Practical 7 — AI Model packaged with MLflow & Docker\n\n"
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

# ── Load model at startup ──────────────────────────
import glob

def find_and_load_model():
    """Find the model folder inside mlruns and load it directly."""
    # Search for the model folder regardless of run ID
    pattern = os.path.join("mlruns", "**", "iris-model", "MLmodel")
    matches = glob.glob(pattern, recursive=True)
    
    if not matches:
        print(" No model found. Run train.py first.")
        return None
    
    # Take the most recent match
    model_dir = os.path.dirname(matches[-1])
    print(f"Found model at: {model_dir}")
    
    loaded = mlflow.sklearn.load_model(model_dir)
    print("Model loaded successfully!")
    return loaded

@app.on_event("startup")
def load_model_on_startup():
    global model
    model = find_and_load_model()

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
        "service": "Iris Classifier — Practical 7",
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
