# docker-mlflow-packaging
## Package and Reproduce an AI Model with Docker & MLflow

> Model Packaging, MLflow Tracking, Docker Containerization

---

## Table of Contents

1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Quick Start (Anaconda)](#quick-start)
4. [Step-by-Step Execution](#step-by-step-execution)
5. [Docker Workflow](#docker-workflow)
6. [API Reference](#api-reference)
7. [Expected Output](#expected-output)
8. [Troubleshooting](#troubleshooting)

---

## Overview

This practical demonstrates how to:

| Goal | Tool Used |
|------|-----------|
| Train a classification model | scikit-learn (RandomForest) |
| Track experiments & log models | **MLflow** |
| Auto-generate `conda.yaml` & `requirements.txt` | MLflow (automatic) |
| Serve predictions via REST API | **FastAPI** |
| Containerize for reproducibility | **Docker** |
| Verify reproducibility | `test_api.py` + Postman |

**Dataset:** Iris Flower Dataset (built-in, 150 samples, 3 classes, 4 features)

---

## Project Structure

```
docker-mlflow-packaging/
│
├── train.py            ← Train model + log everything to MLflow
├── app.py              ← FastAPI server that loads & serves the model
├── test_api.py         ← Test script to verify API endpoints
├── notebook.ipynb      ← Classroom Jupyter demo (interactive)
│
├── requirements.txt    ← pip dependencies
├── environment.yml     ← conda environment definition
│
├── Dockerfile          ← Containerize the FastAPI app
├── docker-compose.yml  ← Easy one-command Docker run
│
├── mlruns/             ← Auto-created by MLflow (do NOT delete)
├── run_id.txt          ← Auto-created by train.py
└── README.md           ← This file
```

---

## Quick Start

### Prerequisites
- [Anaconda](https://www.anaconda.com/download) installed
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed (for Docker section)

### Create Conda Environment

```bash
# Option A: From environment.yml (recommended)
conda env create -f environment.yml
conda activate rtai242p-p7

# Option B: Manual setup
conda create -n rtai242p-p7 python=3.10 -y
conda activate rtai242p-p7
pip install -r requirements.txt
```

---

## Step-by-Step Execution

### Phase 1 — Train & Track with MLflow

```bash
# Step 1: Train the model
python train.py
```

**What happens:**
- Loads Iris dataset
- Trains RandomForestClassifier
- Logs parameters, metrics, and the model to MLflow
- Creates `mlruns/` folder and `run_id.txt`

```bash
# Step 2: Open MLflow UI (in browser)
mlflow ui
# Visit: http://localhost:5000
```

**MLflow UI shows:**
- Experiment name, Run ID
- Logged parameters (n_estimators, max_depth)
- Logged metrics (accuracy, precision, recall, F1)
- Saved model artifacts including **conda.yaml** and **requirements.txt**

---

### Phase 2 — Serve Model via FastAPI

```bash
# Step 3: Start the prediction API
python app.py
# API available at: http://localhost:8000
# Swagger docs at:  http://localhost:8000/docs
```

```bash
# Step 4: Run tests (in a NEW terminal)
python test_api.py
```

---

### Phase 3 — Containerize with Docker

```bash
# Step 5: Build Docker image
docker build -t iris-mlflow-api .

# Step 6: Run Docker container
docker run -p 8000:8000 iris-mlflow-api

# OR using docker-compose (easier):
docker compose up --build
```

```bash
# Step 7: Verify reproducibility inside Docker
python test_api.py
# Same results as local = REPRODUCIBILITY DEMONSTRATED
```

---

## API Reference

Base URL: `http://localhost:8000`

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/docs` | GET | Swagger UI (interactive) |
| `/model-info` | GET | Model details |
| `/predict` | POST | Predict Iris species |

### POST /predict — Sample Request

```json
{
  "sepal_length": 5.1,
  "sepal_width":  3.5,
  "petal_length": 1.4,
  "petal_width":  0.2
}
```

### POST /predict — Sample Response

```json
{
  "predicted_class": "setosa",
  "class_index": 0,
  "confidence": 0.99,
  "all_probabilities": {
    "setosa": 0.99,
    "versicolor": 0.01,
    "virginica": 0.00
  }
}
```

---

## Expected Output

### train.py output
```
============================================================
 MLflow + Docker Demo
============================================================
[1/5] Loading Iris Dataset...
[2/5] Data Split → Train: 120 | Test: 30
[3/5] Starting MLflow Run & Training Model...
[4/5] Evaluation Results:
      Accuracy : 0.9667  (96.67%)
[5/5] MLflow Run Complete!
      Run ID   : abc123...
      Accuracy : 0.9667
```

### test_api.py output
```
Test 3: Predict Setosa
  Status Code : 200
  Response    :
    "predicted_class": "setosa"
    "confidence": 0.99

Test 4: Predict Versicolor
  predicted_class: "versicolor"

Test 5: Predict Virginica
  predicted_class: "virginica"

All tests complete!
```

---

## Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `ModuleNotFoundError: mlflow` | Env not activated | `conda activate rtai242p-p7` |
| `Model not loaded` in API | train.py not run yet | Run `python train.py` first |
| Port 8000 already in use | Another process using port | `taskkill /F /IM python.exe` or change port |
| `docker: command not found` | Docker Desktop not started | Open Docker Desktop app |
| `Error loading model` | `mlruns/` missing in Docker build | Run `train.py` before `docker build` |
| MLflow UI blank | Wrong directory | Run `mlflow ui` from project root folder |

---

## Key Concepts Summary

```
┌─────────────┐    logs to    ┌──────────────┐    packaged in   ┌──────────────┐
│  train.py   │ ────────────► │    MLflow     │ ───────────────► │    Docker    │
│  (sklearn)  │               │  (mlruns/)    │                  │  Container   │
└─────────────┘               └──────────────┘                  └──────────────┘
       │                             │                                  │
       │                    conda.yaml                          Reproducible on
       │                    requirements.txt                    any machine
       │                    model artifacts
       │
       └──────────► app.py (FastAPI) ──────────► /predict endpoint
```

---
