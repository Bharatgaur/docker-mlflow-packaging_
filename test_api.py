# ==============================================================================
# test_api.py
# Description: Simple test script to verify the FastAPI server is working.
#              Run AFTER starting app.py or Docker container.
#              Usage: python test_api.py
# ==============================================================================

import requests
import json

BASE_URL = "http://localhost:8000"

def print_section(title):
    print("\n" + "=" * 55)
    print(f"  {title}")
    print("=" * 55)

# ─────────────────────────────────────────────
# Test 1: Health Check
# ─────────────────────────────────────────────
print_section("Test 1: Health Check (GET /)")
resp = requests.get(f"{BASE_URL}/")
print(f"  Status Code : {resp.status_code}")
print(f"  Response    : {json.dumps(resp.json(), indent=4)}")

# ─────────────────────────────────────────────
# Test 2: Model Info
# ─────────────────────────────────────────────
print_section("Test 2: Model Info (GET /model-info)")
resp = requests.get(f"{BASE_URL}/model-info")
print(f"  Status Code : {resp.status_code}")
print(f"  Response    : {json.dumps(resp.json(), indent=4)}")

# ─────────────────────────────────────────────
# Test 3: Predict — Setosa
# ─────────────────────────────────────────────
print_section("Test 3: Predict Setosa")
payload = {
    "sepal_length": 5.1,
    "sepal_width":  3.5,
    "petal_length": 1.4,
    "petal_width":  0.2
}
resp = requests.post(f"{BASE_URL}/predict", json=payload)
print(f"  Input       : {payload}")
print(f"  Status Code : {resp.status_code}")
print(f"  Response    : {json.dumps(resp.json(), indent=4)}")

# ─────────────────────────────────────────────
# Test 4: Predict — Versicolor
# ─────────────────────────────────────────────
print_section("Test 4: Predict Versicolor")
payload = {
    "sepal_length": 6.4,
    "sepal_width":  3.2,
    "petal_length": 4.5,
    "petal_width":  1.5
}
resp = requests.post(f"{BASE_URL}/predict", json=payload)
print(f"  Input       : {payload}")
print(f"  Status Code : {resp.status_code}")
print(f"  Response    : {json.dumps(resp.json(), indent=4)}")

# ─────────────────────────────────────────────
# Test 5: Predict — Virginica
# ─────────────────────────────────────────────
print_section("Test 5: Predict Virginica")
payload = {
    "sepal_length": 6.3,
    "sepal_width":  3.3,
    "petal_length": 6.0,
    "petal_width":  2.5
}
resp = requests.post(f"{BASE_URL}/predict", json=payload)
print(f"  Input       : {payload}")
print(f"  Status Code : {resp.status_code}")
print(f"  Response    : {json.dumps(resp.json(), indent=4)}")

print("\n" + "=" * 55)
print(" All tests complete!")
print("=" * 55 + "\n")
