#!/usr/bin/env python3
"""
Smoke test script for Pneumonia Detection API
Tests health endpoint and prediction functionality
"""

import requests
import time
import sys
from pathlib import Path

# Configuration
API_BASE_URL = "https://pneumonia-on4f.onrender.com"
TIMEOUT = 300  # 5 minutes for cold start

def test_health():
    """Test health endpoint until model is ready."""
    print("Testing health endpoint...")
    start_time = time.time()

    while time.time() - start_time < TIMEOUT:
        try:
            resp = requests.get(f"{API_BASE_URL}/health", timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("status") == "ok":
                    print("Health check passed - model is ready!")
                    return True
                else:
                    print("Model still loading...")
            else:
                print(f"Health check returned status {resp.status_code}")
        except Exception as e:
            print(f"Health check failed: {e}")

        time.sleep(5)

    print("Health check timeout - model not ready")
    return False

def test_prediction():
    """Test prediction with a sample image."""
    print("Testing prediction endpoint...")

    # Create a simple test image (1x1 pixel PNG)
    from PIL import Image
    import io

    # Create a minimal test image
    img = Image.new('RGB', (224, 224), color='gray')
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)

    try:
        files = {"file": ("test_xray.png", buf, "image/png")}
        resp = requests.post(f"{API_BASE_URL}/predict", files=files, timeout=60)

        if resp.status_code == 200:
            data = resp.json()
            required_keys = ["prediction", "confidence", "model_accuracy"]

            if all(key in data for key in required_keys):
                print("Prediction test passed!")
                print(f"   Prediction: {data['prediction']}")
                print(".1f")
                print(".1f")
                return True
            else:
                print(f"Missing required keys in response: {data.keys()}")
        else:
            print(f"Prediction failed with status {resp.status_code}: {resp.text}")

    except Exception as e:
        print(f"Prediction test failed: {e}")

    return False

def main():
    """Run all smoke tests."""
    print("Starting Pneumonia Detection API Smoke Tests")
    print(f"API Base URL: {API_BASE_URL}")
    print("-" * 50)

    # Test 1: Health check
    if not test_health():
        print("❌ Smoke tests failed - health check")
        sys.exit(1)

    # Test 2: Prediction
    if not test_prediction():
        print("Smoke tests failed - prediction")
        sys.exit(1)
    
        print("-" * 50)
        print("All smoke tests passed!")
        print("API is ready for production use")

if __name__ == "__main__":
    main()