import os
import io
import time
import base64
from typing import List

import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
from tensorflow.keras.models import load_model
import mlflow
import gdown

# utils lokal
from .utils import preprocess_pil, make_gradcam_heatmap, overlay_heatmap_on_image

# ========= Env & Config =========
MODEL_PATH = os.getenv("MODEL_PATH", "/app/models/pneumonia_resnet50_v2.h5")
GDRIVE_FILE_ID = os.getenv("GDRIVE_FILE_ID", "1ABCxyz12345")
MODEL_VERSION = os.getenv("MODEL_VERSION", "v2")
MLFLOW_ENABLED = os.getenv("MLFLOW_ENABLED", "true").lower() == "true"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "").strip()
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "pneumonia-inference")
CORS_ALLOW_ORIGINS = os.getenv("CORS_ALLOW_ORIGINS", "*").split(",")
LAST_CONV_NAME = os.getenv("LAST_CONV_NAME", "conv5_block3_out")  # For ResNet50

# ========= App =========
app = FastAPI(title="Pneumonia Inference API", version="1.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOW_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus /metrics
Instrumentator().instrument(app).expose(app)

# ========= MLflow Initialization =========
def init_mlflow():
    """Initialize MLflow with error handling. Returns True if enabled and reachable."""
    if not MLFLOW_ENABLED:
        print("[MLflow] disabled: MLFLOW_ENABLED=false")
        return False

    if not MLFLOW_TRACKING_URI:
        print("[MLflow] disabled: MLFLOW_TRACKING_URI not set")
        return False

    try:
        # Test connection with short timeout
        import requests
        test_url = MLFLOW_TRACKING_URI.rstrip('/') + "/api/2.0/mlflow/experiments/list"
        requests.head(test_url, timeout=2)

        # Set up MLflow
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        from mlflow.tracking import MlflowClient
        client = MlflowClient()

        # Ensure experiment exists
        exp = client.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
        if exp is None:
            client.create_experiment(MLFLOW_EXPERIMENT_NAME)

        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
        print(f"[MLflow] enabled: uri={MLFLOW_TRACKING_URI} exp={MLFLOW_EXPERIMENT_NAME}")
        return True

    except Exception as e:
        print(f"[MLflow] disabled: {e}")
        return False

# Global MLflow state
MLFLOW_ON = init_mlflow()


# ========= Helpers =========
def ensure_model():
    """
    Pastikan file model tersedia di MODEL_PATH.
    - Support untuk file langsung atau folder Google Drive
    - Auto-reload jika MODEL_VERSION berubah
    """
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    # Check if model exists and version matches
    version_file = MODEL_PATH + ".version"
    current_version = None
    if os.path.exists(version_file):
        try:
            with open(version_file, 'r') as f:
                current_version = f.read().strip()
        except:
            pass

    # Force re-download if version changed or model missing
    if not os.path.exists(MODEL_PATH) or current_version != MODEL_VERSION:
        print(f"📥 Downloading new model version: {MODEL_VERSION}")

        if not GDRIVE_FILE_ID:
            raise RuntimeError("Model missing and GDRIVE_FILE_ID is not set")

        # Try as direct file ID first
        try:
            url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
            gdown.download(url, MODEL_PATH, quiet=False, fuzzy=True)
            if os.path.exists(MODEL_PATH):
                # Save version info
                with open(version_file, 'w') as f:
                    f.write(MODEL_VERSION)
                print(f"✅ Model downloaded: {MODEL_PATH}")
                return
        except Exception as e:
            print(f"⚠️ Direct file download failed: {e}")

        # Try as folder ID
        try:
            folder_url = f"https://drive.google.com/drive/folders/{GDRIVE_FILE_ID}"
            out_dir = os.path.dirname(MODEL_PATH)
            gdown.download_folder(folder_url, output=out_dir, quiet=False, use_cookies=False)

            import glob, shutil
            candidates = glob.glob(os.path.join(out_dir, "**/*.h5"), recursive=True)
            if not candidates:
                raise RuntimeError("No .h5 file found after downloading the Google Drive folder")

            # Move the first .h5 file found
            shutil.move(candidates[0], MODEL_PATH)
            # Save version info
            with open(version_file, 'w') as f:
                f.write(MODEL_VERSION)
            print(f"✅ Model downloaded from folder: {MODEL_PATH}")
        except Exception as e:
            raise RuntimeError(f"Failed to download model: {e}")
    else:
        print(f"✅ Model already exists: {MODEL_PATH} (version: {MODEL_VERSION})")


# ========= Load model at startup =========
ensure_model()
model = load_model(MODEL_PATH)


# ========= Routes =========
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(file: UploadFile = File(...)):
    try:
        t0 = time.time()

        # baca & preprocess
        img = Image.open(file.file).convert("RGB")
        x = preprocess_pil(img)  # harus hasilkan shape (1, 224, 224, 3) dan scaled [0,1]

        # infer
        prob = float(model.predict(x, verbose=0)[0][0])
        label = "PNEUMONIA" if prob > 0.5 else "NORMAL"

        # Grad-CAM
        heatmap = make_gradcam_heatmap(x, model, last_conv_layer_name=LAST_CONV_NAME)
        overlay = overlay_heatmap_on_image(heatmap, img)

        # encode heatmap
        buf = io.BytesIO()
        overlay.save(buf, format="PNG")
        heatmap_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        elapsed_ms = int((time.time() - t0) * 1000)

        # log mlflow (optional, guarded)
        if MLFLOW_ON:
            try:
                with mlflow.start_run(run_name="infer"):
                    mlflow.log_param("filename", file.filename)
                    mlflow.log_metric("prob_pneumonia", prob)
                    mlflow.log_metric("elapsed_ms", elapsed_ms)
            except Exception as e:
                print(f"[MLflow] log skipped: {e}")

        return {
            "filename": file.filename,
            "prediction": label,
            "prob_pneumonia": prob,
            "time_ms": elapsed_ms,
            "heatmap_b64": heatmap_b64,
            "model_accuracy": 0.96,  # pneumonia_resnet50_v2.h5 accuracy
            "model_version": MODEL_VERSION,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")


@app.post("/predict-batch")
def predict_batch(files: List[UploadFile] = File(...)):
    try:
        results = []
        for f in files:
            img = Image.open(f.file).convert("RGB")
            x = preprocess_pil(img)
            prob = float(model.predict(x, verbose=0)[0][0])
            label = "PNEUMONIA" if prob > 0.5 else "NORMAL"
            results.append(
                {"filename": f.filename, "prediction": label, "prob_pneumonia": prob}
            )
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing batch: {str(e)}")
