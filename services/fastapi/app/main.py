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
MODEL_PATH = os.getenv("MODEL_PATH", "/app/models/pneumonia_cnn.h5")
GDRIVE_FILE_ID = os.getenv("GDRIVE_FILE_ID", "1gllKhGHhw0dlAqE10E5uIW1q6A3puFQd")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "")
CORS_ALLOW_ORIGINS = os.getenv("CORS_ALLOW_ORIGINS", "*").split(",")
LAST_CONV_NAME = os.getenv("LAST_CONV_NAME", "conv2d_2")  # sesuaikan dgn model

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

# MLflow (opsional)
if MLFLOW_TRACKING_URI:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("pneumonia-inference")


# ========= Helpers =========
def ensure_model():
    """
    Pastikan file model tersedia di MODEL_PATH.
    - Jika GDRIVE_FILE_ID adalah file: download langsung.
    - Jika folder ID: download folder lalu ambil .h5 pertama.
    """
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    if os.path.exists(MODEL_PATH):
        return

    if not GDRIVE_FILE_ID:
        raise RuntimeError("Model missing and GDRIVE_FILE_ID is not set")

    # Coba sebagai FILE ID dulu
    try:
        url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False, fuzzy=True)
        if os.path.exists(MODEL_PATH):
            return
    except Exception:
        pass  # lanjut coba sebagai folder

    # Jika folder ID → download folder lalu cari .h5
    folder_url = f"https://drive.google.com/drive/folders/{GDRIVE_FILE_ID}"
    out_dir = os.path.dirname(MODEL_PATH)
    gdown.download_folder(folder_url, output=out_dir, quiet=False, use_cookies=False)

    import glob, shutil
    candidates = glob.glob(os.path.join(out_dir, "**/*.h5"), recursive=True)
    if not candidates:
        raise RuntimeError("No .h5 file found after downloading the Google Drive folder")
    # ambil pertama
    shutil.move(candidates[0], MODEL_PATH)


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

        # log mlflow (opsional)
        if MLFLOW_TRACKING_URI:
            with mlflow.start_run(run_name="infer"):
                mlflow.log_param("filename", file.filename)
                mlflow.log_metric("prob_pneumonia", prob)
                mlflow.log_metric("elapsed_ms", elapsed_ms)

        return {
            "filename": file.filename,
            "prediction": label,
            "prob_pneumonia": prob,
            "time_ms": elapsed_ms,
            "heatmap_b64": heatmap_b64,
            "model_accuracy": 0.85,  # pneumonia_cnn.h5 model accuracy
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
