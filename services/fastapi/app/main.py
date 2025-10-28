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
MODEL_VERSION = os.getenv("MODEL_VERSION", "v2")
GDRIVE_FILE_ID = os.getenv("GDRIVE_FILE_ID", "").strip()
MODEL_URL = os.getenv("MODEL_URL", "").strip()
ALLOWED_MODEL_EXTS = [ext.strip().lower() for ext in os.getenv("ALLOWED_MODEL_EXTS", ".h5,.keras").split(",")]
GDOWN_USE_COOKIES = os.getenv("GDOWN_USE_COOKIES", "false").lower() == "true"
FORCE_REDOWNLOAD = os.getenv("FORCE_REDOWNLOAD", "false").lower() == "true"
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
    Robust model bootstrapping with multiple sources and graceful degradation.
    """
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    # Check version file
    version_file = os.path.join(os.path.dirname(MODEL_PATH), ".model_version")
    current_version = None
    if os.path.exists(version_file):
        try:
            with open(version_file, 'r') as f:
                current_version = f.read().strip()
        except:
            pass

    # Force re-download if requested or version mismatch
    needs_download = (not os.path.exists(MODEL_PATH) or
                     current_version != MODEL_VERSION or
                     FORCE_REDOWNLOAD)

    if needs_download:
        print(f"📥 Downloading model version: {MODEL_VERSION}")

        # Try MODEL_URL first (direct HTTP/HTTPS)
        if MODEL_URL:
            try:
                import requests
                print(f"[Model] Trying direct URL: {MODEL_URL}")
                response = requests.get(MODEL_URL, stream=True, timeout=30)
                response.raise_for_status()

                # Check file size (>1MB) and extension
                content_length = response.headers.get('content-length')
                if content_length and int(content_length) < 1024 * 1024:
                    print("[Model] URL file too small (<1MB), skipping")
                else:
                    # Check extension
                    url_path = MODEL_URL.split('?')[0].split('/')[-1].lower()
                    if any(url_path.endswith(ext) for ext in ALLOWED_MODEL_EXTS):
                        # Download to temp file first
                        temp_path = MODEL_PATH + ".tmp"
                        with open(temp_path, 'wb') as f:
                            for chunk in response.iter_content(chunk_size=8192):
                                f.write(chunk)

                        # Move to final location
                        os.rename(temp_path, MODEL_PATH)
                        with open(version_file, 'w') as f:
                            f.write(MODEL_VERSION)
                        print(f"✅ Model downloaded from URL: {MODEL_PATH}")
                        return
                    else:
                        print(f"[Model] URL extension not allowed: {url_path}")
            except Exception as e:
                print(f"[Model] URL download failed: {e}")

        # Try Google Drive
        if GDRIVE_FILE_ID:
            try:
                import gdown
            except ImportError:
                print("[Model] gdown not available, skipping Google Drive download")
                gdown = None

            if gdown:
                # Try direct file ID with retries
                for attempt in range(3):
                    try:
                        print(f"[Model] Trying Google Drive file (attempt {attempt + 1}/3)")
                        url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
                        gdown.download(url, MODEL_PATH, quiet=False, fuzzy=True, use_cookies=GDOWN_USE_COOKIES)
                        if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 1024 * 1024:
                            with open(version_file, 'w') as f:
                                f.write(MODEL_VERSION)
                            print(f"✅ Model downloaded from Google Drive file: {MODEL_PATH}")
                            return
                        else:
                            if os.path.exists(MODEL_PATH):
                                os.remove(MODEL_PATH)
                    except Exception as e:
                        error_msg = str(e)
                        if "Permission denied" in error_msg or "not publicly available" in error_msg:
                            print("[Model] Google Drive file not public. Set 'Anyone with the link' → Viewer.")
                        else:
                            print(f"[Model] File download attempt {attempt + 1} failed: {e}")
                        if attempt < 2:
                            import time
                            time.sleep(2)

                # Try folder ID
                try:
                    print("[Model] Trying Google Drive folder")
                    folder_url = f"https://drive.google.com/drive/folders/{GDRIVE_FILE_ID}"
                    tmp_dir = os.path.join(os.path.dirname(MODEL_PATH), "_gdfolder")

                    gdown.download_folder(folder_url, output=tmp_dir, quiet=False, use_cookies=GDOWN_USE_COOKIES)

                    # Find model file
                    import glob
                    candidates = []
                    for ext in ALLOWED_MODEL_EXTS:
                        candidates.extend(glob.glob(os.path.join(tmp_dir, f"**/*{ext}"), recursive=True))

                    if candidates:
                        # Pick first candidate
                        selected_file = candidates[0]
                        print(f"[Model] Selected from folder: {os.path.basename(selected_file)}")
                        os.rename(selected_file, MODEL_PATH)
                        with open(version_file, 'w') as f:
                            f.write(MODEL_VERSION)
                        print(f"✅ Model downloaded from Google Drive folder: {MODEL_PATH}")

                        # Cleanup
                        import shutil
                        shutil.rmtree(tmp_dir, ignore_errors=True)
                        return
                    else:
                        print("[Model] No valid model files found in folder")
                        if os.path.exists(tmp_dir):
                            import shutil
                            shutil.rmtree(tmp_dir, ignore_errors=True)

                except Exception as e:
                    error_msg = str(e)
                    if "Permission denied" in error_msg or "not publicly available" in error_msg:
                        print("[Model] Google Drive folder not public. Set 'Anyone with the link' → Viewer.")
                    else:
                        print(f"[Model] Folder download failed: {e}")

        print("❌ All download sources failed - starting in degraded mode")
    else:
        print(f"✅ Model already exists: {MODEL_PATH} (version: {MODEL_VERSION})")


# ========= Load model at startup (with graceful degradation) =========
ensure_model()
MODEL_READY = os.path.exists(MODEL_PATH)
model = load_model(MODEL_PATH) if MODEL_READY else None
if MODEL_READY:
    print("✅ Model loaded successfully")
else:
    print("⚠️ Model not available - starting in degraded mode")


# ========= Routes =========
@app.get("/health")
def health():
    return {"status": "ok", "model_ready": MODEL_READY}


@app.post("/predict")
def predict(file: UploadFile = File(...)):
    if not MODEL_READY:
        raise HTTPException(status_code=503, detail="Model not ready. Please try again later.")

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
    if not MODEL_READY:
        raise HTTPException(status_code=503, detail="Model not ready. Please try again later.")

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
