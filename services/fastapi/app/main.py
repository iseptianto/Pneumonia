import os
import io
import time
import base64
import asyncio
from contextlib import asynccontextmanager
from typing import List

import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
from huggingface_hub import hf_hub_download
import tensorflow as tf
import mlflow

# utils lokal
from .utils import preprocess_pil, make_gradcam_heatmap, overlay_heatmap_on_image

# ========= Constants =========
HF_REPO_ID = "palawakampa/Pneumonia"
HF_FILENAME = "pneumonia_resnet50_v2.h5"
LAST_CONV_NAME = "conv5_block3_out"  # For ResNet50

# ========= Global State =========
MODEL = None
MODEL_READY = asyncio.Event()

# ========= Model Loading =========
async def _load_model():
    """Load model asynchronously during startup."""
    global MODEL
    try:
        print("📥 Downloading model from Hugging Face...", flush=True)
        model_path = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_FILENAME, cache_dir="/tmp")
        print("🔄 Loading TensorFlow model...", flush=True)
        MODEL = tf.keras.models.load_model(model_path)
        MODEL_READY.set()
        print("✅ Model loaded successfully!", flush=True)
    except Exception as e:
        print(f"❌ Failed to load model: {e}", flush=True)

# ========= Lifespan =========
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown events."""
    print("🚀 Starting Pneumonia Inference API...", flush=True)
    asyncio.create_task(_load_model())
    yield
    print("🛑 Shutting down Pneumonia Inference API...", flush=True)

# ========= App =========
app = FastAPI(
    title="Pneumonia Inference API",
    version="1.1.0",
    lifespan=lifespan
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus /metrics
Instrumentator().instrument(app).expose(app)

# ========= MLflow Initialization =========
def init_mlflow():
    """Initialize MLflow only if explicitly enabled."""
    if os.getenv("ENABLE_MLFLOW", "0") == "1":
        try:
            # Import and setup MLflow
            import mlflow
            from mlflow.tracking import MlflowClient

            tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "").strip()
            experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "pneumonia-inference")

            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)
                client = MlflowClient()

                # Ensure experiment exists
                exp = client.get_experiment_by_name(experiment_name)
                if exp is None:
                    client.create_experiment(experiment_name)

                mlflow.set_experiment(experiment_name)
                print(f"[MLflow] enabled: uri={tracking_uri} exp={experiment_name}", flush=True)
                return True
            else:
                print("[MLflow] disabled: MLFLOW_TRACKING_URI not set", flush=True)
        except Exception as e:
            print(f"[MLflow] disabled: {e}", flush=True)
    else:
        print("[MLflow] disabled: ENABLE_MLFLOW != 1", flush=True)

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
async def health():
    """Health check endpoint."""
    if MODEL_READY.is_set():
        return {"status": "ok"}
    else:
        return {"status": "loading"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Predict pneumonia from uploaded X-ray image."""
    if not MODEL_READY.is_set():
        raise HTTPException(status_code=503, detail="Model not ready. Please try again later.")

    # Validate file type
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image (JPG, PNG, JPEG)")

    try:
        # Read and preprocess image
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        x = preprocess_pil(img)  # Shape: (1, 224, 224, 3), scaled [0,1]

        # Inference
        prob = float(MODEL.predict(x, verbose=0)[0][0])
        label = "Pneumonia" if prob > 0.5 else "Normal"

        return {
            "prediction": label,
            "confidence": round(prob, 4)
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")
