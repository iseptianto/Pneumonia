from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from huggingface_hub import hf_hub_download
import asyncio, io, os, hashlib, time
from PIL import Image
import numpy as np
import tensorflow as tf
from prometheus_client import Gauge

HF_REPO_ID = "palawakampa/Pneumonia"
HF_FILENAME = "pneumonia_resnet50_v2.h5"

# Global state
MODEL = None
MODEL_READY = asyncio.Event()
MODEL_STATUS = {
    "ready": False,
    "last_error": None,
    "model_path": None,
    "model_exists": False,
    "model_size_bytes": 0,
    "model_sha256_8": None,
    "loaded_at": None,
    "runtime": 0.0
}

# Prometheus metrics
model_ready_gauge = Gauge('model_ready', 'Model readiness status (0=not ready, 1=ready)')

def _preprocess(img: Image.Image) -> np.ndarray:
    """Preprocess image for ResNet50 model."""
    img = img.convert("RGB").resize((224, 224))
    arr = np.asarray(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

def _calculate_sha256_8(file_path):
    """Calculate first 8 chars of SHA256 hash."""
    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)
    return sha256.hexdigest()[:8]

async def _load_model_safe():
    """Load model safely with comprehensive status tracking."""
    global MODEL, MODEL_STATUS
    start_time = time.time()

    try:
        MODEL_STATUS["last_error"] = None
        MODEL_STATUS["ready"] = False
        model_ready_gauge.set(0)

        print("[boot] downloading model via HF Hub...", flush=True)
        model_path = hf_hub_download(repo_id=HF_REPO_ID,
                                     filename=HF_FILENAME,
                                     cache_dir="/tmp")

        MODEL_STATUS["model_path"] = model_path
        MODEL_STATUS["model_exists"] = os.path.exists(model_path)
        MODEL_STATUS["model_size_bytes"] = os.path.getsize(model_path) if MODEL_STATUS["model_exists"] else 0
        MODEL_STATUS["model_sha256_8"] = _calculate_sha256_8(model_path) if MODEL_STATUS["model_exists"] else None

        print(f"[boot] loading keras model from {model_path}", flush=True)
        MODEL = tf.keras.models.load_model(model_path)

        MODEL_READY.set()
        MODEL_STATUS["ready"] = True
        MODEL_STATUS["loaded_at"] = time.time()
        MODEL_STATUS["runtime"] = time.time() - start_time
        model_ready_gauge.set(1)

        print("[boot] model loaded ✅", flush=True)

    except Exception as e:
        MODEL_STATUS["last_error"] = str(e)
        MODEL_STATUS["ready"] = False
        model_ready_gauge.set(0)
        print(f"[boot] model load failed: {e}", flush=True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown events."""
    asyncio.create_task(_load_model_safe())
    yield

app = FastAPI(title="Pneumonia Inference API", version="1.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# ---- Optional: disable MLflow by default ----
if os.getenv("ENABLE_MLFLOW", "0") != "1":
    print("[MLflow] disabled: ENABLE_MLFLOW != 1", flush=True)
else:
    # init mlflow here only if needed
    pass


@app.get("/")
def root():
    return {"message": "Pneumonia API — see /docs"}

@app.get("/health")
def health():
    return {"status": "ok"} if MODEL_READY.is_set() else {"status": "loading"}

@app.get("/_status")
def get_status():
    """Detailed model status for debugging."""
    return MODEL_STATUS.copy()

@app.post("/_warmup")
async def warmup():
    """Force model download and loading if not ready."""
    if MODEL_READY.is_set():
        return {"message": "Model already ready"}

    try:
        await _load_model_safe()
        return {"message": "Model warmup completed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Warmup failed: {str(e)}")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not MODEL_READY.is_set():
        raise HTTPException(status_code=503, detail="Model not ready. Please try again later.")
    if file.content_type not in {"image/jpeg", "image/png", "image/jpg"}:
        raise HTTPException(status_code=400, detail="File must be an image (jpg/png).")

    raw = await file.read()
    try:
        image = Image.open(io.BytesIO(raw))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    x = _preprocess(image)
    probs = MODEL.predict(x)[0]  # shape: (2,) [p_normal, p_pneumonia]
    classes = ["Normal", "Pneumonia"]
    idx = int(np.argmax(probs))
    return {"prediction": classes[idx], "confidence": float(probs[idx])}
