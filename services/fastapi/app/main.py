from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from huggingface_hub import hf_hub_download
import asyncio, io, os
from PIL import Image
import numpy as np
import tensorflow as tf

HF_REPO_ID = "palawakampa/Pneumonia"
HF_FILENAME = "pneumonia_resnet50_v2.h5"

MODEL = None
MODEL_READY = asyncio.Event()

def _preprocess(img: Image.Image) -> np.ndarray:
    """Preprocess image for ResNet50 model."""
    img = img.convert("RGB").resize((224, 224))
    arr = np.asarray(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

async def _load_model():
    """Load model asynchronously during startup."""
    global MODEL
    print("[boot] downloading model via HF Hub...", flush=True)
    model_path = hf_hub_download(repo_id=HF_REPO_ID,
                                 filename=HF_FILENAME,
                                 cache_dir="/tmp")
    print(f"[boot] loading keras model from {model_path}", flush=True)
    MODEL = tf.keras.models.load_model(model_path)
    MODEL_READY.set()
    print("[boot] model loaded ✅", flush=True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown events."""
    asyncio.create_task(_load_model())
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
