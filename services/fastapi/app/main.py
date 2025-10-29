from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import hf_hub_download
from contextlib import asynccontextmanager
import tflite_runtime.interpreter as tflite
import numpy as np, asyncio, io
from PIL import Image

HF_REPO_ID = "palawakampa/Pneumonia"
HF_FILENAME = "pneumonia_resnet50_v2_fp16.tflite"

INTERP = None
IN_DET = None
OUT_DET = None
READY = asyncio.Event()

def preprocess(img: Image.Image, size=(224,224)):
    img = img.convert("RGB").resize(size)
    x = np.asarray(img).astype("float32")/255.0
    return np.expand_dims(x, 0)

async def boot():
    global INTERP, IN_DET, OUT_DET
    path = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_FILENAME, cache_dir="/tmp")
    INTERP = tflite.Interpreter(model_path=path, num_threads=1)
    INTERP.allocate_tensors()
    IN_DET = INTERP.get_input_details()[0]
    OUT_DET = INTERP.get_output_details()[0]
    READY.set()

@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(boot())
    yield

app = FastAPI(title="Pneumonia Inference API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok" if READY.is_set() else "loading"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not READY.is_set():
        raise HTTPException(503, "Model not ready. Please try again later.")
    if file.content_type not in {"image/jpeg","image/png","image/jpg"}:
        raise HTTPException(400, "File must be jpg/png.")

    img = Image.open(io.BytesIO(await file.read()))
    x = preprocess(img)
    INTERP.set_tensor(IN_DET["index"], x)
    INTERP.invoke()
    probs = INTERP.get_tensor(OUT_DET["index"])[0]

    classes = ["Normal","Pneumonia"]  # sesuaikan urutan output
    idx = int(np.argmax(probs))
    return {"prediction": classes[idx], "confidence": float(probs[idx])}
