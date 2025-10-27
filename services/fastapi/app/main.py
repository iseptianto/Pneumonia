import os, io, time, base64
import mlflow
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
from .utils import preprocess_pil, make_gradcam_heatmap, overlay_heatmap_on_image
from prometheus_fastapi_instrumentator import Instrumentator

MODEL_PATH = os.getenv("MODEL_PATH", "/app/models/pneumonia_cnn.h5")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")

app = FastAPI(title="Pneumonia Inference API", version="1.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# Prometheus /metrics
Instrumentator().instrument(app).expose(app)

model = load_model(MODEL_PATH)
last_conv_name = "conv2d_2"  # ganti sesuai nama layer conv terakhirmu

if MLFLOW_TRACKING_URI:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("pneumonia-inference")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(file: UploadFile = File(...)):
    t0 = time.time()
    img = Image.open(file.file).convert("RGB")
    x = preprocess_pil(img)

    prob = float(model.predict(x, verbose=0)[0][0])
    label = "PNEUMONIA" if prob > 0.5 else "NORMAL"

    # Grad-CAM heatmap
    heatmap = make_gradcam_heatmap(x, model, last_conv_layer_name=last_conv_name)
    overlay = overlay_heatmap_on_image(heatmap, img)

    # to base64
    buf = io.BytesIO(); overlay.save(buf, format="PNG"); buf.seek(0)
    heatmap_b64 = base64.b64encode(buf.read()).decode("utf-8")

    elapsed_ms = int((time.time() - t0) * 1000)

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
    }

@app.post("/predict-batch")
def predict_batch(files: list[UploadFile] = File(...)):
    results = []
    for f in files:
        img = Image.open(f.file).convert("RGB")
        x = preprocess_pil(img)
        prob = float(model.predict(x, verbose=0)[0][0])
        label = "PNEUMONIA" if prob > 0.5 else "NORMAL"
        results.append({"filename": f.filename, "prediction": label, "prob_pneumonia": prob})
    return {"results": results}
