import os, io, time, base64
import mlflow
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
from .utils import preprocess_pil, make_gradcam_heatmap, overlay_heatmap_on_image
from prometheus_fastapi_instrumentator import Instrumentator

MODEL_PATH = os.getenv("MODEL_PATH", "/app/models/pneumonia_cnn.h5")
GDRIVE_FILE_ID = os.getenv("GDRIVE_FILE_ID", "1gllKhGHhw0dlAqE10E5uIW1q6A3puFQd")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")

def ensure_model():
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    if not os.path.exists(MODEL_PATH):
        if not GDRIVE_FILE_ID:
            raise RuntimeError("GDRIVE_FILE_ID not set and model file missing")
        import gdown
        url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False, fuzzy=True)

# Load model at startup
ensure_model()
model = load_model(MODEL_PATH)
last_conv_name = "conv2d_2"  # Adjust based on your model's last conv layer

app = FastAPI(title="Pneumonia Inference API", version="1.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# Prometheus /metrics
Instrumentator().instrument(app).expose(app)

if MLFLOW_TRACKING_URI:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("pneumonia-inference")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(file: UploadFile = File(...)):
    try:
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
            "model_accuracy": 0.92,  # Placeholder - replace with actual model accuracy
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

@app.post("/predict-batch")
def predict_batch(files: list[UploadFile] = File(...)):
    try:
        results = []
        for f in files:
            img = Image.open(f.file).convert("RGB")
            x = preprocess_pil(img)
            prob = float(model.predict(x, verbose=0)[0][0])
            label = "PNEUMONIA" if prob > 0.5 else "NORMAL"
            results.append({"filename": f.filename, "prediction": label, "prob_pneumonia": prob})
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing batch: {str(e)}")
