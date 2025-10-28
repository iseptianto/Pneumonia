import os, io, base64, time
import requests
import streamlit as st
from PIL import Image

FASTAPI_URL = os.getenv("FASTAPI_URL", "https://pneumonia-on4f.onrender.com/predict")
FASTAPI_URL_BATCH = os.getenv("FASTAPI_URL_BATCH", "https://pneumonia-on4f.onrender.com/predict-batch")

def wait_until_ready(base_url, timeout=120, interval=2):
    """Wait for FastAPI backend to be ready."""
    import time
    health_url = base_url.replace('/predict', '/health')
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            resp = requests.get(health_url, timeout=5)
            if resp.status_code == 200 and resp.json().get("status") == "ok":
                return True
        except:
            pass
        time.sleep(interval)
    return False

st.set_page_config(page_title="Pneumonia Prediction Diagnosis", page_icon="🩺", layout="wide")

# Language dictionary
texts = {
    "en": {
        "title": "🩺 Pneumonia Prediction Diagnosis",
        "description": "Upload an X-ray or CT scan image to predict pneumonia using AI. This app uses a ResNet50 CNN model trained on medical imaging data for accurate diagnosis.",
        "language": "Language",
        "input_placeholder": "Upload X-ray or CT scan image",
        "file_uploader": "📤 Upload image (JPG, PNG, max 10MB)",
        "predict_button": "🔍 Predict",
        "prediction_result": "Prediction Result",
        "connecting_api": "🔗 Connecting to API at:",
        "diagnosis": "Diagnosis",
        "accuracy": "Confidence",
        "model_usage": "Model Usage",
        "prediction_time": "Prediction Time",
        "heatmap": "Heatmap",
        "gradcam": "Grad-CAM",
        "upload_warning": "⚠️ Please upload an X-ray image first.",
        "error_request": "❌ Request failed",
        "preview": "Image Preview",
        "gradcam_caption": "Grad-CAM Heatmap"
    },
    "id": {
        "title": "🩺 Diagnosis Prediksi Pneumonia",
        "description": "Unggah gambar X-ray atau CT scan untuk memprediksi pneumonia menggunakan AI. Aplikasi ini menggunakan model CNN ResNet50 yang dilatih pada data pencitraan medis untuk diagnosis yang akurat.",
        "language": "Bahasa",
        "input_placeholder": "Unggah gambar X-ray atau CT scan",
        "file_uploader": "📤 Unggah gambar (JPG, PNG, maks 10MB)",
        "predict_button": "🔍 Prediksi",
        "prediction_result": "Hasil Prediksi",
        "connecting_api": "🔗 Terhubung ke API di:",
        "diagnosis": "Diagnosis",
        "accuracy": "Tingkat Kepercayaan",
        "model_usage": "Penggunaan Model",
        "prediction_time": "Waktu Prediksi",
        "heatmap": "Heatmap",
        "gradcam": "Grad-CAM",
        "upload_warning": "⚠️ Silakan unggah gambar X-ray terlebih dahulu.",
        "error_request": "❌ Permintaan gagal",
        "preview": "Pratinjau Gambar",
        "gradcam_caption": "Heatmap Grad-CAM"
    }
}

# Header with left-aligned title and right-aligned controls
header_col1, header_col2 = st.columns([3, 1])
with header_col1:
    st.markdown("# 🩺 Pneumonia Prediction Diagnosis")
with header_col2:
    controls_col1, controls_col2, controls_col3 = st.columns(3)
    with controls_col1:
        lang = st.selectbox("", ["EN", "ID"], index=0, label_visibility="collapsed")
    with controls_col2:
        st.markdown("[📄](https://docs.google.com/document/d/16kKwc9ChYLudeP3MeX18IPlnWezW-DXY9oWYZaVvy84/edit?usp=sharing)", unsafe_allow_html=True)
    with controls_col3:
        st.markdown("[📞](https://wa.me/628983776946)", unsafe_allow_html=True)

lang_key = "en" if lang == "EN" else "id"
t = texts[lang_key]

st.markdown("*Upload an X-ray or CT scan image to predict pneumonia using AI. This app uses a ResNet50 CNN model trained on medical imaging data for accurate diagnosis.*")

# Add custom CSS for medical blue theme and drag-drop styling
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #e3f2fd 0%, #f3e5f5 100%);
        min-height: 100vh;
    }
    .stFileUploader {
        background: #ffffff;
        border: 2px dashed #2196f3;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        transition: all 0.3s ease;
    }
    .stFileUploader:hover {
        border-color: #1976d2;
        background: #f8f9fa;
    }
    .result-box {
        background: #ffffff;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    .diagnosis-normal {
        color: #4caf50;
        font-weight: bold;
    }
    .diagnosis-pneumonia {
        color: #f44336;
        font-weight: bold;
    }
    .metric-value {
        font-size: 1.2em;
        font-weight: bold;
        color: #2196f3;
    }
    .progress-bar {
        width: 100%;
        height: 20px;
        background: #e0e0e0;
        border-radius: 10px;
        overflow: hidden;
        margin: 10px 0;
    }
    .progress-fill {
        height: 100%;
        background: linear-gradient(90deg, #2196f3, #21cbf3);
        width: 0%;
        transition: width 0.3s ease;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("---")

# Main content layout
upload_col, result_col = st.columns([1.2, 1])

with upload_col:
    st.markdown("### 📤 Upload Medical Image")
    st.markdown("*Supported formats: JPG, PNG, JPEG (max 10MB)*")
    uploaded = st.file_uploader("", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    st.write("---")

    # Progress bar placeholder
    progress_bar = st.empty()
    progress_text = st.empty()

    go = st.button("🔍 Analyze Image", type="primary", use_container_width=True)

    # Try another image button (shown after analysis)
    try_again = st.button("🔄 Try Another Image", type="secondary", use_container_width=True)
    if try_again:
        st.rerun()

with result_col:
    st.markdown("### 📊 Analysis Results")

    # Diagnosis result with styled box
    st.markdown('<div class="result-box">', unsafe_allow_html=True)
    st.markdown("**🏥 Diagnosis**")
    diag_text = st.empty()
    st.markdown('</div>', unsafe_allow_html=True)

    # Confidence with styled box
    st.markdown('<div class="result-box">', unsafe_allow_html=True)
    st.markdown("**⚡ Confidence**")
    conf_text = st.empty()
    st.markdown('</div>', unsafe_allow_html=True)

    # Processing time with styled box
    st.markdown('<div class="result-box">', unsafe_allow_html=True)
    st.markdown("**⏱️ Processing Time**")
    time_text = st.empty()
    st.markdown('</div>', unsafe_allow_html=True)

    # Model accuracy with styled box
    st.markdown('<div class="result-box">', unsafe_allow_html=True)
    st.markdown("**🎯 Model Accuracy**")
    acc_text = st.empty()
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    # Visualization section
    st.markdown("### 🔍 AI Analysis Visualization")
    viz_col1, viz_col2 = st.columns(2)
    with viz_col1:
        st.markdown("**🔥 Heatmap**")
        heatmap_placeholder = st.empty()
    with viz_col2:
        st.markdown("**🧠 Grad-CAM**")
        gradcam_placeholder = st.empty()

# --- Predict action ---
if go:
    if uploaded is None:
        st.warning("⚠️ Please upload a medical image first.")
    else:
        try:
            # Handle Streamlit UploadedFile - seek to beginning first
            uploaded.seek(0)
            img = Image.open(uploaded).convert("RGB")

            # Show uploaded image in upload section
            with upload_col:
                st.markdown("### 🖼️ Uploaded Image")
                st.image(img, caption="Medical scan preview", use_column_width=True)

            # Check if backend is ready
            with st.spinner("🔄 Warming up server & loading model..."):
                if not wait_until_ready(FASTAPI_URL):
                    st.error("❌ Server is not ready. Please try again in a few moments.")
                    return

            # Initialize progress
            progress_bar.progress(0)
            progress_text.text("Starting analysis...")

            with st.spinner("🔄 Analyzing image with AI..."):
                # Simulate progress steps
                progress_bar.progress(25)
                progress_text.text("Preprocessing image...")

                buf = io.BytesIO()
                img.save(buf, format="PNG")
                buf.seek(0)

                progress_bar.progress(50)
                progress_text.text("Sending to AI model...")

                t0 = time.time()

                # Try prediction with exponential backoff for 503 errors
                max_retries = 5
                for attempt in range(max_retries):
                    try:
                        resp = requests.post(FASTAPI_URL, files={"file": ("image.png", buf, "image/png")}, timeout=120)
                        if resp.status_code != 503:
                            break
                        elif attempt < max_retries - 1:
                            wait_time = 2 ** attempt  # 2, 3, 5, 8, 13 seconds
                            progress_text.text(f"Model loading... retrying in {wait_time}s")
                            time.sleep(wait_time)
                    except Exception as e:
                        if attempt == max_retries - 1:
                            raise e

                dt = (time.time() - t0) * 1000
                progress_bar.progress(75)
                progress_text.text("Processing results...")

            if not resp.ok:
                progress_bar.progress(0)
                progress_text.text("")
                st.error(f"❌ Analysis failed: {resp.status_code} - {resp.text}")
            else:
                data = resp.json()
                pred = data["prediction"]
                prob = float(data["confidence"])

                progress_bar.progress(100)
                progress_text.text("Analysis complete!")

                # Update results with styled boxes
                diag_class = "diagnosis-normal" if pred == "Normal" else "diagnosis-pneumonia"
                diag_text.markdown(f'<span class="{diag_class}">### {pred}</span>', unsafe_allow_html=True)
                conf_text.markdown(f'<span class="metric-value">### {prob*100:,.1f}%</span>', unsafe_allow_html=True)
                time_text.markdown("### N/A")  # No time returned from simplified API
                acc_text.markdown("### 96.0%")  # Fixed model accuracy

                # Success message with emoji based on result
                emoji = "🟢" if pred == "Normal" else "🔴"
                st.success(f"{emoji} **Analysis Complete!** Diagnosis: **{pred}** with **{prob*100:,.1f}%** confidence")

        except Exception as e:
            progress_bar.progress(0)
            progress_text.text("")
            st.error(f"❌ Error processing image: {str(e)}. Please ensure the file is a valid medical image.")
