import os, io, base64, time
import requests
import streamlit as st
from PIL import Image
import os

FASTAPI_URL = os.getenv("FASTAPI_URL", "http://fastapi:8000/predict")
FASTAPI_URL_BATCH = os.getenv("FASTAPI_URL_BATCH", "http://localhost:8000/predict-batch")

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

st.markdown("---")

# Main content layout
upload_col, result_col = st.columns([1.2, 1])

with upload_col:
    st.markdown("### 📤 Upload Medical Image")
    st.markdown("*Supported formats: JPG, PNG, JPEG (max 10MB)*")
    uploaded = st.file_uploader("", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    st.write("---")
    go = st.button("🔍 Analyze Image", type="primary", use_container_width=True)

with result_col:
    st.markdown("### 📊 Analysis Results")

    # Diagnosis result
    st.markdown("**🏥 Diagnosis**")
    diag_text = st.empty()

    # Confidence
    st.markdown("**⚡ Confidence**")
    conf_text = st.empty()

    # Processing time
    st.markdown("**⏱️ Processing Time**")
    time_text = st.empty()

    # Model accuracy
    st.markdown("**🎯 Model Accuracy**")
    acc_text = st.empty()

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

            with st.spinner("🔄 Analyzing image with AI..."):
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                buf.seek(0)
                t0 = time.time()
                resp = requests.post(FASTAPI_URL, files={"file": ("image.png", buf, "image/png")}, timeout=120)
                dt = (time.time() - t0) * 1000

            if not resp.ok:
                st.error(f"❌ Analysis failed: {resp.status_code} - {resp.text}")
            else:
                data = resp.json()
                pred = data["prediction"]
                prob = float(data["prob_pneumonia"])
                time_ms = int(data.get("time_ms", dt))
                model_acc = data.get("model_accuracy", 0.92) * 100

                # Update results
                diag_text.markdown(f"### **{pred}**")
                conf_text.markdown(f"### **{prob*100:,.1f}%**")
                time_text.markdown(f"### **{time_ms} ms**")
                acc_text.markdown(f"### **{model_acc:,.1f}%**")

                # Success message with emoji based on result
                emoji = "🟢" if pred == "NORMAL" else "🔴"
                st.success(f"{emoji} **Analysis Complete!** Diagnosis: **{pred}** with **{prob*100:,.1f}%** confidence")

                # Show visualizations
                if "heatmap_b64" in data:
                    heatmap_b64 = data["heatmap_b64"]
                    heatmap_bytes = base64.b64decode(heatmap_b64)
                    heatmap_img = Image.open(io.BytesIO(heatmap_bytes)).convert("RGB")

                    with viz_col1:
                        heatmap_placeholder.image(heatmap_img, caption="AI attention heatmap", use_column_width=True)
                    with viz_col2:
                        gradcam_placeholder.image(heatmap_img, caption="Grad-CAM visualization", use_column_width=True)

        except Exception as e:
            st.error(f"❌ Error processing image: {str(e)}. Please ensure the file is a valid medical image.")
