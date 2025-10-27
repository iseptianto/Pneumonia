import os, io, base64, time
import requests
import streamlit as st
from PIL import Image

FASTAPI_URL = os.getenv("FASTAPI_URL", "http://localhost:8000/predict")
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

# Top right corner controls
col1, col2, col3 = st.columns([4, 1, 1])
with col2:
    lang = st.selectbox("", ["EN", "ID"], index=0, label_visibility="collapsed")
with col3:
    st.markdown("[📄 Docs](https://docs.google.com/document/d/16kKwc9ChYLudeP3MeX18IPlnWezW-DXY9oWYZaVvy84/edit?usp=sharing)", unsafe_allow_html=True)

lang_key = "en" if lang == "EN" else "id"
t = texts[lang_key]

# --- Header ---
st.markdown(f"# {t['title']}")
st.markdown(f"*{t['description']}*")
st.write("---")

st.write("")  # spacer

# --- Input area (left) ---
c1, c2 = st.columns([1.2, 1])
with c1:
    st.markdown(f"### 📤 {t['input_placeholder']}")
    uploaded = st.file_uploader(t['file_uploader'], type=["jpg", "jpeg", "png"], label_visibility="visible")
    st.write("---")
    go = st.button(t['predict_button'], type="primary", use_container_width=True)

# --- Result area (right) ---
with c2:
    st.markdown(f"### 📊 {t['prediction_result']}")
    diag_col, acc_col = st.columns([1,1])
    with diag_col:
        st.markdown(f"**{t['diagnosis']}**")
        diag_text = st.empty()
    with acc_col:
        st.markdown("**Prediction Time**")
        acc_text = st.empty()

    st.markdown("**Model Accuracy**")
    st.caption("Accuracy from training")
    t_text = st.empty()

    st.write("")
    hc1, hc2 = st.columns([1,1])
    with hc1:
        st.markdown(f"**🔥 {t['heatmap']}**")
    with hc2:
        st.markdown(f"**🧠 {t['gradcam']}**")

# --- Predict action ---
if go:
    if uploaded is None:
        st.warning(t['upload_warning'])
    else:
        try:
            # Handle Streamlit UploadedFile - seek to beginning first
            uploaded.seek(0)
            img = Image.open(uploaded).convert("RGB")
            with c1:
                st.image(img, caption=t['preview'], use_column_width=True)

            with st.spinner("🔄 Predicting..."):
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                buf.seek(0)
                t0 = time.time()
                resp = requests.post(FASTAPI_URL, files={"file": ("image.png", buf, "image/png")}, timeout=120)
                dt = (time.time() - t0) * 1000

            if not resp.ok:
                st.error(f"{t['error_request']}: {resp.status_code} - {resp.text}")
            else:
                data = resp.json()
                pred = data["prediction"]
                prob = float(data["prob_pneumonia"])
                time_ms = int(data.get("time_ms", dt))

                # Display results
                diag_text.markdown(f"### **{pred}**")
                acc_text.markdown(f"### **{time_ms} ms**")
                t_text.write(f"**{model_acc:,.1f}%**")

                # Success message
                model_acc = data.get("model_accuracy", 0.92) * 100
                st.success(f"✅ **Diagnosis: {pred}** | **Confidence: {prob*100:,.1f}%**")

                # show Grad-CAM heatmap
                if "heatmap_b64" in data:
                    heatmap_b64 = data["heatmap_b64"]
                    heatmap_bytes = base64.b64decode(heatmap_b64)
                    heatmap_img = Image.open(io.BytesIO(heatmap_bytes)).convert("RGB")
                    with c2:
                        st.image(heatmap_img, caption=t['gradcam_caption'], use_column_width=True)
        except Exception as e:
            st.error(f"❌ Error processing image: {str(e)}. Please ensure the file is a valid image.")
