import os, io, base64, time
import requests
import streamlit as st
from PIL import Image

FASTAPI_URL = os.getenv("FASTAPI_URL", "http://localhost/api/predict")
FASTAPI_URL_BATCH = os.getenv("FASTAPI_URL_BATCH", "http://localhost/api/predict-batch")

st.set_page_config(page_title="Pneumonia Prediction Diagnosis", page_icon="🩺", layout="wide")

# --- Header & Language switcher ---
left, right = st.columns([1,1])
with left:
    st.markdown("## **Pneumonia Prediction Diagnosis**")
with right:
    lang = st.selectbox("Language / Bahasa", ["us English", "Bahasa Indonesia"], index=0)

st.write("")  # spacer

# --- Input area (left) ---
c1, c2 = st.columns([1.2, 1])
with c1:
    st.text_input(" ", value="input xray and ct scan", label_visibility="collapsed", disabled=True)
    uploaded = st.file_uploader("*) file under 10 MB", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    st.write("---")
    go = st.button("🧪 Predict", type="primary")

# --- Result area (right) ---
with c2:
    st.markdown("**Prediction Result**")
    st.caption(f"Connecting to API at : {FASTAPI_URL}")
    diag_col, acc_col = st.columns([1,1])
    with diag_col:
        st.markdown("**Diagnosis**")
        diag_text = st.empty()
    with acc_col:
        st.markdown("**Accuracy diagnosis**")
        acc_text = st.empty()

    st.markdown("**Model Usage**")
    st.caption("Prediction Time")
    t_text = st.empty()

    st.write("")
    hc1, hc2 = st.columns([1,1])
    with hc1:
        st.markdown("**heatmap**")
    with hc2:
        st.markdown("**GradCAM**")

# --- Predict action ---
if go:
    if uploaded is None:
        st.warning("Please upload an X-ray image first.")
    else:
        img = Image.open(uploaded).convert("RGB")
        with c1:
            st.image(img, caption="Preview", use_container_width=True)

        with st.spinner("Predicting..."):
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            buf.seek(0)
            t0 = time.time()
            resp = requests.post(FASTAPI_URL, files={"file": ("image.png", buf, "image/png")}, timeout=120)
            dt = (time.time() - t0) * 1000

        if not resp.ok:
            st.error(f"Request failed: {resp.status_code} - {resp.text}")
        else:
            data = resp.json()
            pred = data["prediction"]
            prob = float(data["prob_pneumonia"])
            time_ms = int(data.get("time_ms", dt))

            diag_text.markdown(f"### **{pred}**")
            acc_text.markdown(f"### **{prob*100:,.1f} %**")
            t_text.write(f"{time_ms} ms")

            # show Grad-CAM heatmap
            if "heatmap_b64" in data:
                heatmap_b64 = data["heatmap_b64"]
                heatmap_bytes = base64.b64decode(heatmap_b64)
                heatmap_img = Image.open(io.BytesIO(heatmap_bytes)).convert("RGB")
                with c2:
                    st.image(heatmap_img, caption="Grad-CAM Heatmap", use_container_width=True)
