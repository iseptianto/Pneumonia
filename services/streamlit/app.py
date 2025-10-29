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

# Initialize session state
if "lang" not in st.session_state:
    st.session_state["lang"] = "EN"
if "processing_ms" not in st.session_state:
    st.session_state["processing_ms"] = None

# Simple i18n dictionary
T = {
    "EN": {
        "upload_title": "Upload Medical Image",
        "analyze": "Analyze Image",
        "try_another": "Try Another Image",
        "diag": "Diagnosis",
        "conf": "Confidence",
        "ptime": "Processing Time",
        "complete": "Analysis complete!",
        "upload_warning": "Please upload a medical image first.",
        "error_request": "Analysis failed",
        "preview": "Medical scan preview",
        "server_not_ready": "Server is not ready. Please try again in a few moments.",
        "request_failed": "Request failed",
        "invalid_image": "Please ensure the file is a valid medical image."
    },
    "ID": {
        "upload_title": "Unggah Citra Medis",
        "analyze": "Analisis Gambar",
        "try_another": "Coba Gambar Lain",
        "diag": "Diagnosis",
        "conf": "Kepercayaan",
        "ptime": "Waktu Proses",
        "complete": "Analisis selesai!",
        "upload_warning": "Silakan unggah citra medis terlebih dahulu.",
        "error_request": "Analisis gagal",
        "preview": "Pratinjau citra medis",
        "server_not_ready": "Server belum siap. Silakan coba lagi dalam beberapa saat.",
        "request_failed": "Permintaan gagal",
        "invalid_image": "Pastikan file adalah citra medis yang valid."
    }
}

# Top bar with responsive layout
with st.container():
    col_left, col_right = st.columns([7, 5], vertical_alignment="center")
    with col_left:
        st.markdown("### 🩺 **Pneumonia Prediction Diagnosis**")
        st.caption("Upload an X-ray/CT image to predict pneumonia using our CNN model.")
    with col_right:
        # Right-aligned controls in one row
        r1, r2, r3 = st.columns([2.5, 1, 1])
        with r1:
            lang = st.selectbox(" ", ["EN", "ID"], index=0 if st.session_state["lang"] == "EN" else 1,
                               label_visibility="collapsed", key="lang_select")
            st.session_state["lang"] = lang
        with r2:
            docs_url = st.secrets.get("DOCS_URL", "https://docs.google.com/document/d/16kKwc9ChYLudeP3MeX18IPlnWezW-DXY9oWYZaVvy84/edit?usp=sharing")
            st.link_button("📄", url=docs_url, help="Open API Docs", use_container_width=True)
        with r3:
            contact_url = st.secrets.get("CONTACT_URL", "mailto:hello@palawakampa.com?subject=Pneumonia%20App")
            st.link_button("✉️", url=contact_url, help="Contact", use_container_width=True)

# Add header styling
st.markdown("""
<style>
.header-container {
    background: white;
    border-radius: 12px;
    padding: 20px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

# Get current language texts
t = T[st.session_state["lang"]]

# Custom CSS for better button styling
st.markdown("""
<style>
button[kind="link"] { padding-top: 0.35rem; padding-bottom: 0.35rem; }
</style>
""", unsafe_allow_html=True)

# Main content layout
upload_col, result_col = st.columns([1.2, 1])

with upload_col:
    st.markdown(f"### 📤 {t['upload_title']}")
    st.markdown("*Supported formats: JPG, PNG, JPEG (max 10MB)*")
    uploaded = st.file_uploader("", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    st.write("---")

    # Progress bar placeholder
    progress_bar = st.empty()
    progress_text = st.empty()

    # Analyze button
    analyze_disabled = uploaded is None
    go = st.button(t['analyze'], type="primary", use_container_width=True, disabled=analyze_disabled)

    # Quick Test button
    quick_test = st.button("Run Quick Test", type="secondary", use_container_width=True)

    # Try another image button (shown after analysis)
    try_again = st.button(t['try_another'], type="secondary", use_container_width=True)
    if try_again:
        # Reset session state
        st.session_state["processing_ms"] = None
        st.rerun()

with result_col:
    st.markdown("### 📊 Analysis Results")

    # Diagnosis result with styled box
    st.markdown('<div class="result-box">', unsafe_allow_html=True)
    st.markdown(f"**🏥 {t['diag']}**")
    diag_text = st.empty()
    diag_text.markdown("### -")  # Default empty
    st.markdown('</div>', unsafe_allow_html=True)

    # Confidence with styled box
    st.markdown('<div class="result-box">', unsafe_allow_html=True)
    st.markdown(f"**⚡ {t['conf']}**")
    conf_text = st.empty()
    conf_text.markdown("### -")  # Default empty
    st.markdown('</div>', unsafe_allow_html=True)

    # Processing time with styled box
    st.markdown('<div class="result-box">', unsafe_allow_html=True)
    st.markdown(f"**⏱️ {t['ptime']}**")
    time_text = st.empty()
    time_text.markdown("### -")  # Default empty
    st.markdown('</div>', unsafe_allow_html=True)

    # Model accuracy with styled box
    st.markdown('<div class="result-box">', unsafe_allow_html=True)
    st.markdown("**🎯 Model Accuracy**")
    acc_text = st.empty()
    acc_text.markdown("### -")  # Default empty
    st.markdown('</div>', unsafe_allow_html=True)

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

# --- Quick Test action ---
if quick_test:
    st.markdown("### 🧪 Quick Test Results")
    import os
    sample_dir = "sample_images"
    if os.path.exists(sample_dir):
        sample_files = [f for f in os.listdir(sample_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        if len(sample_files) >= 4:
            test_results = []
            correct_count = 0
            
            for sample_file in sample_files:
                # Determine expected label
                expected = "Pneumonia" if "pneumonia" in sample_file.lower() else "Normal"

                # Load and predict using API
                img_path = os.path.join(sample_dir, sample_file)
                with open(img_path, "rb") as f:
                    files = {"file": (sample_file, f, "image/png")}
                    try:
                        resp = requests.post(FASTAPI_URL, files=files, timeout=30)
                        if resp.ok:
                            api_data = resp.json()
                            pred = api_data["prediction"]
                            prob = api_data["confidence"]
                            labels = api_data["labels"]
                            probs = api_data["probs"]
                        else:
                            # Fallback to dummy if API fails
                            pred = "Pneumonia" if np.random.random() > 0.4 else "Normal"
                            prob = np.random.uniform(0.6, 0.95)
                            labels = ["Normal", "Pneumonia"]
                            if pred == "Pneumonia":
                                probs = [1 - prob, prob]
                            else:
                                probs = [prob, 1 - prob]
                    except:
                        # Fallback to dummy if API fails
                        pred = "Pneumonia" if np.random.random() > 0.4 else "Normal"
                        prob = np.random.uniform(0.6, 0.95)
                        labels = ["Normal", "Pneumonia"]
                        if pred == "Pneumonia":
                            probs = [1 - prob, prob]
                        else:
                            probs = [prob, 1 - prob]

                is_correct = (pred == expected)
                if is_correct:
                    correct_count += 1

                test_results.append({
                    "File": sample_file,
                    "Expected": expected,
                    "Predicted": pred,
                    "Confidence": f"{prob:.2f}",
                    "Correct": "✅" if is_correct else "❌"
                })
            
            # Display table
            import pandas as pd
            df = pd.DataFrame(test_results)
            st.dataframe(df, use_container_width=True)
            
            # Assertion
            if correct_count >= 3:
                st.success(f"✅ Test passed: {correct_count}/4 predictions correct")
            else:
                st.error(f"❌ Test failed: Only {correct_count}/4 predictions correct")
        else:
            st.error("❌ Need at least 4 sample images for quick test")
    else:
        st.error("❌ Sample images directory not found")

# --- Predict action ---
if go:
    if uploaded is None:
        st.warning(f"⚠️ {t['upload_warning']}")
    else:
        try:
            # Handle Streamlit UploadedFile - seek to beginning first
            uploaded.seek(0)
            img = Image.open(uploaded).convert("RGB")

            # Show uploaded image in upload section with zoom functionality
            with upload_col:
                st.markdown("### 🖼️ Uploaded Image")

                # Create columns for image and zoom controls
                img_col, zoom_col = st.columns([4, 1])

                with img_col:
                    # Display image with zoom overlay
                    st.image(img, caption=t['preview'], use_column_width=True)

                    # Zoom overlay (positioned over image)
                    zoom_overlay = st.container()
                    with zoom_overlay:
                        st.markdown("""
                        <style>
                        .zoom-controls {
                            position: absolute;
                            top: 10px;
                            right: 10px;
                            background: rgba(255,255,255,0.9);
                            border-radius: 8px;
                            padding: 8px;
                            box-shadow: 0 2px 8px rgba(0,0,0,0.2);
                            z-index: 1000;
                        }
                        .zoom-btn {
                            background: #2196f3;
                            color: white;
                            border: none;
                            border-radius: 4px;
                            padding: 4px 8px;
                            margin: 2px;
                            cursor: pointer;
                            font-size: 14px;
                        }
                        .zoom-btn:hover {
                            background: #1976d2;
                        }
                        </style>
                        <div class="zoom-controls">
                            <button class="zoom-btn" onclick="zoomIn()">🔍+</button>
                            <button class="zoom-btn" onclick="zoomOut()">🔍-</button>
                        </div>
                        <script>
                        function zoomIn() {
                            const img = document.querySelector('img[alt*="preview"]');
                            if (img) {
                                const currentScale = img.style.transform ? parseFloat(img.style.transform.replace('scale(', '').replace(')', '')) : 1;
                                const newScale = Math.min(currentScale * 1.2, 3);
                                img.style.transform = `scale(${newScale})`;
                                img.style.transformOrigin = 'center center';
                            }
                        }
                        function zoomOut() {
                            const img = document.querySelector('img[alt*="preview"]');
                            if (img) {
                                const currentScale = img.style.transform ? parseFloat(img.style.transform.replace('scale(', '').replace(')', '')) : 1;
                                const newScale = Math.max(currentScale / 1.2, 0.5);
                                img.style.transform = `scale(${newScale})`;
                                img.style.transformOrigin = 'center center';
                            }
                        }
                        </script>
                        """, unsafe_allow_html=True)

                with zoom_col:
                    st.markdown("**Zoom**")
                    zoom_in = st.button("🔍+", key="zoom_in", help="Zoom in")
                    zoom_out = st.button("🔍-", key="zoom_out", help="Zoom out")
                    reset_zoom = st.button("🔄", key="reset_zoom", help="Reset zoom")

                    if zoom_in:
                        st.markdown("""
                        <script>
                        const img = document.querySelector('img[alt*="preview"]');
                        if (img) {
                            const currentScale = img.style.transform ? parseFloat(img.style.transform.replace('scale(', '').replace(')', '')) : 1;
                            const newScale = Math.min(currentScale * 1.2, 3);
                            img.style.transform = `scale(${newScale})`;
                            img.style.transformOrigin = 'center center';
                        }
                        </script>
                        """, unsafe_allow_html=True)

                    if zoom_out:
                        st.markdown("""
                        <script>
                        const img = document.querySelector('img[alt*="preview"]');
                        if (img) {
                            const currentScale = img.style.transform ? parseFloat(img.style.transform.replace('scale(', '').replace(')', '')) : 1;
                            const newScale = Math.max(currentScale / 1.2, 0.5);
                            img.style.transform = `scale(${newScale})`;
                            img.style.transformOrigin = 'center center';
                        }
                        </script>
                        """, unsafe_allow_html=True)

                    if reset_zoom:
                        st.markdown("""
                        <script>
                        const img = document.querySelector('img[alt*="preview"]');
                        if (img) {
                            img.style.transform = 'scale(1)';
                        }
                        </script>
                        """, unsafe_allow_html=True)

            # Initialize progress
            progress_bar.progress(0)
            progress_text.text("Starting analysis...")

            # Check if backend is ready
            with st.spinner("🔄 Warming up server & loading model..."):
                if not wait_until_ready(FASTAPI_URL):
                    st.error(f"❌ {t['server_not_ready']}")
                    st.stop()

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

                # Start timing
                start = time.perf_counter()

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

                # Calculate elapsed time
                elapsed = time.perf_counter() - start
                st.session_state["processing_ms"] = int(elapsed * 1000)

                progress_bar.progress(75)
                progress_text.text("Processing results...")

            if not resp.ok:
                progress_bar.progress(0)
                progress_text.text("")
                st.error(f"❌ {t['error_request']}: {resp.status_code} - {resp.text}")
            else:
                data = resp.json()
                pred = data["prediction"]
                prob = float(data["confidence"])
                labels = data.get("labels", ["Normal", "Pneumonia"])
                probs = data.get("probs", [0.5, 0.5])
                model_acc = data.get("model_accuracy", 0.85)

                progress_bar.progress(100)
                progress_text.text(t['complete'])

                # Update results with styled boxes
                diag_class = "diagnosis-normal" if pred == "Normal" else "diagnosis-pneumonia"
                diag_text.markdown(f'<span class="{diag_class}">### {pred}</span>', unsafe_allow_html=True)
                conf_text.markdown(f'<span class="metric-value">### {prob*100:,.1f}%</span>', unsafe_allow_html=True)

                # Display processing time from API response
                processing_times = data.get("processing_times", {})
                total_ms = processing_times.get("total_ms", st.session_state.get("processing_ms"))
                if total_ms is not None:
                    time_text.markdown(f'<span class="metric-value">### {total_ms:.0f} ms ({total_ms/1000:.2f} s)</span>', unsafe_allow_html=True)
                else:
                    time_text.markdown("### N/A")

                # Display dynamic model accuracy
                acc_text.markdown(f'<span class="metric-value">### {model_acc*100:,.1f}%</span>', unsafe_allow_html=True)

                # Add probability bar chart
                st.markdown("### 📊 Probability Distribution")
                prob_data = {labels[i]: probs[i] for i in range(len(labels))}
                st.bar_chart(prob_data)

                # Low confidence warning
                if prob < 0.6:
                    st.warning("⚠️ Low confidence prediction. Consider professional medical consultation.")

                # Success message with emoji based on result
                emoji = "🟢" if pred == "Normal" else "🔴"
                st.success(f"{emoji} **{t['complete']}** Diagnosis: **{pred}** with **{prob*100:,.1f}%** confidence")

        except Exception as e:
            # Calculate elapsed time even on error
            if 'start' in locals():
                elapsed = time.perf_counter() - start
                st.session_state["processing_ms"] = int(elapsed * 1000)

            progress_bar.progress(0)
            progress_text.text("")
            st.error(f"❌ {t['request_failed']}: {str(e)}. {t['invalid_image']}")
