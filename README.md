# 🩺 Pneumonia Detection — API & UI

> Sistem end-to-end deteksi pneumonia dari foto X-ray menggunakan model CNN (ResNet50) — tersedia sebagai API (FastAPI) dan antarmuka pengguna (Streamlit). Model dihosting di Hugging Face, aplikasi dapat dideploy di Render.com atau dijalankan secara lokal/dengan Docker.

---

## 📌 Ringkasan singkat
Pneumonia adalah penyakit pernapasan serius. Proyek ini menyediakan layanan inference yang mengklasifikasikan citra X-ray dada menjadi dua kelas: PNEUMONIA atau NORMAL. Selain prediksi, sistem juga menyediakan visualisasi Grad-CAM untuk membantu interpretabilitas.

Hal-hal yang sudah ada di repo:
- Backend inference: services/fastapi (FastAPI)
- Frontend UI: services/streamlit (Streamlit)
- Skrip pengujian: scripts/
- Model terhosting: palawakampa/Pneumonia (Hugging Face)

---

## ✨ Fitur utama
- Deteksi Pneumonia vs Normal menggunakan ResNet50 dengan head klasifikasi kustom
- Output: label, confidence, processing time, heatmap Grad-CAM (base64)
- Streamlit UI dengan drag-and-drop dan progress bar
- Endpoint batch untuk memproses banyak gambar sekaligus
- Penanganan cold-start (model di-download saat awal)
- Contoh konfigurasi untuk Render.com, Docker Compose untuk pengembangan

---

## 🔖 Badges (opsional)
Tambahkan badge sesuai kebutuhan:
- Build / CI
- License
- Python version
- Tested with (contoh): GitHub Actions / Render status

---

## Prasyarat
- Python 3.8+ (direkomendasikan 3.9/3.10)
- pip atau conda
- Docker & docker-compose (opsional, untuk mode container)
- Akses internet (untuk mengunduh model dari Hugging Face jika belum ada lokal)

---

## Struktur repo (ringkasan)
- services/fastapi — backend FastAPI (endpoints, model loading, Grad-CAM)
- services/streamlit — antarmuka Streamlit (UI)
- scripts/ — utilitas (contoh: smoke_test.py)
- README.md — dokumen ini
- LICENSE — lisensi MIT

---

## Cara menjalankan (Panduan terperinci)

Pilihan A — Jalankan secara lokal (virtualenv / pip)
1. Clone repo
   ```bash
   git clone https://github.com/iseptianto/Pneumonia.git
   cd Pneumonia
   ```
2. Buat virtual environment
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Linux/Mac
   .venv\Scripts\activate      # Windows
   ```
3. Install dependensi
   ```bash
   pip install -r requirements.txt
   ```
4. Jalankan API (FastAPI)
   ```bash
   cd services/fastapi
   uvicorn main:app --host 0.0.0.0 --port 8000
   ```
   - Endpoint health: GET http://localhost:8000/health
   - Endpoint predict: POST http://localhost:8000/predict

5. Jalankan UI (Streamlit) pada terminal berbeda
   ```bash
   cd services/streamlit
   export FASTAPI_URL=http://localhost:8000/predict
   streamlit run streamlit_app.py --server.port 8501
   ```
   - Buka http://localhost:8501

Pilihan B — Jalankan dengan Docker Compose (direkomendasikan untuk dev)
1. Pastikan Docker dan docker-compose terpasang
2. Jalankan
   ```bash
   docker-compose up --build -d
   ```
3. Akses:
   - API: http://localhost:8000
   - Streamlit: http://localhost:8501
   - (Opsional) MLflow UI jika diaktifkan: http://localhost:5001

Pilihan C — Deploy di Render.com
- Gunakan file konfigurasi sample (render.yaml) dan set environment variables (contoh ada di bawah).
- Contoh startCommand yang dipakai:
  - API: `uvicorn fastapi_app.main:app --host 0.0.0.0 --port $PORT`
  - UI: `streamlit run streamlit_app.py --server.port $PORT --server.address 0.0.0.0`

---

## Endpoint API (ringkasan)
Health check
```http
GET /health
```
- Output contoh:
```json
{ "status": "ok", "model_ready": true }
```

Single prediction
```http
POST /predict
Content-Type: multipart/form-data
file: <image_file>
```
Response (contoh)
```json
{
  "prediction": "PNEUMONIA",            // atau "NORMAL"
  "confidence": 0.87,
  "processing_time_ms": 245,
  "model_accuracy": 0.96,               // metadata model (opsional)
  "model_version": "v2",
  "heatmap_b64": "<base64_encoded_image>"
}
```

Batch prediction
```http
POST /predict-batch
Content-Type: multipart/form-data
files: <multiple_image_files>
```
- Mengembalikan daftar hasil per file.

---

## Contoh penggunaan API

Python (requests)
```python
import requests

files = {"file": open("xray.jpg", "rb")}
resp = requests.post("https://pneumonia-on4f.onrender.com/predict", files=files)
print(resp.json())
```

curl
```bash
curl -F file=@xray.jpg https://pneumonia-on4f.onrender.com/predict
```

Mendecode heatmap (base64) di Python
```python
import base64
from PIL import Image
from io import BytesIO

data = resp.json()
img_bytes = base64.b64decode(data["heatmap_b64"])
img = Image.open(BytesIO(img_bytes))
img.show()
```

---

## Detail Model
- Arsitektur dasar: ResNet50 + head klasifikasi kustom
- Input: gambar ukuran 224×224 (RGB), range dinormalisasi ke [0,1]
- Output: Probabilitas untuk kelas PNEUMONIA / NORMAL
- Model ter-hosting: palawakampa/Pneumonia, file: pneumonia_resnet50_v2.h5
  - Jika model belum tersedia lokal, aplikasi akan men-download dari Hugging Face saat cold-start (~30–60 detik tergantung koneksi).

Catatan: Jika Anda memiliki metrik validasi/training yang lebih lengkap (precision, recall, F1, confusion matrix), tambahkan di bagian Model Performance.

---

## Perilaku Cold Start
- Render free-tier dapat menyebabkan cold start:
  1. Saat container baru berjalan, endpoint health bisa mengembalikan {"status":"loading"}.
  2. Aplikasi men-download model dari Hugging Face (30–60 detik).
  3. Setelah model siap, health mengembalikan {"status":"ok","model_ready":true}.
- UI (Streamlit) menunggu readiness sebelum menampilkan hasil, implementasikan retry/backoff jika perlu.

---

## Environment Variables (penting)
Contoh variabel yang digunakan oleh service:
- MODEL_REPO_ID — (default) palawakampa/Pneumonia
- MODEL_FILENAME — pneumonia_resnet50_v2.h5
- CORS_ALLOW_ORIGINS — default "*"
- FASTAPI_URL — URL endpoint API untuk Streamlit
- FASTAPI_URL_BATCH — URL batch endpoint
- MLFLOW_ENABLED — "true" / "false"
- MLFLOW_TRACKING_URI — URL MLflow (opsional)
- GDRIVE_FILE_ID — (opsional) jika model disimpan di Google Drive
- MODEL_VERSION — tag/version string

Pastikan variabel di-set pada hosting (Render, Easypanel, atau .env saat lokal).

---

## Pengujian (Testing)
- Smoke test dasar:
  ```bash
  python scripts/smoke_test.py
  ```
  Test mencakup:
  - Health endpoint
  - Single prediction
  - Validasi respons (skema JSON)

- Untuk CI/CD: tambahkan test yang memanggil endpoint di runner dan memverifikasi shape/tipe data hasil.

---

## Observabilitas & Metrics
- Tersedia endpoint Prometheus (jika di-enable) `GET /metrics`
- Opsional: integrasi MLflow untuk tracking model dan eksperimen (lihat services/fastapi/README.md)

---

## Keamanan & Validasi
- Input file di-validasi: ekstensi, ukuran, dan tipe MIME
- CORS dikonfigurasi melalui CORS_ALLOW_ORIGINS
- Jangan menyimpan data sensitif pasien dalam repositori ini
- Untuk produksi: gunakan HTTPS/TLS, batasi CORS domain, dan tambahkan autentikasi jika diperlukan

---

## Debugging & Troubleshooting (masalah umum)
- Model tidak terunduh / network error:
  - Cek MODEL_REPO_ID dan koneksi internet di host
  - Lihat log FastAPI untuk tracing download
- Cold start terlalu lama:
  - Pastikan model di-cache (volume persisten atau pre-warmed instance)
  - Gunakan model yang lebih ringan / quantization jika perlu
- Hasil prediksi tidak sesuai:
  - Verifikasi preprocessing gambar (resize, normalisasi)
  - Periksa versi model (MODEL_VERSION)
- Streamlit tidak dapat terhubung ke API:
  - Pastikan FASTAPI_URL benar dan endpoint /predict reachable dari hosting UI

---

## Kontribusi
Terima kasih atas kontribusi! Alur kontribusi yang disarankan:
1. Fork repository
2. Buat branch fitur: `git checkout -b feat/your-feature`
3. Tambahkan test jika relevan
4. Commit dan push branch Anda
5. Ajukan Pull Request dengan deskripsi perubahan dan run smoke tests

Kode gaya:
- Gunakan flake8/black (jika ada konfigurasi), sertakan tests untuk logic baru

---

## Lisensi
MIT License — lihat file LICENSE untuk detail.

---

## Kontak & Dukungan
- Buat issue di GitHub untuk bug/perbaikan/fitur
- Dokumen tambahan: [Google Docs (proyek)](https://docs.google.com/document/d/16kKwc9ChYLudeP3MeX18IPlnWezW-DXY9oWYZaVvy84/edit?usp=sharing)
- Contact / WhatsApp: https://wa.me/628983776946

---

## Catatan tambahan & rekomendasi
- Tambahkan bagian DATASET yang menjelaskan sumber data X-ray (lisensi dataset, split train/val/test) — penting untuk kepatuhan etika.
- Cantumkan metrik evaluasi lebih lengkap (precision, recall, F1) dan sample confusion matrix.
- Pertimbangkan menambahkan contoh gambar input dan output heatmap di folder docs/ atau README (gunakan subfolder `assets/`).
- Jika ingin, saya bisa:
  - Membuat branch baru dan commit README yang diperbarui, atau
  - Membuat PR dengan perbaikan terperinci ini.

Jika Anda ingin saya langsung commit perbaikan README ke repo, tuliskan: "commit ke branch <nama-branch>" — saya akan membuat branch dan push perubahan (mohon sebutkan nama branch yang diinginkan).  
Jika belum, konfirmasi jika ada bagian yang mau Anda ubah bahasa/tingkat teknisnya sebelum saya commit.
