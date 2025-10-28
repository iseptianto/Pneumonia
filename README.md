# 🩺 Pneumonia Detection API & UI

End-to-end pneumonia detection system using ResNet50 CNN deployed on Render.com with Hugging Face model hosting.

## 🚀 Features

- **AI-Powered Diagnosis**: ResNet50 CNN model for pneumonia detection from X-ray images
- **Real-time Processing**: Fast inference with Grad-CAM visualization
- **Production Ready**: Deployed on Render.com with automatic scaling
- **User-Friendly UI**: Streamlit interface with drag-and-drop upload
- **Model Reliability**: Graceful degradation and readiness checks

## 📋 API Endpoints

### Health Check
```http
GET /health
```
Returns model readiness status.

### Single Prediction
```http
POST /predict
Content-Type: multipart/form-data

file: <image_file>
```
Returns diagnosis, confidence, processing time, and Grad-CAM heatmap.

### Batch Prediction
```http
POST /predict-batch
Content-Type: multipart/form-data

files: <multiple_image_files>
```
Returns batch results for multiple images.

## 🎯 Usage

### Web Interface
Visit the deployed Streamlit app and upload X-ray images for instant analysis.

### API Integration
```python
import requests

# Single prediction
files = {"file": open("xray.jpg", "rb")}
response = requests.post("https://pneumonia-on4f.onrender.com/predict", files=files)
result = response.json()

# Result format
{
    "prediction": "PNEUMONIA|NORMAL",
    "confidence": 0.87,
    "processing_time_ms": 245,
    "model_accuracy": 0.96,
    "model_version": "v2",
    "heatmap_b64": "<base64_encoded_image>"
}
```

## 🏗️ Architecture

### Backend (FastAPI)
- **Framework**: FastAPI with async support
- **Model**: ResNet50 CNN from Hugging Face Hub
- **Deployment**: Render.com with automatic HTTPS
- **Caching**: Model cached in `/tmp` for fast restarts

### Frontend (Streamlit)
- **UI**: Medical-themed interface with progress bars
- **Features**: Drag-and-drop upload, real-time results
- **Integration**: Direct API calls with retry logic

### Model Hosting
- **Platform**: Hugging Face Hub (public repo)
- **Location**: `palawakampa/Pneumonia/pneumonia_resnet50_v2.h5`
- **Access**: No authentication required

## 🔧 Cold Start Behavior

Render.com free tier has cold starts. The system handles this gracefully:

1. **Startup**: App starts immediately, returns `{"status": "loading"}`
2. **Model Download**: Downloads from Hugging Face Hub (~30-60 seconds)
3. **Ready**: Returns `{"status": "ok", "model_ready": true}`
4. **UI Handling**: Streamlit waits up to 2 minutes for readiness

## 🚀 Deployment

### Render.com Configuration

#### API Service
```yaml
# render.yaml
services:
  - type: web
    name: pneumonia-api
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn fastapi_app.main:app --host 0.0.0.0 --port $PORT
    envVars:
      - key: MODEL_REPO_ID
        value: palawakampa/Pneumonia
      - key: MODEL_FILENAME
        value: pneumonia_resnet50_v2.h5
      - key: CORS_ALLOW_ORIGINS
        value: "*"
```

#### UI Service (Optional)
```yaml
  - type: web
    name: pneumonia-ui
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: streamlit run streamlit_app.py --server.port $PORT --server.address 0.0.0.0
    envVars:
      - key: FASTAPI_URL
        value: https://pneumonia-api.onrender.com/predict
```

## 🧪 Testing

Run smoke tests to verify deployment:

```bash
python scripts/smoke_test.py
```

Tests include:
- Health endpoint readiness
- Single prediction functionality
- Response format validation

## 📊 Model Performance

- **Accuracy**: 96% on validation set
- **Architecture**: ResNet50 with custom classification head
- **Input**: 224×224 RGB images, normalized [0,1]
- **Output**: Binary classification (Pneumonia/Normal)

## 🔒 Security

- CORS enabled for web interface
- Input validation for image files
- No sensitive data stored
- Public model access only

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Add tests for new functionality
4. Ensure smoke tests pass
5. Submit pull request

## 📄 License

MIT License - see LICENSE file for details.

## 📞 Support

For issues or questions:
- Create GitHub issue
- Check documentation: [Google Docs](https://docs.google.com/document/d/16kKwc9ChYLudeP3MeX18IPlnWezW-DXY9oWYZaVvy84/edit?usp=sharing)
- Contact: [WhatsApp](https://wa.me/628983776946)