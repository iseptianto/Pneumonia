# FastAPI Pneumonia Inference Service

## Overview
This service provides pneumonia detection from chest X-ray images using a ResNet50 deep learning model deployed with FastAPI.

## Features
- REST API for pneumonia prediction
- Grad-CAM explainability
- MLflow experiment tracking (optional)
- Prometheus metrics
- Docker containerization

## Quick Start

### Local Development with Docker Compose
```bash
# Start all services including MLflow
docker-compose up -d

# Access MLflow UI at http://localhost:5001
# API available at http://localhost:8000
```

### Manual MLflow Setup
If running FastAPI without docker-compose, enable MLflow locally:
```bash
# Start MLflow server
mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db

# Set environment variables
export MLFLOW_TRACKING_URI=http://localhost:5000
export MLFLOW_EXPERIMENT_NAME=pneumonia-inference
```

### Render Deployment
For Render.com deployment, either:
- Set `MLFLOW_ENABLED=false` (recommended for Free tier)
- Or point `MLFLOW_TRACKING_URI` to a reachable external MLflow server

## Environment Variables
- `MLFLOW_ENABLED`: Enable/disable MLflow logging (default: "true")
- `MLFLOW_TRACKING_URI`: MLflow server URL (optional)
- `MLFLOW_EXPERIMENT_NAME`: Experiment name (default: "pneumonia-inference")
- `MODEL_PATH`: Path to model file
- `GDRIVE_FILE_ID`: Google Drive file ID for model download
- `MODEL_VERSION`: Model version for auto-updates

## API Endpoints
- `GET /health`: Health check
- `POST /predict`: Single image prediction
- `POST /predict-batch`: Multiple image predictions
- `GET /metrics`: Prometheus metrics