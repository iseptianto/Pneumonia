# services/streamlit/Dockerfile
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# (opsional) tool utk healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# copy requirements dulu biar layer cache
COPY services/streamlit/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# copy source code streamlit
COPY services/streamlit/ ./

# ENV penting untuk front-end
ENV FASTAPI_URL="https://pneumonia-on4f.onrender.com/predict" \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

EXPOSE 8501

# Healthcheck (Streamlit punya endpoint internal)
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -fsS http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
