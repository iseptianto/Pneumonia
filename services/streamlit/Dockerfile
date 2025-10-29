# Base
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# (opsional) alat bantu
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl && rm -rf /var/lib/apt/lists/*

# Workdir di dalam container
WORKDIR /app

# 1) copy requirements dari context SEKARANG (services/streamlit)
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# 2) copy source code streamlit (lagi-lagi dari context sekarang)
COPY . .

# Port streamlit
EXPOSE 8501
ENV PORT=8501

# Jalankan app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
