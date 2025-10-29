FROM python:3.11-slim

# 1) deps kecil + curl utk healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 2) install deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3) copy app
COPY . .

# 4) environment penting streamlit
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV PYTHONUNBUFFERED=1

EXPOSE 8501

# 5) healthcheck (Easypanel akan anggap healthy kalau endpoint ini OK)
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -fsS http://127.0.0.1:8501/_stcore/health || exit 1

# 6) start command
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
