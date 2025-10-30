FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 curl && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN mkdir -p /.streamlit && printf "[server]\nheadless = true\nport = 8501\naddress = \"0.0.0.0\"\n" > /.streamlit/config.toml
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --retries=5 CMD curl -fsS http://127.0.0.1:8501/_stcore/health || exit 1
CMD ["streamlit","run","app.py","--server.port=8501","--server.address=0.0.0.0","--server.headless=true"]
