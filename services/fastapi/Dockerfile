FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt

# Fix blinker conflict: upgrade pip & force install blinker before requirements
RUN python -m pip install --upgrade pip setuptools wheel \
 && pip install --no-cache-dir --ignore-installed blinker==1.7.0 \
 && pip install --no-cache-dir -r /app/requirements.txt

# Create model cache folder
RUN mkdir -p /app/models

COPY app /app/app

ENV PYTHONUNBUFFERED=1
ENV PIP_BREAK_SYSTEM_PACKAGES=1

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
