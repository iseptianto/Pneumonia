FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Default ENV variables for deployment (can be overridden)
ENV STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    DOCS_URL="https://docs.google.com/document/d/16kKwc9ChYLudeP3MeX18IPlnWezW-DXY9oWYZaVvy84/edit?usp=sharing" \
    FASTAPI_URL="https://pneumonia-on4f.onrender.com/predict" \
    DEBUG="false"

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
