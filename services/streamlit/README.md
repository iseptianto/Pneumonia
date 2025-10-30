# Streamlit Pneumonia Diagnosis App

## Configuration

App membaca config dari ENV (disarankan di Easypanel → Environment):

```
FASTAPI_URL=https://pneumonia-on4f.onrender.com/predict
FASTAPI_URL_BATCH=https://pneumonia-on4f.onrender.com/predict-batch
DOCS_URL=https://docs.google.com/document/d/16kKwc9ChYLudeP3MeX18IPlnWezW-DXY9oWYZaVvy84/edit?usp=sharing
CONTACT_URL=mailto:hello@palawakampa.com?subject=Pneumonia%20App
```

secrets.toml bersifat opsional. Jika ingin, mount ke `/app/.streamlit/secrets.toml`.