import os
from typing import Optional

def get_config(key: str, default: Optional[str] = None):
    # 1) ENV dulu
    val = os.getenv(key)
    if val is not None:
        return val
    # 2) st.secrets kalau ada, tapi jangan crash kalau file tidak ada
    try:
        import streamlit as st
        # st.secrets akan melempar FileNotFoundError saat diparse jika file tak ada
        _ = st.secrets  # trigger lazy load
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    # 3) fallback
    return default

def has_secrets_file() -> bool:
    try:
        import streamlit as st
        _ = st.secrets
        return True
    except Exception:
        return False