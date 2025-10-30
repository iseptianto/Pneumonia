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
        # Cek apakah secrets file ada dengan cara yang lebih aman
        if hasattr(st, 'secrets') and st.secrets is not None:
            # Coba akses key secara spesifik tanpa trigger parsing seluruh file
            try:
                return st.secrets[key]
            except (KeyError, FileNotFoundError):
                pass
    except Exception:
        pass
    # 3) fallback
    return default

def has_secrets_file() -> bool:
    try:
        import streamlit as st
        # Cek dengan cara yang lebih aman
        if hasattr(st, 'secrets') and st.secrets is not None:
            # Coba akses dummy key untuk trigger parsing
            try:
                _ = st.secrets['_dummy_check_']
            except KeyError:
                # KeyError berarti file ada tapi key tidak ada - ini normal
                return True
            except FileNotFoundError:
                return False
            # Jika berhasil akses dummy key, berarti ada file
            return True
        return False
    except Exception:
        return False