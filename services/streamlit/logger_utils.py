# logger_utils.py
import os, json, threading, requests

LOGGER_URL = os.getenv("LOGGER_URL")
ENABLE_LOGGER = os.getenv("ENABLE_LOGGER", "true").lower() == "true"

def send_log_async(model: str, prompt: dict, response: dict, elapsed_ms: int, meta: dict):
    if not (ENABLE_LOGGER and LOGGER_URL):
        return
    payload = {
        "model": model,
        "prompt": prompt or {},
        "response": response or {},
        "elapsed_ms": int(elapsed_ms or 0),
        "meta": meta or {}
    }
    def _send():
        try:
            requests.post(LOGGER_URL, json=payload, timeout=4)
        except Exception:
            pass
    threading.Thread(target=_send, daemon=True).start()
