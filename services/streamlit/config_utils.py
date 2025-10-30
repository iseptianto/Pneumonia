import os
from typing import Optional, List

def _from_env(key: str) -> Optional[str]:
    val = os.getenv(key)
    return None if val is None else str(val)

def _from_secrets(key: str) -> Optional[str]:
    try:
        import streamlit as st  # type: ignore
        obj = st.secrets  # may raise internally; that's why we're in try
        if isinstance(obj, dict) and key in obj:
            return str(obj[key])
        return None
    except Exception:
        return None

def _pick_value(key: str, default: Optional[str]) -> Optional[str]:
    return _from_env(key) or _from_secrets(key) or default

def get_config(key: str, default: Optional[str] = None) -> Optional[str]:
    return _pick_value(key, default)

def get_bool(key: str, default: bool = False) -> bool:
    """Get boolean config value. Accepts: true/false, 1/0, yes/no, y/n, on/off (case-insensitive)"""
    val = get_config(key)
    if val is None:
        return default

    val_lower = val.lower().strip()
    return val_lower in ('true', '1', 'yes', 'y', 'on')

def get_int(key: str, default: int = 0) -> int:
    """Get integer config value"""
    val = get_config(key)
    if val is None:
        return default

    try:
        return int(val.strip())
    except (ValueError, AttributeError):
        return default

def get_list(key: str, default: Optional[List[str]] = None, sep: str = ",") -> List[str]:
    """Get list config value, split by separator, trim whitespace, drop empty items"""
    val = get_config(key)
    if val is None:
        return default or []

    # Split, strip, filter empty
    items = [item.strip() for item in val.split(sep)]
    return [item for item in items if item]

def has_secrets_file() -> bool:
    """Check if secrets.toml file exists without crashing"""
    try:
        import streamlit as st
        if hasattr(st, 'secrets') and st.secrets is not None:
            try:
                _ = st.secrets['_dummy_check_']
            except KeyError:
                return True  # File exists, key doesn't
            except FileNotFoundError:
                return False  # File doesn't exist
            return True  # File exists and accessible
        return False
    except Exception:
        return False

# Tiny test examples (optional, can be removed in production)
if __name__ == "__main__":
    # Test get_bool edge cases
    test_cases_bool = ["TRUE", "0", "off", "maybe", "yes", "no", "1", "false"]
    for case in test_cases_bool:
        os.environ["TEST_BOOL"] = case
        result = get_bool("TEST_BOOL", False)
        print(f"get_bool('{case}') -> {result}")

    # Test get_int edge cases
    test_cases_int = ["  8501  ", "NaN", "42", ""]
    for case in test_cases_int:
        os.environ["TEST_INT"] = case
        result = get_int("TEST_INT", 0)
        print(f"get_int('{case}') -> {result}")

    # Test get_list edge cases
    test_cases_list = ["a, b, , c", "x;y;z", "single", ""]
    for case in test_cases_list:
        os.environ["TEST_LIST"] = case
        result_comma = get_list("TEST_LIST", [], ",")
        result_semi = get_list("TEST_LIST", [], ";")
        print(f"get_list('{case}', sep=',') -> {result_comma}")
        print(f"get_list('{case}', sep=';') -> {result_semi}")