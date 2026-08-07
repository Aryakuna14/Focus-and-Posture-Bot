"""Runtime configuration for Project Angelina.

Values are intentionally read from environment variables so the code can run on
Windows, macOS, Linux, and demo machines without editing source files.
"""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


DASHBOARD_HOST = os.getenv("ANGELINA_DASHBOARD_HOST", "127.0.0.1")
DASHBOARD_PORT = env_int("ANGELINA_DASHBOARD_PORT", 5001)
DASHBOARD_URL = os.getenv(
    "ANGELINA_DASHBOARD_URL",
    f"http://{DASHBOARD_HOST}:{DASHBOARD_PORT}/api/update",
)
DASHBOARD_ENABLED = env_bool("ANGELINA_DASHBOARD_ENABLED", True)

ESP32_PORT = os.getenv("ANGELINA_ESP32_PORT", "")
ESP32_BAUD = env_int("ANGELINA_ESP32_BAUD", 115200)
SERIAL_ENABLED = env_bool("ANGELINA_SERIAL_ENABLED", bool(ESP32_PORT))

ALERT_HOLD_S = env_float("ANGELINA_ALERT_HOLD_S", 3.0)
COOLDOWN_S = env_float("ANGELINA_COOLDOWN_S", 10.0)
WINDOW_SIZE = env_int("ANGELINA_WINDOW_SIZE", 30)

CNN_MODEL_PATH = os.getenv(
    "ANGELINA_CNN_MODEL_PATH",
    os.path.join(BASE_DIR, "angelina_cnn_model.keras"),
)
SVM_MODEL_PATH = os.getenv(
    "ANGELINA_SVM_MODEL_PATH",
    os.path.join(BASE_DIR, "angelina_svm_model.pkl"),
)
SCALER_PATH = os.getenv(
    "ANGELINA_SCALER_PATH",
    os.path.join(BASE_DIR, "angelina_scaler.pkl"),
)
LABEL_MAP_PATH = os.getenv(
    "ANGELINA_LABEL_MAP_PATH",
    os.path.join(BASE_DIR, "angelina_label_map.pkl"),
)

