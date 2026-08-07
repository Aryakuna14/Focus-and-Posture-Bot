"""Local dashboard for Project Angelina posture sessions."""

import secrets
import sys
import threading
from collections import deque
from datetime import datetime

from flask import Flask, render_template, request
from flask_socketio import SocketIO

from config import DASHBOARD_HOST, DASHBOARD_PORT

if sys.stdout is not None and getattr(sys.stdout, "encoding", None):
    if sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")

app = Flask(__name__)
app.config["SECRET_KEY"] = secrets.token_hex(32)
socketio = SocketIO(
    app,
    cors_allowed_origins=[f"http://{DASHBOARD_HOST}:{DASHBOARD_PORT}"],
    async_mode="threading",
)

session_data = {
    "current_label": "unknown",
    "current_confidence": 0.0,
    "session_start": datetime.now().isoformat(),
    "total_triggers": 0,
    "posture_counts": {
        "good_posture": 0,
        "slouching": 0,
        "tech_neck": 0,
        "decaying_posture": 0,
        "unknown": 0,
    },
    "alert_log": [],
    "history": deque(maxlen=120),
}

lock = threading.Lock()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/update", methods=["POST"])
def update():
    data = request.get_json(silent=True) or {}

    with lock:
        label = str(data.get("label", "unknown"))
        confidence = float(data.get("confidence", 0.0) or 0.0)
        triggered = bool(data.get("triggered", False))

        session_data["current_label"] = label
        session_data["current_confidence"] = confidence

        if label in session_data["posture_counts"]:
            session_data["posture_counts"][label] += 1

        if triggered:
            session_data["total_triggers"] += 1
            session_data["alert_log"].append({
                "time": datetime.now().strftime("%H:%M:%S"),
                "label": label,
            })
            session_data["alert_log"] = session_data["alert_log"][-50:]

        session_data["history"].append({
            "time": datetime.now().strftime("%H:%M:%S"),
            "label": label,
            "confidence": round(confidence * 100, 1),
        })

        emit_data = {
            "label": label,
            "confidence": round(confidence * 100, 1),
            "triggered": triggered,
            "counts": dict(session_data["posture_counts"]),
            "triggers": session_data["total_triggers"],
            "alert_log": session_data["alert_log"][-10:],
            "history": list(session_data["history"])[-60:],
            "alert_frac": float(data.get("alert_frac", 0.0) or 0.0),
        }

    socketio.emit("posture_update", emit_data)
    return {"status": "ok"}


@app.route("/api/reset", methods=["POST"])
def reset():
    with lock:
        session_data["total_triggers"] = 0
        session_data["posture_counts"] = {
            "good_posture": 0,
            "slouching": 0,
            "tech_neck": 0,
            "decaying_posture": 0,
            "unknown": 0,
        }
        session_data["alert_log"] = []
        session_data["history"] = deque(maxlen=120)
        session_data["session_start"] = datetime.now().isoformat()
        new_start = session_data["session_start"]

    socketio.emit("session_reset", {"session_start": new_start})
    return {"status": "reset"}


@app.route("/api/state")
def state():
    with lock:
        return {
            "label": session_data["current_label"],
            "confidence": round(session_data["current_confidence"] * 100, 1),
            "counts": dict(session_data["posture_counts"]),
            "triggers": session_data["total_triggers"],
            "alert_log": session_data["alert_log"][-10:],
            "history": list(session_data["history"])[-60:],
            "session_start": session_data["session_start"],
        }


if __name__ == "__main__":
    print("\nProject Angelina dashboard")
    print(f"Open http://{DASHBOARD_HOST}:{DASHBOARD_PORT}")
    print("Set ANGELINA_DASHBOARD_HOST=0.0.0.0 only when you intentionally want LAN access.\n")
    socketio.run(app, host=DASHBOARD_HOST, port=DASHBOARD_PORT, debug=False, allow_unsafe_werkzeug=True)
