"""Real-time SVM posture inference with optional ESP32 feedback.

Security note: this script loads scikit-learn artifacts with joblib. Only load
artifacts you trained yourself or explicitly trust, because pickle/joblib files
can execute code during deserialization.
"""

import os
import sys
import time
import threading

import cv2
import joblib
import mediapipe as mp
import numpy as np
import requests

from config import (
    ALERT_HOLD_S,
    COOLDOWN_S,
    DASHBOARD_ENABLED,
    DASHBOARD_URL,
    ESP32_BAUD,
    ESP32_PORT,
    LABEL_MAP_PATH,
    SCALER_PATH,
    SERIAL_ENABLED,
    SVM_MODEL_PATH,
)

if sys.stdout is not None and getattr(sys.stdout, "encoding", None):
    if sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")

try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False
    print("pyserial is not installed. Running in demo mode with no ESP32 output.")

LANDMARK_INDICES = [0, 2, 5, 7, 8, 11, 12, 13, 14, 23, 24]
BAD_POSTURE_LABELS = {"slouching", "tech_neck"}


def extract_features(landmarks):
    features = []
    for idx in LANDMARK_INDICES:
        lm = landmarks[idx]
        features.extend([lm.x, lm.y, lm.z, lm.visibility])

    nose = landmarks[0]
    left_shoulder = landmarks[11]
    right_shoulder = landmarks[12]
    left_ear = landmarks[7]
    right_ear = landmarks[8]
    left_hip = landmarks[23]
    right_hip = landmarks[24]

    shoulder_mid_y = (left_shoulder.y + right_shoulder.y) / 2
    hip_mid_y = (left_hip.y + right_hip.y) / 2
    torso_length = abs(hip_mid_y - shoulder_mid_y) + 1e-6
    ear_mid_y = (left_ear.y + right_ear.y) / 2

    features.extend([
        (nose.y - shoulder_mid_y) / torso_length,
        (ear_mid_y - shoulder_mid_y) / torso_length,
        abs(right_shoulder.x - left_shoulder.x),
        left_shoulder.y - right_shoulder.y,
        torso_length,
    ])
    return np.array(features, dtype=np.float32)


class SerialManager:
    def __init__(self, port, baud, enabled):
        self.ser = None
        self.enabled = enabled and SERIAL_AVAILABLE and bool(port)
        if self.enabled:
            try:
                self.ser = serial.Serial(port, baud, timeout=1, write_timeout=0.1)
                time.sleep(2.5)
                print(f"ESP32 connected on {port} at {baud} baud.")
            except Exception as exc:
                print(f"Serial unavailable ({exc}). Continuing without haptics.")
                self.enabled = False
        else:
            print("ESP32 output disabled. Set ANGELINA_ESP32_PORT to enable haptics.")

    def _write_async(self, data):
        def worker():
            if self.enabled and self.ser and self.ser.is_open:
                try:
                    self.ser.write(data)
                except Exception as exc:
                    print(f"Serial write failed: {exc}")

        threading.Thread(target=worker, daemon=True).start()

    def send_trigger(self):
        if self.enabled:
            self._write_async(b"TRIGGER\n")
        else:
            print("[demo] Posture trigger would fire here.")

    def send_warning(self):
        if self.enabled:
            self._write_async(b"WARNING\n")
        else:
            print("[demo] Posture warning would fire here.")

    def close(self):
        if self.ser and self.ser.is_open:
            self.ser.close()


LABEL_COLORS = {
    "good_posture": (50, 220, 80),
    "slouching": (30, 80, 220),
    "tech_neck": (0, 140, 255),
    "decaying_posture": (0, 165, 255),
    "unknown": (120, 120, 120),
}


def draw_hud(frame, label, confidence, alert_bar_frac, cooldown_remaining):
    h, w = frame.shape[:2]
    color = LABEL_COLORS.get(label, LABEL_COLORS["unknown"])
    cv2.rectangle(frame, (0, 0), (w, 50), (15, 15, 15), -1)
    cv2.putText(frame, f"POSTURE: {label.upper().replace('_', ' ')} ({confidence * 100:.1f}%)", (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

    bar_x, bar_y, bar_w, bar_h = 10, h - 40, w - 20, 18
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (40, 40, 40), -1)
    fill = int(bar_w * min(alert_bar_frac, 1.0))
    if fill > 0:
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill, bar_y + bar_h), (0, 180, 255), -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (80, 80, 80), 1)

    if cooldown_remaining > 0:
        cv2.putText(frame, f"Cooldown: {cooldown_remaining:.1f}s", (w - 190, h - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1)


def require_artifacts(paths):
    missing = [path for path in paths if not os.path.exists(path)]
    if missing:
        print("Missing model artifacts:")
        for path in missing:
            print(f"  - {path}")
        print("Train local artifacts with 2_train_svm.py before running inference.")
        sys.exit(1)


def main():
    require_artifacts([SVM_MODEL_PATH, SCALER_PATH, LABEL_MAP_PATH])
    print("Loading local model artifacts. Only proceed if you trust or trained these files.")
    clf = joblib.load(SVM_MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    label_map = joblib.load(LABEL_MAP_PATH)

    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose(model_complexity=1, min_detection_confidence=0.6, min_tracking_confidence=0.6)
    serial_mgr = SerialManager(ESP32_PORT, ESP32_BAUD, SERIAL_ENABLED)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam. Check camera permissions and camera index.")

    bad_posture_since = None
    last_trigger_time = 0.0
    last_dash_update = 0.0
    frames_missing = 0
    baseline_features = None
    label = "unknown"
    confidence = 0.0
    bad_hold = 0.0

    def dash_sender(payload):
        try:
            requests.post(DASHBOARD_URL, json=payload, timeout=0.5)
        except Exception:
            pass

    print("Running SVM inference. Press C to calibrate while sitting tall. Press Q to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue

            now = time.time()
            results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                features = extract_features(results.pose_landmarks.landmark)
                if not np.isnan(features).any() and baseline_features is not None:
                    delta_features = features - baseline_features
                    x_scaled = scaler.transform(delta_features.reshape(1, -1))
                    probs = clf.predict_proba(x_scaled)[0]
                    pred_idx = int(np.argmax(probs))
                    confidence = float(probs[pred_idx])
                    label = label_map.get(pred_idx, "unknown")
                elif baseline_features is None:
                    label = "CALIBRATE: Sit straight & press C"
                    confidence = 0.0
                frames_missing = 0
            else:
                frames_missing += 1
                if frames_missing > 10:
                    label = "unknown"
                    confidence = 0.0

            cooldown_remaining = max(0.0, COOLDOWN_S - (now - last_trigger_time))
            if label == "decaying_posture":
                if cooldown_remaining == 0.0:
                    serial_mgr.send_warning()
                    last_trigger_time = now
                bad_posture_since = None
                bad_hold = 0.0
            elif label in BAD_POSTURE_LABELS:
                if bad_posture_since is None:
                    bad_posture_since = now
                bad_hold = now - bad_posture_since
            else:
                bad_posture_since = None
                bad_hold = 0.0

            alert_bar_frac = bad_hold / ALERT_HOLD_S
            just_triggered = (bad_hold >= ALERT_HOLD_S) and (cooldown_remaining == 0.0)
            if just_triggered:
                serial_mgr.send_trigger()
                last_trigger_time = now
                bad_posture_since = None

            draw_hud(frame, label, confidence, alert_bar_frac, cooldown_remaining)
            if DASHBOARD_ENABLED and (now - last_dash_update > 0.1):
                last_dash_update = now
                payload = {"label": label, "confidence": confidence, "triggered": just_triggered, "alert_frac": float(alert_bar_frac)}
                threading.Thread(target=dash_sender, args=(payload,), daemon=True).start()

            cv2.imshow("Project Angelina - SVM Inference", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("c") and results.pose_landmarks:
                raw_features = extract_features(results.pose_landmarks.landmark)
                if not np.isnan(raw_features).any():
                    baseline_features = raw_features
                    print("Baseline calibrated. Inference is active.")
            elif key == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        pose.close()
        serial_mgr.close()


if __name__ == "__main__":
    main()
