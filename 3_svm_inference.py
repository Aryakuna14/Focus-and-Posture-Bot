import sys
if sys.stdout is not None and hasattr(sys.stdout, 'encoding') and sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

import cv2
import mediapipe as mp
import numpy as np
import joblib
import time
import threading
import sys
import requests
import os

DASHBOARD_URL     = "http://localhost:5001/api/update"
DASHBOARD_ENABLED = True

try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False
    print("⚠️  pyserial not installed. Running in demo mode.")

BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH      = os.path.join(BASE_DIR, "angelina_svm_model.pkl")
SCALER_PATH     = os.path.join(BASE_DIR, "angelina_scaler.pkl")
LABEL_MAP_PATH  = os.path.join(BASE_DIR, "angelina_label_map.pkl")

ESP32_PORT      = "COM11"
ESP32_BAUD      = 115200
SERIAL_ENABLED  = True

ALERT_HOLD_S    = 3.0
COOLDOWN_S      = 10.0

LANDMARK_INDICES = [0, 2, 5, 7, 8, 11, 12, 13, 14, 23, 24]
BAD_POSTURE_LABELS = {"slouching", "tech_neck"}

def extract_features(landmarks):
    features = []
    for idx in LANDMARK_INDICES:
        lm = landmarks[idx]
        features.extend([lm.x, lm.y, lm.z, lm.visibility])

    nose       = landmarks[0]
    l_shoulder = landmarks[11]
    r_shoulder = landmarks[12]
    l_ear      = landmarks[7]
    r_ear      = landmarks[8]
    l_hip      = landmarks[23]
    r_hip      = landmarks[24]

    shoulder_mid_y = (l_shoulder.y + r_shoulder.y) / 2
    hip_mid_y      = (l_hip.y + r_hip.y) / 2
    torso_length   = abs(hip_mid_y - shoulder_mid_y) + 1e-6

    nose_to_shoulder_y = (nose.y - shoulder_mid_y) / torso_length
    ear_mid_y          = (l_ear.y + r_ear.y) / 2
    ear_to_shoulder_y  = (ear_mid_y - shoulder_mid_y) / torso_length
    shoulder_width     = abs(r_shoulder.x - l_shoulder.x)
    shoulder_roll      = l_shoulder.y - r_shoulder.y

    features.extend([
        nose_to_shoulder_y,
        ear_to_shoulder_y,
        shoulder_width,
        shoulder_roll,
        torso_length,
    ])
    return np.array(features, dtype=np.float32)

class SerialManager:
    def __init__(self, port, baud, enabled):
        self.ser     = None
        self.enabled = enabled and SERIAL_AVAILABLE
        if self.enabled:
            try:
                self.ser = serial.Serial(port, baud, timeout=1, write_timeout=0.1)
                time.sleep(2.5)
            except Exception:
                self.enabled = False

    def _write_async(self, data):
        def worker():
            if self.enabled and self.ser and self.ser.is_open:
                try:
                    self.ser.write(data)
                except Exception: pass
        threading.Thread(target=worker, daemon=True).start()

    def send_trigger(self):
        if self.enabled: self._write_async(b"TRIGGER\n")
    def send_warning(self):
        if self.enabled: self._write_async(b"WARNING\n")
    def close(self):
        if self.ser and self.ser.is_open: self.ser.close()

LABEL_COLORS = {
    "good_posture":     (50, 220, 80),
    "slouching":        (30, 80, 220),
    "tech_neck":        (0, 140, 255),
    "decaying_posture": (0, 165, 255),
    "unknown":          (120, 120, 120),
}

def draw_hud(frame, label, confidence, alert_bar_frac, cooldown_remaining):
    h, w = frame.shape[:2]
    color = LABEL_COLORS.get(label, LABEL_COLORS["unknown"])
    cv2.rectangle(frame, (0, 0), (w, 50), (15, 15, 15), -1)
    cv2.putText(frame, f"POSTURE: {label.upper().replace('_', ' ')}  ({confidence*100:.1f}%)", (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
    bar_x, bar_y, bar_w, bar_h = 10, h - 40, w - 20, 18
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (40, 40, 40), -1)
    fill = int(bar_w * min(alert_bar_frac, 1.0))
    bar_color = (0, 180, 255) if alert_bar_frac < 1.0 else (0, 0, 220)
    if fill > 0: cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill, bar_y + bar_h), bar_color, -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (80, 80, 80), 1)
    if cooldown_remaining > 0: cv2.putText(frame, f"Cooldown: {cooldown_remaining:.1f}s", (w - 190, h - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1)

def main():
    for path in [MODEL_PATH, SCALER_PATH, LABEL_MAP_PATH]:
        if not os.path.exists(path):
            print(f"❌  Missing: '{path}'\n    Run 2_train_svm.py first.")
            sys.exit(1)

    clf       = joblib.load(MODEL_PATH)
    scaler    = joblib.load(SCALER_PATH)
    label_map = joblib.load(LABEL_MAP_PATH)
    print("✅ Model loaded.")

    mp_pose    = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose       = mp_pose.Pose(model_complexity=1, min_detection_confidence=0.6, min_tracking_confidence=0.6)

    serial_mgr = SerialManager(ESP32_PORT, ESP32_BAUD, SERIAL_ENABLED)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    bad_posture_since = None
    last_trigger_time = 0.0
    last_dash_update  = 0.0
    frames_missing    = 0
    baseline_features = None  # Personal calibration baseline

    def dash_sender(payload):
        try: requests.post(DASHBOARD_URL, json=payload, timeout=0.5)
        except: pass

    print("\n  🎯  Running real-time inference. Press 'C' to calibrate, 'Q' to quit.")
    print("  ⚠️  You MUST press 'C' while sitting straight to calibrate before inference works!\n")

    label = "unknown"
    confidence = 0.0
    bad_hold = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret: continue
            now = time.time()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                features = extract_features(results.pose_landmarks.landmark)
                if not np.isnan(features).any() and baseline_features is not None:
                    # Subtract personal baseline (matches training pipeline)
                    delta_features = features - baseline_features
                    X_scaled = scaler.transform(delta_features.reshape(1, -1))
                    probs = clf.predict_proba(X_scaled)[0]
                    pred_idx = np.argmax(probs)
                    confidence = probs[pred_idx]
                    label = label_map.get(pred_idx, "unknown")
                elif baseline_features is None:
                    label = "CALIBRATE: Sit straight & press 'C'"
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
                if bad_posture_since is None: bad_posture_since = now
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
                threading.Thread(target=dash_sender, args=({'label':label, 'confidence':float(confidence), 'triggered':just_triggered, 'alert_frac':float(alert_bar_frac)},), daemon=True).start()

            cv2.imshow("PROJECT ANGELINA — Live Inference (SVM)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('c') and results.pose_landmarks:
                raw_features = extract_features(results.pose_landmarks.landmark)
                if not np.isnan(raw_features).any():
                    baseline_features = raw_features
                    print("  ⚖️  BASELINE CALIBRATED — inference is now active!")
            elif key == ord('q'): break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        pose.close()
        serial_mgr.close()

if __name__ == "__main__": main()
