"""
PROJECT ANGELINA — Neural-Ergonomic Focus Bot
============================================================
SCRIPT 3: REAL-TIME INFERENCE + ESP32 SERIAL TRIGGER
Purpose : Run live webcam inference using the trained SVM.
          When slouching or tech_neck is detected for
          ALERT_HOLD_S consecutive seconds, send 'TRIGGER\n'
          over Serial to the ESP32 to activate haptic feedback.
============================================================
USAGE:
  python 3_realtime_inference.py

  Press 'q' to quit.
============================================================
INSTALL DEPENDENCIES:
  pip install opencv-python mediapipe joblib pyserial numpy

FIND YOUR ESP32 PORT:
  Windows → Device Manager → COM Ports  (e.g. "COM3")
  Linux   → ls /dev/tty*                (e.g. "/dev/ttyUSB0")
  macOS   → ls /dev/cu.*               (e.g. "/dev/cu.usbserial-0001")
============================================================
"""

import sys
if sys.stdout is not None and hasattr(sys.stdout, 'encoding') and sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

import os
import cv2
import mediapipe as mp
import numpy as np
import joblib
import time
import threading
import collections
import sys
import requests
import tensorflow as tf

# Fixed: use localhost, not a hardcoded LAN IP that only works on one network
DASHBOARD_URL     = "http://localhost:5001/api/update"
DASHBOARD_ENABLED = True

# Serial is optional — script works without ESP32 for demo
try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False
    print("⚠️  pyserial not installed. Running in demo mode (no haptics).")

# ─────────────────────────────────────────────
#  CONFIGURATION — Edit these
# ─────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH      = os.path.join(BASE_DIR, "angelina_cnn_model.keras")
SCALER_PATH     = os.path.join(BASE_DIR, "angelina_scaler.pkl")
LABEL_MAP_PATH  = os.path.join(BASE_DIR, "angelina_label_map.pkl")

WINDOW_SIZE     = 30              # Must match 2_train_cnn.py

ESP32_PORT      = "COM11"          # ← Change to your port
ESP32_BAUD      = 115200
SERIAL_ENABLED  = True            # Set False to disable serial (demo mode)

ALERT_HOLD_S    = 3.0             # Seconds of bad posture before trigger fires
COOLDOWN_S      = 10.0            # Seconds to wait before re-triggering
SMOOTH_WINDOW   = 8               # Frames for majority-vote smoothing

# Landmark indices MUST match 1_data_collector.py
LANDMARK_INDICES = [0, 2, 5, 7, 8, 11, 12, 13, 14, 23, 24]

BAD_POSTURE_LABELS = {"slouching", "tech_neck"}   # Labels that trigger ESP32


# ─────────────────────────────────────────────
#  FEATURE EXTRACTION  (identical to collector)
# ─────────────────────────────────────────────
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


# ─────────────────────────────────────────────
#  SERIAL MANAGER
# ─────────────────────────────────────────────
class SerialManager:
    def __init__(self, port, baud, enabled):
        self.ser     = None
        self.enabled = enabled and SERIAL_AVAILABLE
        if self.enabled:
            try:
                # Default init to prevent holding ESP32 in reset via DTR
                self.ser = serial.Serial(port, baud, timeout=1, write_timeout=0.1)
                time.sleep(2.5)          # Wait for ESP32 to reboot after connect
                print(f"  🔌  ESP32 connected on {port} @ {baud} baud")
            except serial.SerialException as e:
                print(f"  ⚠️  Serial FAILED: {e}")
                print(      "      Running without haptic output.")
                self.enabled = False

    def _write_async(self, data):
        def worker():
            if self.enabled and self.ser and self.ser.is_open:
                try:
                    self.ser.write(data)
                except Exception as e:
                    print(f"  ⚠️  Serial write blocked/error: {e}")
        threading.Thread(target=worker, daemon=True).start()

    def send_trigger(self):
        if self.enabled:
            print("  📳  TRIGGER sent → ESP32")
            self._write_async(b"TRIGGER\n")
        else:
            print("  📳  [DEMO] TRIGGER would fire here (no serial)")

    def send_warning(self):
        if self.enabled:
            print("  📳  WARNING sent → ESP32")
            self._write_async(b"WARNING\n")
        else:
            print("  📳  [DEMO] WARNING would fire here (no serial)")

    def close(self):
        if self.ser and self.ser.is_open:
            self.ser.close()


# ─────────────────────────────────────────────
#  HUD DRAWING HELPERS
# ─────────────────────────────────────────────
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

    # ── Top status bar ───────────────────────────────────────
    cv2.rectangle(frame, (0, 0), (w, 50), (15, 15, 15), -1)
    cv2.putText(frame,
        f"POSTURE: {label.upper().replace('_', ' ')}  ({confidence*100:.1f}%)",
        (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

    # ── Alert hold progress bar ──────────────────────────────
    bar_x, bar_y, bar_w, bar_h = 10, h - 40, w - 20, 18
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (40, 40, 40), -1)
    fill = int(bar_w * min(alert_bar_frac, 1.0))
    bar_color = (0, 180, 255) if alert_bar_frac < 1.0 else (0, 0, 220)
    if fill > 0:
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill, bar_y + bar_h), bar_color, -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (80, 80, 80), 1)
    cv2.putText(frame, "Alert hold", (bar_x + 4, bar_y + 13),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)

    # ── Cooldown indicator ───────────────────────────────────
    if cooldown_remaining > 0:
        cv2.putText(frame, f"Cooldown: {cooldown_remaining:.1f}s",
                    (w - 190, h - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1)

    # ── Corner accent ────────────────────────────────────────
    cv2.putText(frame, "PROJECT ANGELINA",
                (w - 200, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 80), 1)


# ─────────────────────────────────────────────
#  MAIN INFERENCE LOOP
# ─────────────────────────────────────────────
def main():
    # ── Load model artifacts ─────────────────────────────────
    for path in [MODEL_PATH, SCALER_PATH, LABEL_MAP_PATH]:
        if not __import__('os').path.exists(path):
            print(f"❌  Missing: '{path}'\n    Run 2_train_svm.py first.")
            sys.exit(1)

    clf       = tf.keras.models.load_model(MODEL_PATH)
    scaler    = joblib.load(SCALER_PATH)
    label_map = joblib.load(LABEL_MAP_PATH)   # {0: 'good_posture', 1: 'slouching', …}
    print(f"  ✅  Model loaded. Classes: {list(label_map.values())}")

    # ── Init MediaPipe Pose ──────────────────────────────────
    mp_pose    = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose       = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    # ── Init Serial ──────────────────────────────────────────
    serial_mgr = SerialManager(ESP32_PORT, ESP32_BAUD, SERIAL_ENABLED)

    # ── Init webcam ──────────────────────────────────────────
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("❌  Cannot open webcam.")

    # ── State variables ──────────────────────────────────────
    feature_buffer    = collections.deque(maxlen=WINDOW_SIZE)
    bad_posture_since = None      # Timestamp when bad posture streak started
    last_trigger_time = 0.0       # Timestamp of last trigger send
    last_dash_update  = 0.0       # Throttle dashboard requests
    frames_missing    = 0         # Count consecutive dropped frames
    baseline_features = None      # Personal calibration baseline

    def dash_sender(payload):
        try:
            requests.post(DASHBOARD_URL, json=payload, timeout=0.5)
        except Exception:
            pass

    print("\n  🎯  Running real-time inference. Press 'C' to calibrate, 'Q' to quit.\n")
    print("  ⚠️  You MUST press 'C' while sitting straight to calibrate before inference works!\n")

    label = "unknown"
    confidence = 0.0

    # Wrap main loop in try/finally so webcam is ALWAYS released, even on crash
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("⚠️  Frame read failed. Retrying...")
                time.sleep(0.1)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            now        = time.time()
            frame_rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results    = pose.process(frame_rgb)

            # We don't reset label/confidence here immediately to tolerate short flickers

            if results.pose_landmarks:
                # ── Draw skeleton ─────────────────────────────────
                mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 120), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=2),
                )

                # ── Feature extraction + inference ────────────────
                features = extract_features(results.pose_landmarks.landmark)
                if not np.isnan(features).any() and baseline_features is not None:
                    # Subtract personal baseline (matches training pipeline)
                    delta_features = features - baseline_features
                    # Scale the single frame
                    X_scaled = scaler.transform(delta_features.reshape(1, -1))[0]
                    feature_buffer.append(X_scaled)

                    # Predict only when we have a full window
                    if len(feature_buffer) == WINDOW_SIZE:
                        # Reshape to (batch_size, window_size, num_features)
                        X_batch = np.array(feature_buffer).reshape(1, WINDOW_SIZE, len(X_scaled))
                        # FIX: Using model() directly instead of model.predict() is 10x faster for single batches
                        probs = clf(X_batch, training=False).numpy()[0]
                        pred_idx = np.argmax(probs)
                        confidence = probs[pred_idx]
                        label = label_map.get(pred_idx, "unknown")
                    else:
                        label = "buffering..."
                        confidence = 0.0
                elif baseline_features is None:
                    label = "CALIBRATE: Sit straight & press 'C'"
                    confidence = 0.0

                frames_missing = 0
            else:
                frames_missing += 1
                if frames_missing > 10:  # ~0.3-0.5s at 30fps
                    label = "unknown"
                    confidence = 0.0

            # ── Alert logic (trigger only after ALERT_HOLD_S) ────
            cooldown_remaining = max(0.0, COOLDOWN_S - (now - last_trigger_time))

            if label == "decaying_posture":
                if cooldown_remaining == 0.0:
                    serial_mgr.send_warning()
                    last_trigger_time = now
                bad_posture_since = None
                bad_hold = 0.0
                cv2.putText(frame, "WARNING: POSTURE DECAYING!", (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 165, 255), 2)
            elif label in BAD_POSTURE_LABELS:
                if bad_posture_since is None:
                    bad_posture_since = now
                bad_hold = now - bad_posture_since
            else:
                bad_posture_since = None
                bad_hold = 0.0

            alert_bar_frac = bad_hold / ALERT_HOLD_S

            # ── Fire trigger ──────────────────────────────────────
            # FIX: compute just_triggered BEFORE resetting bad_posture_since,
            # so the dashboard receives the correct True value (was always False before).
            just_triggered = (bad_hold >= ALERT_HOLD_S) and (cooldown_remaining == 0.0)
            if just_triggered:
                serial_mgr.send_trigger()
                last_trigger_time = now
                bad_posture_since = None    # Reset streak after trigger

            # ── Draw HUD ──────────────────────────────────────────
            draw_hud(frame, label, confidence, alert_bar_frac, cooldown_remaining)
            if DASHBOARD_ENABLED and (now - last_dash_update > 0.1):  # Limit to 10 FPS
                last_dash_update = now
                payload = {
                    'label':      label,
                    'confidence': float(confidence),
                    'triggered':  just_triggered,
                    'alert_frac': float(alert_bar_frac),
                }
                threading.Thread(target=dash_sender, args=(payload,), daemon=True).start()

            cv2.imshow("PROJECT ANGELINA — Live Inference", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('c') and results.pose_landmarks:
                raw_features = extract_features(results.pose_landmarks.landmark)
                if not np.isnan(raw_features).any():
                    baseline_features = raw_features
                    feature_buffer.clear()  # Reset buffer after recalibration
                    print("  ⚖️  BASELINE CALIBRATED — inference is now active!")
            elif key == ord('q'):
                break

    finally:
        # ── Cleanup — always runs even if an exception occurs ─────
        cap.release()
        cv2.destroyAllWindows()
        pose.close()
        serial_mgr.close()
        print("\n  👋  Session ended.\n")


if __name__ == "__main__":
    main()
