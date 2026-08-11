import sys
if sys.stdout is not None and hasattr(sys.stdout, 'encoding') and sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

import cv2
import mediapipe as mp
import numpy as np
import joblib
import time
import threading
import requests
import os

from utils import extract_features, SerialManager, draw_hud, BAD_POSTURE_LABELS, LABEL_COLORS
from config import (SVM_MODEL_PATH as MODEL_PATH, SCALER_PATH, LABEL_MAP_PATH,
                    ESP32_PORT, ESP32_BAUD, SERIAL_ENABLED,
                    ALERT_HOLD_S, COOLDOWN_S, DASHBOARD_URL, DASHBOARD_ENABLED)

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
        except Exception: pass

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
