"""
PROJECT ANGELINA — Neural-Ergonomic Focus Bot
============================================================
SHARED UTILITIES MODULE
Purpose : Single source of truth for feature extraction,
          serial management, HUD drawing, and data windowing.
          Imported by data collection, training, inference,
          and evaluation scripts.
============================================================
"""

import numpy as np
import cv2
import time
import threading

# Serial is optional — scripts work without ESP32
try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False


# ─────────────────────────────────────────────
#  SHARED CONSTANTS
# ─────────────────────────────────────────────

# MediaPipe Pose landmark indices (upper-body focused)
# Full list: https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
LANDMARK_INDICES = [
    0,   # nose
    2,   # left_eye_inner
    5,   # right_eye_inner
    7,   # left_ear
    8,   # right_ear
    11,  # left_shoulder
    12,  # right_shoulder
    13,  # left_elbow
    14,  # right_elbow
    23,  # left_hip
    24,  # right_hip
]

BAD_POSTURE_LABELS = {"slouching", "tech_neck"}


# ─────────────────────────────────────────────
#  FEATURE EXTRACTION
# ─────────────────────────────────────────────
def extract_features(landmarks):
    """
    Flatten x, y, z, visibility for each chosen landmark into a 1D feature vector.
    Also computes normalised derived angles for richer ML features.
    Returns: np.ndarray of float32 (feature vector)
    """
    features = []
    for idx in LANDMARK_INDICES:
        lm = landmarks[idx]
        features.extend([lm.x, lm.y, lm.z, lm.visibility])

    # ── Derived geometric features ──────────────────────────────
    nose       = landmarks[0]
    l_shoulder = landmarks[11]
    r_shoulder = landmarks[12]
    l_ear      = landmarks[7]
    r_ear      = landmarks[8]
    l_hip      = landmarks[23]
    r_hip      = landmarks[24]

    # Shoulder midpoint Y vs Hip midpoint Y (vertical slouch proxy)
    shoulder_mid_y = (l_shoulder.y + r_shoulder.y) / 2
    hip_mid_y      = (l_hip.y + r_hip.y) / 2
    torso_length   = abs(hip_mid_y - shoulder_mid_y) + 1e-6  # avoid div/0

    # Nose Y relative to shoulder midpoint (head forward / tech neck proxy)
    nose_to_shoulder_y = (nose.y - shoulder_mid_y) / torso_length

    # Ear Y relative to shoulder Y (forward head posture)
    ear_mid_y          = (l_ear.y + r_ear.y) / 2
    ear_to_shoulder_y  = (ear_mid_y - shoulder_mid_y) / torso_length

    # Shoulder width (slouching reduces apparent width from camera POV)
    shoulder_width = abs(r_shoulder.x - l_shoulder.x)

    # Shoulder roll — difference in shoulder Y (asymmetric slouch)
    shoulder_roll  = l_shoulder.y - r_shoulder.y

    features.extend([
        nose_to_shoulder_y,
        ear_to_shoulder_y,
        shoulder_width,
        shoulder_roll,
        torso_length,
    ])

    return np.array(features, dtype=np.float32)


# ─────────────────────────────────────────────
#  TEMPORAL WINDOWING (for CNN / evaluation)
# ─────────────────────────────────────────────
def create_sliding_windows(X, y, window_size):
    """
    Create overlapping temporal windows for sequence models.

    Args:
        X: (N, features) array
        y: (N,) array of labels
        window_size: number of frames per window

    Returns:
        X_out: (N-window_size+1, window_size, features)
        y_out: (N-window_size+1,) — label of the LAST frame in each window
    """
    X_out, y_out = [], []
    for i in range(len(X) - window_size + 1):
        X_out.append(X[i : i + window_size])
        y_out.append(y[i + window_size - 1])
    return np.array(X_out), np.array(y_out)


# ─────────────────────────────────────────────
#  SERIAL MANAGER (ESP32 communication)
# ─────────────────────────────────────────────
class SerialManager:
    """Thread-safe, non-blocking serial manager for ESP32 haptic feedback."""

    def __init__(self, port, baud, enabled):
        self.ser     = None
        self.enabled = enabled and SERIAL_AVAILABLE
        if self.enabled:
            try:
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
#  HUD DRAWING (webcam overlay)
# ─────────────────────────────────────────────
LABEL_COLORS = {
    "good_posture":     (50, 220, 80),
    "slouching":        (30, 80, 220),
    "tech_neck":        (0, 140, 255),
    "decaying_posture": (0, 165, 255),
    "unknown":          (120, 120, 120),
}


def draw_hud(frame, label, confidence, alert_bar_frac, cooldown_remaining):
    """Draw the heads-up display overlay on the inference camera frame."""
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
