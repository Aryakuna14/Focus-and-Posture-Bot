"""
PROJECT ANGELINA — Neural-Ergonomic Focus Bot
============================================================
SCRIPT 1: DATA COLLECTOR
Purpose : Record labeled skeletal landmarks from 100+ subjects
          into a CSV file for SVM training.
Sensors : Laptop Webcam (via OpenCV + MediaPipe Pose)
Output  : posture_dataset.csv
============================================================
USAGE:
  python 1_data_collector.py

  Press 'g' → label next N frames as 'good_posture'
  Press 's' → label next N frames as 'slouching'
  Press 't' → label next N frames as 'tech_neck'
  Press 'q' → quit and save CSV
============================================================
INSTALL DEPENDENCIES:
  pip install opencv-python mediapipe pandas
"""
import sys
if sys.stdout is not None and hasattr(sys.stdout, 'encoding') and sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
import os
import logging

# Suppress annoying TensorFlow / MediaPipe C++ logs BEFORE importing them
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GLOG_minloglevel'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
logging.getLogger('absl').setLevel(logging.ERROR)

import cv2
import tkinter as tk
from tkinter import messagebox
import mediapipe as mp
import pandas as pd
import time
logging.getLogger('absl').setLevel(logging.ERROR)

# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────
OUTPUT_CSV       = r"C:\Users\aryas\OneDrive\Desktop\final\posture_dataset.csv"
FRAMES_PER_LABEL = 60              # How many frames to capture per button press
CAPTURE_DELAY_S  = 0.05            # Seconds between frame captures (~20 FPS effective)

def ask_yes_no(title, message):
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    result = messagebox.askyesno(title, message)
    root.destroy()
    return result

def show_info(title, message):
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    messagebox.showinfo(title, message)
    root.destroy()

def get_next_subject_id():
    """Auto-detect the next subject ID from the CSV file."""
    if not os.path.exists(OUTPUT_CSV):
        return "S001"
    try:
        df = pd.read_csv(OUTPUT_CSV, usecols=["subject_id"])
        if df.empty:
            return "S001"
        # Extract numeric parts from all subject IDs (e.g. "S021" -> 21)
        nums = df["subject_id"].str.extract(r"S(\d+)", expand=False).dropna().astype(int)
        if nums.empty:
            return "S001"
        return f"S{nums.max() + 1:03d}"
    except Exception:
        return "S001"

SUBJECT_ID = get_next_subject_id()

# MediaPipe Pose landmark indices we care about (upper-body focused)
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

LABELS = {
    ord('g'): 'good_posture',
    ord('s'): 'slouching',
    ord('t'): 'tech_neck',
    ord('d'): 'decaying_posture',
}

# Calibration state
baseline_features = None

# ─────────────────────────────────────────────
#  FEATURE EXTRACTION
# ─────────────────────────────────────────────
def extract_features(landmarks):
    """
    Flatten x, y, z, visibility for each chosen landmark into a 1D feature vector.
    Also computes normalised derived angles for richer ML features.
    Returns: list of floats (feature vector)
    """
    features = []
    for idx in LANDMARK_INDICES:
        lm = landmarks[idx]
        features.extend([lm.x, lm.y, lm.z, lm.visibility])

    # ── Derived geometric features ──────────────────────────────
    # (These give the SVM more discriminative power)

    nose       = landmarks[0]
    l_shoulder = landmarks[11]
    r_shoulder = landmarks[12]
    l_ear      = landmarks[7]
    r_ear      = landmarks[8]
    l_hip      = landmarks[23]
    r_hip      = landmarks[24]

    # 1. Shoulder midpoint Y vs Hip midpoint Y  (vertical slouch proxy)
    shoulder_mid_y = (l_shoulder.y + r_shoulder.y) / 2
    hip_mid_y      = (l_hip.y + r_hip.y) / 2
    torso_length   = abs(hip_mid_y - shoulder_mid_y) + 1e-6  # avoid div/0

    # 2. Nose Y relative to shoulder midpoint (head forward/tech neck proxy)
    nose_to_shoulder_y = (nose.y - shoulder_mid_y) / torso_length

    # 3. Ear Y relative to shoulder Y (forward head posture)
    ear_mid_y          = (l_ear.y + r_ear.y) / 2
    ear_to_shoulder_y  = (ear_mid_y - shoulder_mid_y) / torso_length

    # 4. Shoulder width (slouching reduces apparent width from camera POV)
    shoulder_width = abs(r_shoulder.x - l_shoulder.x)

    # 5. Shoulder roll — difference in shoulder Y (asymmetric slouch)
    shoulder_roll  = l_shoulder.y - r_shoulder.y

    features.extend([
        nose_to_shoulder_y,
        ear_to_shoulder_y,
        shoulder_width,
        shoulder_roll,
        torso_length,
    ])

    return features


def build_header():
    """Build CSV column headers matching extract_features() output."""
    cols = []
    for idx in LANDMARK_INDICES:
        cols += [f"lm{idx}_x", f"lm{idx}_y", f"lm{idx}_z", f"lm{idx}_vis"]
    cols += [
        "feat_nose_to_shoulder_y",
        "feat_ear_to_shoulder_y",
        "feat_shoulder_width",
        "feat_shoulder_roll",
        "feat_torso_length",
    ]
    return cols


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def main():
    mp_pose    = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose       = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,          # 0=lite, 1=full, 2=heavy
        smooth_landmarks=True,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("❌  Cannot open webcam. Check camera index.")

    try:
        while True:
            # Re-fetch the dynamic subject ID and total rows at the start of each session
            current_subject = get_next_subject_id()
            try:
                total_rows = len(pd.read_csv(OUTPUT_CSV)) if os.path.exists(OUTPUT_CSV) else 0
            except Exception:
                total_rows = 0

            # Console UI (Logged to terminal if open)
            os.system('cls' if os.name == 'nt' else 'clear')
            print("\n" + "="*55)
            print("  PROJECT ANGELINA — Continuous Data Collector")
            print("="*55)
            print(f"  Current Entry Number : {current_subject}")
            print(f"  Total Rows Collected : {total_rows}")
            print("="*55)

            prompt_msg = f"Current Entry Number: {current_subject}\nTotal Rows Collected: {total_rows}\n\nDo you want to collect new posture data for {current_subject}?"
            ans = ask_yes_no("Project Angelina - Data Collector", prompt_msg)
            if not ans:
                break

            print("\n  [C] SET BASELINE (Sit straight and press C first!)")
            print("  [G] Record GOOD POSTURE")
            print("  [S] Record SLOUCHING")
            print("  [T] Record TECH NECK")
            print("  [D] Record DECAYING POSTURE (Start good, slowly slump!)")
            print("  [Q] Save CSV and advance to NEXT subject\n")

            all_rows = []
            header = build_header() + ["label", "subject_id"]
            active_label = None
            frames_left = 0
            global baseline_features
            baseline_features = None

            # Inner webcam loop
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("⚠️  Frame read failed, skipping...")
                    time.sleep(0.1)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    continue

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results   = pose.process(frame_rgb)

                # Draw skeleton overlay
                annotated = frame.copy()
                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        annotated,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 255, 120), thickness=2, circle_radius=3),
                        mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=2),
                    )

                # HUD overlay - Clean Neon Look
                neon_blue = (255, 220, 0)  # BGR for bright neon blue
                recording_color = (0, 0, 255) # Red for recording
                baseline_color = (0, 255, 0) # Green
                
                if baseline_features is None:
                    hud_color = (0, 165, 255) # Orange warning
                    status = "ACTION REQUIRED: Sit straight and press 'C' to Calibrate!"
                else:
                    hud_color = recording_color if active_label else neon_blue
                    status    = f"RECORDING [{active_label}] - {frames_left} frames left" if active_label else "IDLE: Press G / S / T / D to record"
                
                # Function to draw drop-shadow text for clean readability without blocky backgrounds
                def draw_neon_text(img, text, pos, font_scale, color, thickness=2):
                    # Subtle shadow
                    cv2.putText(img, text, (pos[0]+2, pos[1]+2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 1)
                    # Main text
                    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

                draw_neon_text(annotated, status, (15, 35), 0.7, hud_color, 2)
                draw_neon_text(annotated, f"Rows: {len(all_rows)}", (15, frame.shape[0] - 20), 0.6, neon_blue, 1)
                draw_neon_text(annotated, f"Subject: {current_subject}", (frame.shape[1] - 180, frame.shape[0] - 20), 0.6, neon_blue, 1)

                if baseline_features is not None:
                    draw_neon_text(annotated, "[CALIBRATED]", (frame.shape[1] - 180, 35), 0.6, baseline_color, 2)

                # Capture logic
                if active_label and frames_left > 0 and results.pose_landmarks:
                    features = extract_features(results.pose_landmarks.landmark)
                    delta_features = list(np.array(features) - baseline_features)
                    all_rows.append(delta_features + [active_label, current_subject])
                    frames_left -= 1

                    if frames_left == 0:
                        print(f"  ✅  Captured {FRAMES_PER_LABEL} frames → '{active_label}'  (Total session rows: {len(all_rows)})")
                        active_label = None

                # Show frame
                cv2.imshow("PROJECT ANGELINA — Data Collector", annotated)

                # Key handling
                key = cv2.waitKey(1) & 0xFF
                if key == ord('c') and results.pose_landmarks:
                    import numpy as np
                    baseline_features = np.array(extract_features(results.pose_landmarks.landmark))
                    print("  ⚖️  BASELINE CALIBRATED for this session.")
                elif key in LABELS:
                    if baseline_features is None:
                        show_info("Calibration Required", "You must sit straight and press 'C' to set the baseline before recording!")
                    else:
                        active_label = LABELS[key]
                        frames_left  = FRAMES_PER_LABEL
                        if active_label == 'decaying_posture':
                            show_info("Trajectory Capture", "Get into a PERFECT posture.\n\nOnce you click OK, you will have 3 seconds to SLOWLY decay into a slouch.")
                        print(f"  🎬  Starting capture → '{active_label}' ({FRAMES_PER_LABEL} frames)…")
                elif key == ord('q'):
                    break
            
            # End of inner loop: save CSV
            if all_rows:
                df = pd.DataFrame(all_rows, columns=header)
                write_header = not os.path.exists(OUTPUT_CSV)
                df.to_csv(OUTPUT_CSV, mode='a', header=write_header, index=False)
                print(f"\n  💾  Saved {len(all_rows)} new rows to '{OUTPUT_CSV}'")
            else:
                print("\n  ⚠️  No data was recorded this session.")

            # Temporarily close OpenCV window while asking Y/N in terminal
            cv2.destroyAllWindows()

    finally:
        print("\n  ============================================================")
        print("     Session Ended. Goodbye!")
        print("  ============================================================\n")
        cap.release()
        cv2.destroyAllWindows()
        pose.close()
        show_info("Project Angelina", "Data collection session ended successfully.")

if __name__ == "__main__":
    main()
