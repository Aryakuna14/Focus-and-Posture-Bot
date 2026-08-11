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
import numpy as np
import pandas as pd
import time

from utils import LANDMARK_INDICES, extract_features

# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────
BASE_DIR         = os.path.dirname(os.path.abspath(__file__))
OUTPUT_CSV       = os.path.join(BASE_DIR, "posture_dataset.csv")
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
                    delta_features = (features - baseline_features).tolist()
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
                    baseline_features = extract_features(results.pose_landmarks.landmark)
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
