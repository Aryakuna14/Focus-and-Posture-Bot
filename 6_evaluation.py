"""
PROJECT ANGELINA — Neural-Ergonomic Focus Bot
============================================================
SCRIPT 6: IEEE EVALUATION & BASELINE COMPARISON
Purpose : Generate publication-ready evaluation metrics for
          the trained CNN posture-classification model.

Outputs (saved to evaluation_results/):
  • confusion_matrix.png    — per-class heatmap
  • roc_curves.png          — multi-class OvR ROC with AUC
  • comparison_table.png    — ANGELINA vs baselines bar chart
  • training_history.png    — accuracy & loss curves (if history exists)
  • evaluation_report.txt   — full text report (paste into paper)
  • metrics.json            — machine-readable metrics
============================================================
USAGE:
  python 6_evaluation.py
============================================================
"""
import sys
if sys.stdout is not None and hasattr(sys.stdout, 'encoding') and sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

import os, json, warnings
import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GLOG_minloglevel'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
logging.getLogger('absl').setLevel(logging.ERROR)
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(3)

from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    precision_score, recall_score, f1_score, accuracy_score,
)
import matplotlib
matplotlib.use('Agg')          # Non-interactive backend — works without display
import matplotlib.pyplot as plt
import seaborn as sns
from utils import create_sliding_windows

# ─────────────────────────────────────────────
#  PATHS  (must match 2_train_cnn.py)
# ─────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
INPUT_CSV      = os.path.join(BASE_DIR, "posture_dataset_backup.csv")
MODEL_PATH     = os.path.join(BASE_DIR, "angelina_cnn_model.keras")
SCALER_PATH    = os.path.join(BASE_DIR, "angelina_scaler.pkl")
LABEL_MAP_PATH = os.path.join(BASE_DIR, "angelina_label_map.pkl")
OUTPUT_DIR     = os.path.join(BASE_DIR, "evaluation_results")

WINDOW_SIZE = 30
LABEL_COL   = "label"
DROP_COLS   = ["subject_id"]

# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────


def compute_fpr_fnr(y_true, y_pred, classes):
    """Compute per-class False Positive Rate and False Negative Rate."""
    cm = confusion_matrix(y_true, y_pred, labels=range(len(classes)))
    fpr_dict, fnr_dict = {}, {}
    for i, cls in enumerate(classes):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp
        fpr_dict[cls] = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr_dict[cls] = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    return fpr_dict, fnr_dict


# ─────────────────────────────────────────────
#  VISION-ONLY BASELINE (rule-based thresholds)
# ─────────────────────────────────────────────
def vision_only_baseline(df_test, feature_cols, classes):
    """
    Simulates a naive, non-ML baseline using raw landmark thresholds.
    Uses the derived geometric features already in the dataset:
      - feat_nose_to_shoulder_y  (tech-neck proxy)
      - feat_ear_to_shoulder_y   (forward-head proxy)
      - feat_shoulder_width      (slouch proxy — slouching reduces apparent width)
      - feat_torso_length        (overall posture collapse)
    """
    preds = []
    nose_col  = feature_cols.index("feat_nose_to_shoulder_y") if "feat_nose_to_shoulder_y" in feature_cols else None
    ear_col   = feature_cols.index("feat_ear_to_shoulder_y") if "feat_ear_to_shoulder_y" in feature_cols else None
    width_col = feature_cols.index("feat_shoulder_width") if "feat_shoulder_width" in feature_cols else None

    for _, row in df_test.iterrows():
        vals = row[feature_cols].values.astype(float)

        # Simple threshold rules derived from feature semantics
        if nose_col is not None and vals[nose_col] > 0.35:
            preds.append("tech_neck")
        elif ear_col is not None and vals[ear_col] > 0.30:
            preds.append("tech_neck")
        elif width_col is not None and vals[width_col] < 0.15:
            preds.append("slouching")
        elif nose_col is not None and vals[nose_col] > 0.20:
            preds.append("decaying_posture")
        else:
            preds.append("good_posture")

    return preds


# ─────────────────────────────────────────────
#  ALWAYS-ON BASELINE
# ─────────────────────────────────────────────
def always_on_baseline(n_samples):
    """Flags every frame as 'slouching' (bad posture). Max recall, zero precision."""
    return ["slouching"] * n_samples


# ─────────────────────────────────────────────
#  PLOT: CONFUSION MATRIX
# ─────────────────────────────────────────────
def plot_confusion_matrix(y_true, y_pred, classes, save_path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes, ax=ax,
                linewidths=0.5, linecolor='gray')
    ax.set_title('ANGELINA CNN — Confusion Matrix', fontsize=14, fontweight='bold')
    ax.set_xlabel('Predicted Posture', fontsize=12)
    ax.set_ylabel('True Posture', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  📊  Saved: {save_path}")


# ─────────────────────────────────────────────
#  PLOT: ROC CURVES
# ─────────────────────────────────────────────
def plot_roc_curves(y_true, y_probs, classes, save_path):
    y_bin = label_binarize(y_true, classes=range(len(classes)))
    fig, ax = plt.subplots(figsize=(9, 7))
    colors = ['#00e68a', '#ff2244', '#ff6b2c', '#ffaa00']

    for i, (cls, color) in enumerate(zip(classes, colors)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_probs[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f'{cls} (AUC = {roc_auc:.3f})')

    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Random Chance')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ANGELINA CNN — Multi-Class ROC Curves (One-vs-Rest)', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  📊  Saved: {save_path}")


# ─────────────────────────────────────────────
#  PLOT: BASELINE COMPARISON BAR CHART
# ─────────────────────────────────────────────
def plot_comparison_table(results_dict, save_path):
    """Bar chart comparing Accuracy, Macro-F1, Macro-FPR, Macro-FNR across approaches."""
    approaches = list(results_dict.keys())
    metrics_names = ['Accuracy', 'Macro F1', 'Macro FPR', 'Macro FNR']

    data = []
    for approach in approaches:
        r = results_dict[approach]
        data.append([r['accuracy'], r['macro_f1'], r['macro_fpr'], r['macro_fnr']])

    data = np.array(data)
    x = np.arange(len(metrics_names))
    width = 0.25
    colors = ['#00d4ff', '#ff6b2c', '#8b5cf6']

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (approach, color) in enumerate(zip(approaches, colors)):
        bars = ax.bar(x + i * width, data[i], width, label=approach, color=color, alpha=0.85, edgecolor='white', linewidth=0.5)
        # Add value labels on top of bars
        for bar, val in zip(bars, data[i]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('ANGELINA CNN vs Baselines — Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(metrics_names, fontsize=11)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  📊  Saved: {save_path}")


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def main():
    os.system('cls' if os.name == 'nt' else 'clear')
    print(f"\n{'='*60}")
    print("  PROJECT ANGELINA — IEEE Evaluation & Baseline Comparison")
    print(f"{'='*60}\n")

    # ── Verify files exist ────────────────────────────────────
    for p in [INPUT_CSV, MODEL_PATH, SCALER_PATH, LABEL_MAP_PATH]:
        if not os.path.exists(p):
            print(f"  ❌  Missing: {p}")
            sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────
    print("  [1/6] Loading dataset and model artifacts...")
    df = pd.read_csv(INPUT_CSV)
    feature_cols = [c for c in df.columns if c not in [LABEL_COL] + DROP_COLS]
    df = df.dropna(subset=feature_cols)

    le = LabelEncoder()
    df['encoded_label'] = le.fit_transform(df[LABEL_COL])
    classes = list(le.classes_)
    n_classes = len(classes)

    # Keep a copy of raw features for vision-only baseline BEFORE scaling
    df_raw = df.copy()

    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols].values)

    # Build sliding windows per subject per label with STRICT SESSION SPLIT
    X_test_windows, y_test_windows = [], []
    df_test_list = []
    
    subjects = sorted(df['subject_id'].unique())
    split_idx = int(len(subjects) * 0.8)
    train_subjects = set(subjects[:split_idx])

    for (subj, lbl), group in df.groupby(['subject_id', LABEL_COL]):
        if subj in train_subjects:
            continue
            
        group_X = group[feature_cols].values
        group_y = group['encoded_label'].values
        
        if len(group_X) >= WINDOW_SIZE:
            xw, yw = create_sliding_windows(group_X, group_y, WINDOW_SIZE)
            X_test_windows.append(xw)
            y_test_windows.append(yw)
            
        # For the vision-only flat baseline
        raw_group = df_raw.loc[group.index]
        df_test_list.append(raw_group)

    X_test = np.concatenate(X_test_windows, axis=0) if X_test_windows else np.array([])
    y_test = np.concatenate(y_test_windows, axis=0) if y_test_windows else np.array([])
    df_test_flat = pd.concat(df_test_list)

    print(f"       Dataset    : {len(df)} frames")
    print(f"       Test split : {len(X_test)} windows")
    print(f"       Classes    : {classes}")

    # ── Load trained model ────────────────────────────────────
    print("\n  [2/6] Running ANGELINA CNN inference on test set...")
    model = tf.keras.models.load_model(MODEL_PATH)
    y_probs = model.predict(X_test, verbose=0)
    y_pred  = np.argmax(y_probs, axis=1)

    cnn_acc  = accuracy_score(y_test, y_pred)
    cnn_f1   = f1_score(y_test, y_pred, average='macro')
    cnn_prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
    cnn_rec  = recall_score(y_test, y_pred, average='macro', zero_division=0)

    fpr_dict, fnr_dict = compute_fpr_fnr(y_test, y_pred, classes)
    cnn_macro_fpr = np.mean(list(fpr_dict.values()))
    cnn_macro_fnr = np.mean(list(fnr_dict.values()))

    print(f"       Accuracy   : {cnn_acc*100:.2f}%")
    print(f"       Macro F1   : {cnn_f1:.4f}")
    print(f"       Macro FPR  : {cnn_macro_fpr:.4f}")
    print(f"       Macro FNR  : {cnn_macro_fnr:.4f}")

    # ── Vision-Only Baseline ──────────────────────────────────
    print("\n  [3/6] Running Vision-Only (threshold) baseline...")

    # Use the raw (unscaled) test portion of the flat dataframe for thresholds

    vo_preds_raw = vision_only_baseline(df_test_flat, feature_cols, classes)
    vo_preds_enc = le.transform(vo_preds_raw)
    vo_true_enc  = df_test_flat['encoded_label'].values

    vo_acc  = accuracy_score(vo_true_enc, vo_preds_enc)
    vo_f1   = f1_score(vo_true_enc, vo_preds_enc, average='macro')
    vo_fpr, vo_fnr = compute_fpr_fnr(vo_true_enc, vo_preds_enc, classes)
    vo_macro_fpr = np.mean(list(vo_fpr.values()))
    vo_macro_fnr = np.mean(list(vo_fnr.values()))

    print(f"       Accuracy   : {vo_acc*100:.2f}%")
    print(f"       Macro F1   : {vo_f1:.4f}")
    print(f"       Macro FPR  : {vo_macro_fpr:.4f}")
    print(f"       Macro FNR  : {vo_macro_fnr:.4f}")

    # ── Always-On Baseline ────────────────────────────────────
    print("\n  [4/6] Running Always-On Alert baseline...")
    ao_preds_raw = always_on_baseline(len(df_test_flat))
    ao_preds_enc = le.transform(ao_preds_raw)

    ao_acc  = accuracy_score(vo_true_enc, ao_preds_enc)
    ao_f1   = f1_score(vo_true_enc, ao_preds_enc, average='macro', zero_division=0)
    ao_fpr, ao_fnr = compute_fpr_fnr(vo_true_enc, ao_preds_enc, classes)
    ao_macro_fpr = np.mean(list(ao_fpr.values()))
    ao_macro_fnr = np.mean(list(ao_fnr.values()))

    print(f"       Accuracy   : {ao_acc*100:.2f}%")
    print(f"       Macro F1   : {ao_f1:.4f}")
    print(f"       Macro FPR  : {ao_macro_fpr:.4f}")
    print(f"       Macro FNR  : {ao_macro_fnr:.4f}")

    # ── Generate plots ────────────────────────────────────────
    print("\n  [5/6] Generating publication-ready plots...")

    plot_confusion_matrix(
        y_test, y_pred, classes,
        os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    )

    plot_roc_curves(
        y_test, y_probs, classes,
        os.path.join(OUTPUT_DIR, "roc_curves.png")
    )

    comparison = {
        'ANGELINA CNN': {
            'accuracy': cnn_acc, 'macro_f1': cnn_f1,
            'macro_fpr': cnn_macro_fpr, 'macro_fnr': cnn_macro_fnr,
        },
        'Vision-Only (Threshold)': {
            'accuracy': vo_acc, 'macro_f1': vo_f1,
            'macro_fpr': vo_macro_fpr, 'macro_fnr': vo_macro_fnr,
        },
        'Always-On Alert': {
            'accuracy': ao_acc, 'macro_f1': ao_f1,
            'macro_fpr': ao_macro_fpr, 'macro_fnr': ao_macro_fnr,
        },
    }

    plot_comparison_table(
        comparison,
        os.path.join(OUTPUT_DIR, "comparison_table.png")
    )

    # ── Generate text report ──────────────────────────────────
    print("\n  [6/6] Writing evaluation report...")

    report_lines = []
    report_lines.append("=" * 65)
    report_lines.append("  PROJECT ANGELINA — IEEE Evaluation Report")
    report_lines.append("=" * 65)
    report_lines.append(f"\n  Dataset       : {INPUT_CSV}")
    report_lines.append(f"  Total Frames  : {len(df)}")
    report_lines.append(f"  Test Windows  : {len(X_test)}")
    report_lines.append(f"  Classes       : {classes}")
    report_lines.append(f"  Window Size   : {WINDOW_SIZE}")

    report_lines.append(f"\n{'─'*65}")
    report_lines.append("  ANGELINA CNN — Per-Class Metrics")
    report_lines.append(f"{'─'*65}")
    report_lines.append(f"\n  {'Class':<22} {'FPR':>8} {'FNR':>8} {'Precision':>10} {'Recall':>8} {'F1':>8}")
    report_lines.append(f"  {'─'*22} {'─'*8} {'─'*8} {'─'*10} {'─'*8} {'─'*8}")

    cr = classification_report(y_test, y_pred, target_names=classes, output_dict=True)
    for cls in classes:
        report_lines.append(
            f"  {cls:<22} {fpr_dict[cls]:>8.4f} {fnr_dict[cls]:>8.4f} "
            f"{cr[cls]['precision']:>10.4f} {cr[cls]['recall']:>8.4f} {cr[cls]['f1-score']:>8.4f}"
        )

    report_lines.append(f"\n  {'MACRO AVG':<22} {cnn_macro_fpr:>8.4f} {cnn_macro_fnr:>8.4f} "
                         f"{cnn_prec:>10.4f} {cnn_rec:>8.4f} {cnn_f1:>8.4f}")
    report_lines.append(f"  Overall Accuracy  : {cnn_acc*100:.2f}%")

    report_lines.append(f"\n{'─'*65}")
    report_lines.append("  BASELINE COMPARISON")
    report_lines.append(f"{'─'*65}")
    report_lines.append(f"\n  {'Approach':<28} {'Accuracy':>10} {'Macro F1':>10} {'Macro FPR':>10} {'Macro FNR':>10}")
    report_lines.append(f"  {'─'*28} {'─'*10} {'─'*10} {'─'*10} {'─'*10}")

    for name, vals in comparison.items():
        report_lines.append(
            f"  {name:<28} {vals['accuracy']*100:>9.2f}% {vals['macro_f1']:>10.4f} "
            f"{vals['macro_fpr']:>10.4f} {vals['macro_fnr']:>10.4f}"
        )

    report_lines.append(f"\n{'─'*65}")
    report_lines.append("  KEY FINDINGS")
    report_lines.append(f"{'─'*65}")

    improvement_vo = ((cnn_acc - vo_acc) / vo_acc) * 100 if vo_acc > 0 else 0
    improvement_ao = ((cnn_acc - ao_acc) / ao_acc) * 100 if ao_acc > 0 else 0
    report_lines.append(f"  • ANGELINA CNN achieves {cnn_acc*100:.2f}% accuracy")
    report_lines.append(f"  • {improvement_vo:+.1f}% improvement over Vision-Only baseline ({vo_acc*100:.2f}%)")
    report_lines.append(f"  • {improvement_ao:+.1f}% improvement over Always-On baseline ({ao_acc*100:.2f}%)")
    report_lines.append(f"  • Macro FPR reduced to {cnn_macro_fpr:.4f} (vs {vo_macro_fpr:.4f} Vision-Only)")
    report_lines.append(f"  • Macro FNR reduced to {cnn_macro_fnr:.4f} (vs {vo_macro_fnr:.4f} Vision-Only)")
    report_lines.append("")

    # Full sklearn classification report
    report_lines.append(f"{'─'*65}")
    report_lines.append("  FULL CLASSIFICATION REPORT (sklearn)")
    report_lines.append(f"{'─'*65}\n")
    report_lines.append(classification_report(y_test, y_pred, target_names=classes))
    report_lines.append("=" * 65)

    report_text = "\n".join(report_lines)
    report_path = os.path.join(OUTPUT_DIR, "evaluation_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    print(f"  📄  Saved: {report_path}")

    # ── Save metrics JSON ─────────────────────────────────────
    metrics_json = {
        "angelina_cnn": {
            "accuracy": round(cnn_acc, 4),
            "macro_f1": round(cnn_f1, 4),
            "macro_precision": round(cnn_prec, 4),
            "macro_recall": round(cnn_rec, 4),
            "macro_fpr": round(cnn_macro_fpr, 4),
            "macro_fnr": round(cnn_macro_fnr, 4),
            "per_class_fpr": {k: round(v, 4) for k, v in fpr_dict.items()},
            "per_class_fnr": {k: round(v, 4) for k, v in fnr_dict.items()},
        },
        "vision_only_baseline": {
            "accuracy": round(vo_acc, 4),
            "macro_f1": round(vo_f1, 4),
            "macro_fpr": round(vo_macro_fpr, 4),
            "macro_fnr": round(vo_macro_fnr, 4),
        },
        "always_on_baseline": {
            "accuracy": round(ao_acc, 4),
            "macro_f1": round(ao_f1, 4),
            "macro_fpr": round(ao_macro_fpr, 4),
            "macro_fnr": round(ao_macro_fnr, 4),
        },
    }
    json_path = os.path.join(OUTPUT_DIR, "metrics.json")
    with open(json_path, 'w') as f:
        json.dump(metrics_json, f, indent=2)
    print(f"  📄  Saved: {json_path}")

    # ── Print summary ─────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  ✅  EVALUATION COMPLETE")
    print(f"{'='*60}")
    print(f"  All outputs saved to: {OUTPUT_DIR}/")
    print(f"{'='*60}\n")
    print(report_text)


if __name__ == "__main__":
    main()
