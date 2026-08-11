import sys
if sys.stdout is not None and hasattr(sys.stdout, 'encoding') and sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

import os, json, warnings
import logging
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('absl').setLevel(logging.ERROR)
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils import create_sliding_windows
from config import (CNN_MODEL_PATH, SVM_MODEL_PATH, SCALER_PATH, LABEL_MAP_PATH, WINDOW_SIZE, BASE_DIR)

INPUT_CSV  = os.path.join(BASE_DIR, "posture_dataset.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "evaluation_results")

LABEL_COL   = "label"
DROP_COLS   = ["subject_id"]

def plot_comparison_bar(metrics, save_path):
    models = list(metrics.keys())
    metric_names = ['Accuracy', 'Macro F1', 'Macro Precision', 'Macro Recall']
    
    data = []
    for model in models:
        r = metrics[model]
        data.append([r['accuracy'], r['macro_f1'], r['macro_prec'], r['macro_rec']])
        
    data = np.array(data)
    x = np.arange(len(metric_names))
    width = 0.35
    colors = ['#00d4ff', '#ff6b2c']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (model, color) in enumerate(zip(models, colors)):
        bars = ax.bar(x + i*width - width/2, data[i], width, label=model, color=color, edgecolor='white')
        for bar, val in zip(bars, data[i]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
            
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('ANGELINA: CNN vs SVM Head-to-Head Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, fontsize=11)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"  📊  Saved: {save_path}")

def plot_roc_curves(y_test, y_probs_cnn, y_probs_svm, classes, save_path):
    from sklearn.preprocessing import label_binarize
    from sklearn.metrics import roc_curve, auc
    
    y_bin = label_binarize(y_test, classes=range(len(classes)))
    colors_cnn = ['#00e68a', '#ff2244', '#ff6b2c', '#ffaa00']
    colors_svm = ['#0055ff', '#8800ff', '#ff00aa', '#00aaff']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # CNN ROC
    for i, (cls, color) in enumerate(zip(classes, colors_cnn)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_probs_cnn[:, i])
        roc_auc = auc(fpr, tpr)
        ax1.plot(fpr, tpr, color=color, lw=2, label=f'{cls} (AUC = {roc_auc:.3f})')
    ax1.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate', fontsize=12)
    ax1.set_ylabel('True Positive Rate', fontsize=12)
    ax1.set_title('CNN-LSTM ROC Curves', fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=10)
    ax1.grid(True, alpha=0.3)

    # SVM ROC
    for i, (cls, color) in enumerate(zip(classes, colors_svm)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_probs_svm[:, i])
        roc_auc = auc(fpr, tpr)
        ax2.plot(fpr, tpr, color=color, lw=2, label=f'{cls} (AUC = {roc_auc:.3f})')
    ax2.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('False Positive Rate', fontsize=12)
    ax2.set_ylabel('True Positive Rate', fontsize=12)
    ax2.set_title('Classical SVM ROC Curves', fontsize=14, fontweight='bold')
    ax2.legend(loc='lower right', fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"  📊  Saved: {save_path}")

def main():
    print(f"\n{'='*65}")
    print("  PROJECT ANGELINA — CNN vs SVM Comparison")
    print(f"{'='*65}\n")

    for p in [INPUT_CSV, CNN_MODEL_PATH, SVM_MODEL_PATH, SCALER_PATH, LABEL_MAP_PATH]:
        if not os.path.exists(p):
            print(f"  ❌  Missing: {p}")
            sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("  [1/4] Loading and aligning dataset...")
    df = pd.read_csv(INPUT_CSV)
    feature_cols = [c for c in df.columns if c not in [LABEL_COL] + DROP_COLS]
    df = df.dropna(subset=feature_cols)

    le = LabelEncoder()
    df['encoded_label'] = le.fit_transform(df[LABEL_COL])
    classes = list(le.classes_)

    scaler = joblib.load(SCALER_PATH)
    df[feature_cols] = scaler.transform(df[feature_cols].values)

    X_train_windows, y_train_windows = [], []
    X_test_windows, y_test_windows = [], []
    subjects = sorted(df['subject_id'].unique())
    split_idx = int(len(subjects) * 0.8)
    train_subjects = set(subjects[:split_idx])

    for (subj, lbl), group in df.groupby(['subject_id', LABEL_COL]):
        group_X = group[feature_cols].values
        group_y = group['encoded_label'].values
        
        if len(group_X) >= WINDOW_SIZE:
            xw, yw = create_sliding_windows(group_X, group_y, WINDOW_SIZE)
            if subj in train_subjects:
                X_train_windows.append(xw)
                y_train_windows.append(yw)
            else:
                X_test_windows.append(xw)
                y_test_windows.append(yw)

    X_train = np.concatenate(X_train_windows, axis=0) if X_train_windows else np.array([])
    y_train = np.concatenate(y_train_windows, axis=0) if y_train_windows else np.array([])
    X_test = np.concatenate(X_test_windows, axis=0) if X_test_windows else np.array([])
    y_test = np.concatenate(y_test_windows, axis=0) if y_test_windows else np.array([])
    
    print("\n  🧬  Injecting synthetic geometric noise to simulate real-world variance...")
    noise_factor = 0.6  # std of 0.6 relative to scaled features
    X_test = X_test + np.random.normal(0, noise_factor, X_test.shape)

    print(f"       Train windows aligned : {len(X_train)}")
    print(f"       Test windows aligned  : {len(X_test)}")

    print("\n  [2/4] Running 1D-CNN (Temporal Sequence) Inference...")
    cnn_model = tf.keras.models.load_model(CNN_MODEL_PATH)
    y_probs_cnn_test = cnn_model.predict(X_test, verbose=0)
    y_pred_cnn_test  = np.argmax(y_probs_cnn_test, axis=1)
    
    y_probs_cnn_train = cnn_model.predict(X_train, verbose=0)
    y_pred_cnn_train  = np.argmax(y_probs_cnn_train, axis=1)

    print("  [3/4] Running SVM (Single-Frame) Inference...")
    svm_model = joblib.load(SVM_MODEL_PATH)
    
    X_test_svm = X_test[:, -1, :]
    y_probs_svm_test = svm_model.predict_proba(X_test_svm)
    y_pred_svm_test = np.argmax(y_probs_svm_test, axis=1)
    
    X_train_svm = X_train[:, -1, :]
    y_probs_svm_train = svm_model.predict_proba(X_train_svm)
    y_pred_svm_train = np.argmax(y_probs_svm_train, axis=1)

    print("\n  [4/4] Generating comparison report...")
    metrics = {
        'CNN-LSTM Hybrid': {
            'accuracy': accuracy_score(y_test, y_pred_cnn_test),
            'macro_f1': f1_score(y_test, y_pred_cnn_test, average='macro'),
            'macro_prec': precision_score(y_test, y_pred_cnn_test, average='macro', zero_division=0),
            'macro_rec': recall_score(y_test, y_pred_cnn_test, average='macro', zero_division=0)
        },
        'Classical SVM': {
            'accuracy': accuracy_score(y_test, y_pred_svm_test),
            'macro_f1': f1_score(y_test, y_pred_svm_test, average='macro'),
            'macro_prec': precision_score(y_test, y_pred_svm_test, average='macro', zero_division=0),
            'macro_rec': recall_score(y_test, y_pred_svm_test, average='macro', zero_division=0)
        }
    }
    
    train_metrics = {
        'CNN-LSTM Hybrid': accuracy_score(y_train, y_pred_cnn_train),
        'Classical SVM': accuracy_score(y_train, y_pred_svm_train)
    }

    plot_comparison_bar(metrics, os.path.join(OUTPUT_DIR, "cnn_vs_svm_comparison.png"))
    plot_roc_curves(y_test, y_probs_cnn_test, y_probs_svm_test, classes, os.path.join(OUTPUT_DIR, "cnn_vs_svm_roc.png"))

    report_lines = []
    report_lines.append("=" * 65)
    report_lines.append("  ANGELINA MODEL COMPARISON REPORT (CNN vs SVM)")
    report_lines.append("=" * 65)
    report_lines.append(f"\n  {'Metric':<20} | {'CNN-LSTM':<18} | {'Classical SVM':<18}")
    report_lines.append("-" * 65)
    
    m_cnn = metrics['CNN-LSTM Hybrid']
    m_svm = metrics['Classical SVM']
    report_lines.append(f"  {'Train Accuracy':<20} | {train_metrics['CNN-LSTM Hybrid']*100:>7.2f}%            | {train_metrics['Classical SVM']*100:>7.2f}%")
    report_lines.append(f"  {'Test Accuracy (Noisy)':<20} | {m_cnn['accuracy']*100:>7.2f}%            | {m_svm['accuracy']*100:>7.2f}%")
    report_lines.append(f"  {'Macro F1':<20} | {m_cnn['macro_f1']:>9.4f}           | {m_svm['macro_f1']:>9.4f}")
    report_lines.append(f"  {'Macro Precision':<20} | {m_cnn['macro_prec']:>9.4f}           | {m_svm['macro_prec']:>9.4f}")
    report_lines.append(f"  {'Macro Recall':<20} | {m_cnn['macro_rec']:>9.4f}           | {m_svm['macro_rec']:>9.4f}")
    
    report_lines.append("\n" + "=" * 65)
    report_lines.append("  CNN CLASSIFICATION REPORT (TEST SET)")
    report_lines.append("=" * 65 + "\n")
    report_lines.append(classification_report(y_test, y_pred_cnn_test, target_names=classes))
    
    report_lines.append("\n" + "=" * 65)
    report_lines.append("  SVM CLASSIFICATION REPORT (TEST SET)")
    report_lines.append("=" * 65 + "\n")
    report_lines.append(classification_report(y_test, y_pred_svm_test, target_names=classes))
    
    report_path = os.path.join(OUTPUT_DIR, "cnn_vs_svm_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(report_lines))
    print(f"  📄  Saved: {report_path}")

    print(f"\n{'='*65}")
    print("  ✅ COMPARISON COMPLETE")
    print(f"{'='*65}\n")
    print("\n".join(report_lines[:12]))

if __name__ == "__main__":
    main()
