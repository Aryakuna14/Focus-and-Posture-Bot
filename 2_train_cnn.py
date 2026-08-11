import sys
if sys.stdout is not None and hasattr(sys.stdout, 'encoding') and sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

import os
import logging
# Suppress C++ logs BEFORE importing tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GLOG_minloglevel'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
logging.getLogger('absl').setLevel(logging.ERROR)

import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(3)

import keras
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Flatten
from keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_CSV = os.path.join(BASE_DIR, "posture_dataset.csv")
MODEL_OUT = os.path.join(BASE_DIR, "angelina_cnn_model.keras")
SCALER_OUT = os.path.join(BASE_DIR, "angelina_scaler.pkl")
LABEL_ENCODER_OUT = os.path.join(BASE_DIR, "angelina_label_encoder.pkl")
LABEL_MAP_OUT = os.path.join(BASE_DIR, "angelina_label_map.pkl")

WINDOW_SIZE = 30
LABEL_COL = "label"
DROP_COLS = ["subject_id"]

def create_sliding_windows(X, y, window_size):
    """
    X: (N, features)
    y: (N,)
    Returns X_out: (N-window_size+1, window_size, features), y_out: (N-window_size+1,)
    """
    X_out, y_out = [], []
    for i in range(len(X) - window_size + 1):
        X_out.append(X[i : i + window_size])
        y_out.append(y[i + window_size - 1]) # Label of the LAST frame in the window
    return np.array(X_out), np.array(y_out)

def load_and_preprocess(csv_path):
    os.system('cls' if os.name == 'nt' else 'clear')
    print(f"\n{'='*55}")
    print("  PROJECT ANGELINA — 1D-CNN Trainer")
    print(f"{'='*55}")

    df = pd.read_csv(csv_path)
    print(f"\n  📂  Loaded  : {csv_path}")
    print(f"      Rows    : {df.shape[0]}")
    
    # 1. Clean NaN
    feature_cols = [c for c in df.columns if c not in [LABEL_COL] + DROP_COLS]
    df = df.dropna(subset=feature_cols)

    # 2. Fit Scaler & LabelEncoder on the flat data first
    le = LabelEncoder()
    df['encoded_label'] = le.fit_transform(df[LABEL_COL])

    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols].values)

    # 3. Build sliding windows PER SUBJECT PER LABEL with SESSION SPLIT
    # We must split by subject_id (recording session) to prevent overlap leakage
    # while ensuring each session has enough frames to form 30-frame windows.
    X_train_windows, y_train_windows = [], []
    X_test_windows, y_test_windows = [], []
    
    subjects = sorted(df['subject_id'].unique())
    split_idx = int(len(subjects) * 0.8)
    train_subjects = set(subjects[:split_idx])

    for (subj, lbl), group in df.groupby(['subject_id', LABEL_COL]):
        group_X = group[feature_cols].values
        group_y = group['encoded_label'].values
        
        if len(group_X) < WINDOW_SIZE:
            continue
            
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

    print("\n  🧬  Injecting synthetic geometric noise to validation set to simulate real-world variance...")
    noise_factor = 0.6  # std of 0.6 relative to scaled features
    X_test = X_test + np.random.normal(0, noise_factor, X_test.shape)

    print(f"\n  🧩  Generated temporal windows of size {WINDOW_SIZE}.")
    print(f"      Train   : {len(X_train)} windows")
    print(f"      Test    : {len(X_test)} windows (w/ synthetic noise)")
    print(f"      Classes : {list(le.classes_)}")

    return X_train, X_test, y_train, y_test, le, scaler

def build_model(input_shape, num_classes):
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=128, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        LSTM(64, return_sequences=False),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def main():
    X_train, X_test, y_train, y_test, le, scaler = load_and_preprocess(INPUT_CSV)
    
    model = build_model(input_shape=(WINDOW_SIZE, X_train.shape[2]), num_classes=len(le.classes_))

    print(f"\n  Training model with {len(X_train)} inputs...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=30,
        batch_size=32,
        verbose=0,
        callbacks=[keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)]
    )

    print("\n  Model has been trained.")
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n  Test Accuracy : {acc*100:.2f}%")

    # ── Create output directory ──────────────────────────────
    eval_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "evaluation_results")
    os.makedirs(eval_dir, exist_ok=True)

    # Evaluate
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)

    # ── Per-class FPR / FNR ──────────────────────────────────
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n  {'Class':<22} {'FPR':>8} {'FNR':>8}")
    print(f"  {'─'*22} {'─'*8} {'─'*8}")
    for i, cls in enumerate(le.classes_):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp
        fpr_val = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr_val = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        print(f"  {cls:<22} {fpr_val:>8.4f} {fnr_val:>8.4f}")

    # ── Classification Report (saved) ────────────────────────
    report_text = classification_report(y_test, y_pred, target_names=le.classes_)
    print(f"\n{report_text}")
    report_path = os.path.join(eval_dir, "classification_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("PROJECT ANGELINA — Classification Report (from training)\n")
        f.write("=" * 55 + "\n\n")
        f.write(report_text)
    print(f"  📄  Classification report saved → {report_path}")

    # ── Plot Confusion Matrix ────────────────────────────────
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=le.classes_, yticklabels=le.classes_,
                linewidths=0.5, linecolor='gray')
    plt.title('ANGELINA CNN — Confusion Matrix', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Posture')
    plt.ylabel('Actual Posture')
    plt.tight_layout()
    cm_path = os.path.join(eval_dir, "confusion_matrix_training.png")
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"  📊  Confusion matrix saved → {cm_path}")

    # ── Plot ROC Curves ──────────────────────────────────────
    from sklearn.preprocessing import label_binarize
    from sklearn.metrics import roc_curve, auc

    y_probs = model.predict(X_test, verbose=0)
    y_bin = label_binarize(y_test, classes=range(len(le.classes_)))
    colors = ['#00e68a', '#ff2244', '#ff6b2c', '#ffaa00']

    plt.figure(figsize=(9, 7))
    for i, (cls, color) in enumerate(zip(le.classes_, colors)):
        fpr_r, tpr_r, _ = roc_curve(y_bin[:, i], y_probs[:, i])
        roc_auc = auc(fpr_r, tpr_r)
        plt.plot(fpr_r, tpr_r, color=color, lw=2,
                 label=f'{cls} (AUC = {roc_auc:.3f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Random Chance')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ANGELINA CNN — Multi-Class ROC Curves (One-vs-Rest)', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    roc_path = os.path.join(eval_dir, "roc_curves_training.png")
    plt.savefig(roc_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"  📊  ROC curves saved → {roc_path}")

    # ── Plot Training History ────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(history.history['accuracy'], label='Train Accuracy', color='#00d4ff', lw=2)
    ax1.plot(history.history['val_accuracy'], label='Val Accuracy', color='#ff6b2c', lw=2)
    ax1.set_title('Model Accuracy', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(history.history['loss'], label='Train Loss', color='#00d4ff', lw=2)
    ax2.plot(history.history['val_loss'], label='Val Loss', color='#ff6b2c', lw=2)
    ax2.set_title('Model Loss', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    hist_path = os.path.join(eval_dir, "training_history.png")
    plt.savefig(hist_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"  📊  Training history saved → {hist_path}")

    # Save
    model.save(MODEL_OUT)
    joblib.dump(scaler, SCALER_OUT)
    joblib.dump(le, LABEL_ENCODER_OUT)
    
    label_map = {i: name for i, name in enumerate(le.classes_)}
    joblib.dump(label_map, LABEL_MAP_OUT)

    print(f"\n  💾  Model saved  → {MODEL_OUT}")
    print(f"  💾  Scaler saved → {SCALER_OUT}")

if __name__ == "__main__":
    main()

