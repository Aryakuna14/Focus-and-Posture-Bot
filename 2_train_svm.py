"""
PROJECT ANGELINA — Neural-Ergonomic Focus Bot
============================================================
SCRIPT 2: SVM MODEL TRAINER
Purpose : Load posture_dataset.csv, preprocess, train an SVM,
          evaluate it, and save the model + scaler as .pkl files
          ready for real-time inference.
============================================================
USAGE:
  python 2_train_svm.py

OUTPUT FILES:
  angelina_svm_model.pkl   ← Trained SVM classifier
  angelina_scaler.pkl      ← Feature StandardScaler (MUST ship with model)
============================================================
INSTALL DEPENDENCIES:
  pip install pandas scikit-learn matplotlib seaborn joblib
"""

import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.svm            import SVC
from sklearn.preprocessing  import StandardScaler, LabelEncoder
from sklearn.model_selection import (train_test_split, GridSearchCV,
                                      StratifiedKFold, cross_val_score)
from sklearn.metrics        import (classification_report, confusion_matrix,
                                    ConfusionMatrixDisplay)
from sklearn.pipeline       import Pipeline

# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────
INPUT_CSV    = "posture_dataset.csv"
MODEL_OUT    = "angelina_svm_model.pkl"
SCALER_OUT   = "angelina_scaler.pkl"
LABEL_COL    = "label"
DROP_COLS    = ["subject_id"]         # columns to exclude from features
TEST_SIZE    = 0.20                   # 80/20 train-test split
RANDOM_STATE = 42
RUN_GRIDSEARCH = True                 # Set False for quick training


def load_and_inspect(csv_path: str) -> pd.DataFrame:
    print(f"\n{'='*55}")
    print("  PROJECT ANGELINA — SVM Trainer")
    print(f"{'='*55}")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"❌  '{csv_path}' not found.\n"
            "    Run 1_data_collector.py first to generate data."
        )

    df = pd.read_csv(csv_path)
    print(f"\n  📂  Loaded  : {csv_path}")
    print(f"      Shape   : {df.shape[0]} rows × {df.shape[1]} cols")
    print(f"\n  Label distribution:")
    print(df[LABEL_COL].value_counts().to_string(header=False))

    # Minimum data guard
    for label, count in df[LABEL_COL].value_counts().items():
        if count < 30:
            print(f"\n  ⚠️  Warning: '{label}' has only {count} samples.")
            print("      Aim for ≥100 per class for reliable SVM boundaries.")

    return df


def preprocess(df: pd.DataFrame):
    """Split features / labels, encode labels, scale features using Temporal Split."""
    feature_cols = [c for c in df.columns if c not in [LABEL_COL] + DROP_COLS]
    
    # 1. Clean NaN
    df = df.dropna(subset=feature_cols)
    
    # 2. Encode Labels
    le = LabelEncoder()
    df['encoded_label'] = le.fit_transform(df[LABEL_COL])
    print(f"\n  Classes : {list(le.classes_)}")
    
    # 3. Scale Features
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols].values)
    
    # 4. Strict SESSION SPLIT PER SUBJECT PER LABEL
    X_train_list, y_train_list = [], []
    X_test_list, y_test_list = [], []
    
    subjects = sorted(df['subject_id'].unique())
    split_idx = int(len(subjects) * 0.8)
    train_subjects = set(subjects[:split_idx])
    
    for (subj, lbl), group in df.groupby(['subject_id', LABEL_COL]):
        group_X = group[feature_cols].values
        group_y = group['encoded_label'].values
        
        if subj in train_subjects:
            X_train_list.append(group_X)
            y_train_list.append(group_y)
        else:
            X_test_list.append(group_X)
            y_test_list.append(group_y)
        
    X_train = np.concatenate(X_train_list, axis=0) if X_train_list else np.array([])
    y_train = np.concatenate(y_train_list, axis=0) if y_train_list else np.array([])
    X_test = np.concatenate(X_test_list, axis=0) if X_test_list else np.array([])
    y_test = np.concatenate(y_test_list, axis=0) if y_test_list else np.array([])
    
    print("\n  🧬  Injecting synthetic geometric noise to test set to simulate real-world variance...")
    noise_factor = 0.6  # std of 0.6 relative to scaled features
    X_test = X_test + np.random.normal(0, noise_factor, X_test.shape)

    return X_train, X_test, y_train, y_test, le, scaler, feature_cols


def train(X_train, X_test, y_train, y_test, run_grid: bool):
    """Train SVM with optional GridSearch."""

    if run_grid:
        print("\n  🔍  Running GridSearchCV (this may take ~1–3 min)…")
        param_grid = {
            'C':      [0.1, 1, 10, 100],
            'gamma':  ['scale', 'auto', 0.01, 0.001],
            'kernel': ['rbf', 'poly'],
        }
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        grid = GridSearchCV(
            SVC(probability=True, class_weight='balanced'),
            param_grid, cv=cv,
            scoring='f1_macro', n_jobs=-1, verbose=1
        )
        grid.fit(X_train, y_train)
        best_params = grid.best_params_
        print(f"\n  ✅  Best params : {best_params}")
        print(f"      CV F1 macro : {grid.best_score_:.4f}")
        clf = grid.best_estimator_
    else:
        print("\n  ⚡  Training SVM with default RBF kernel…")
        clf = SVC(C=10, gamma='scale', kernel='rbf',
                  probability=True, class_weight='balanced',
                  random_state=RANDOM_STATE)
        clf.fit(X_train, y_train)

    return clf, X_train, X_test, y_train, y_test


def evaluate(clf, X_train, X_test, y_train, y_test, le):
    """Print metrics, plot confusion matrix."""
    y_pred = clf.predict(X_test)

    print(f"\n{'─'*55}")
    print("  EVALUATION RESULTS")
    print(f"{'─'*55}")
    print(f"  Train accuracy : {clf.score(X_train, y_train):.4f}")
    print(f"  Test  accuracy : {clf.score(X_test,  y_test):.4f}")
    print(f"\n  Classification Report:\n")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # ── 5-Fold Cross-Validation ──────────────────────────────
    # Run CV only on the training split to avoid data leakage from
    # the test set that was used during GridSearch model selection.
    cv_scores = cross_val_score(
        clf, X_train, y_train,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        scoring='f1_macro', n_jobs=-1
    )
    print(f"  5-Fold CV F1 (macro, train-only): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # ── Confusion Matrix ─────────────────────────────────────
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=le.classes_, yticklabels=le.classes_, ax=ax
    )
    ax.set_title("Project ANGELINA — SVM Confusion Matrix", fontsize=13, fontweight='bold')
    ax.set_ylabel("True Label"); ax.set_xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig("angelina_confusion_matrix.png", dpi=150)
    print("\n  📊  Confusion matrix saved → 'angelina_confusion_matrix.png'")
    plt.show()

    # ── Feature Importance proxy via SVM weights (linear) ────
    # For RBF kernel this is not directly available; skip gracefully.
    if clf.kernel == 'linear':
        importances = np.abs(clf.coef_).mean(axis=0)
        print("\n  Top 10 most discriminative features (linear SVM):")
        top_idx = np.argsort(importances)[::-1][:10]
        for rank, i in enumerate(top_idx, 1):
            print(f"    {rank:2d}. Feature index {i:3d} — weight {importances[i]:.4f}")


def save_artifacts(clf, scaler, le):
    """Persist model, scaler, and label mapping."""
    joblib.dump(clf,    MODEL_OUT)
    joblib.dump(scaler, SCALER_OUT)

    # Save human-readable label map for inference script
    label_map = {i: name for i, name in enumerate(le.classes_)}
    joblib.dump(label_map, "angelina_label_map.pkl")
    joblib.dump(le,         "angelina_label_encoder.pkl")

    print(f"\n  💾  Model  saved → '{MODEL_OUT}'")
    print(f"  💾  Scaler saved → '{SCALER_OUT}'")
    print(f"  💾  Label map    → 'angelina_label_map.pkl'")
    print(f"\n  Ship these 3 files alongside your inference script.\n")


def main():
    df = load_and_inspect(INPUT_CSV)
    X_tr, X_te, y_tr, y_te, le, scaler, feature_cols = preprocess(df)
    clf, X_tr, X_te, y_tr, y_te = train(X_tr, X_te, y_tr, y_te, run_grid=RUN_GRIDSEARCH)
    evaluate(clf, X_tr, X_te, y_tr, y_te, le)
    save_artifacts(clf, scaler, le)


if __name__ == "__main__":
    main()
