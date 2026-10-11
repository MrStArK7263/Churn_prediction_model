"""
train.py – end-to-end training script.

Run from the project root:
    python src/train.py

This script:
  1. Loads and cleans the data  (using src/data_cleaning.py)
  2. Engineers features          (using src/feature_engineering.py)
  3. Splits into train/test      (stratified, random_state=42)
  4. Builds sklearn Pipelines for 4 models
  5. Evaluates all models with cross-validation and hold-out metrics
  6. Tunes the best 2 with GridSearchCV
  7. Selects the final model
  8. Saves the Pipeline + metadata to models/

For beginners: read top to bottom; every step has a comment explaining WHY.
"""

import sys
import json
import time
import logging
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
import sklearn

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

# ── Make sure the project root is on sys.path so we can import src/ ──────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    RAW_DATA_PATH, MODELS_DIR, MODEL_PATH, METADATA_PATH,
    MODEL_VERSION, TARGET_COL, POSITIVE_CLASS, COLS_TO_DROP,
    RANDOM_STATE, TEST_SIZE, CV_FOLDS, DEFAULT_THRESHOLD,
)
from src.data_cleaning import load_data, full_clean
from src.feature_engineering import build_features_df, encode_target, ChurnFeatureEngineer

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(model, X_test, y_test):
    """
    Compute accuracy, precision, recall, F1, ROC-AUC.

    WHY: We need multiple metrics because accuracy alone is misleading
    on imbalanced data.  A model that predicts "No churn" for everyone
    would get ~73% accuracy but would miss every churner.
    Recall of the churn class (1) is our most important metric –
    it tells us what fraction of actual churners we caught.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, pos_label=1, zero_division=0), 4),
        "recall":    round(recall_score(y_test, y_pred, pos_label=1, zero_division=0), 4),
        "f1":        round(f1_score(y_test, y_pred, pos_label=1, zero_division=0), 4),
        "roc_auc":   round(roc_auc_score(y_test, y_prob), 4),
    }


def build_preprocessor(X: pd.DataFrame):
    """
    Build a ColumnTransformer that:
      - Imputes + scales numeric columns
      - Imputes + one-hot-encodes nominal categorical columns
      - Imputes + ordinal-encodes the Contract column

    WHY we use a Pipeline inside ColumnTransformer:
      Imputers must be fit only on the training data.  By putting them
      inside a Pipeline they automatically never see the test data.

    WHY StandardScaler:
      Logistic Regression is sensitive to feature scale.  Tree-based
      models do not need scaling but it does not hurt them.

    WHY median imputation for numerics:
      The median is robust to outliers.  A single very large TotalCharges
      value would skew the mean, so median is safer.

    WHY mode imputation for categoricals:
      For categorical columns the most frequent value is the safest
      default when we do not have a label to guide us.
    """
    # Identify numeric and categorical columns in the feature-engineered DataFrame
    # (These are determined after build_features_df has been called)
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    # Contract gets ordinal encoding (order matters: month-to-month > one year > two year for churn risk)
    contract_col = ["Contract"] if "Contract" in X.columns else []
    # tenure_group is also ordinal
    tenure_group_col = ["tenure_group"] if "tenure_group" in X.columns else []
    # All other object/category columns get one-hot encoding
    cat_cols = [
        c for c in X.select_dtypes(include=["object", "category"]).columns
        if c not in contract_col + tenure_group_col
    ]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])

    nominal_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    contract_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(
            categories=[["Month-To-Month", "One Year", "Two Year"]],
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )),
    ])

    tenure_group_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(
            categories=[["0-12 Months", "13-24 Months", "25-48 Months", "49-72 Months"]],
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )),
    ])

    transformers = [("num", numeric_pipeline, num_cols)]
    if cat_cols:
        transformers.append(("cat", nominal_pipeline, cat_cols))
    if contract_col:
        transformers.append(("contract", contract_pipeline, contract_col))
    if tenure_group_col:
        transformers.append(("tenure_grp", tenure_group_pipeline, tenure_group_col))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    return preprocessor


def build_models():
    """
    Return a dict of model name → sklearn estimator.

    WHY class_weight='balanced':
      The dataset has ~73% Non-churn and ~27% Churn.  Without class weights
      the model would optimise for the majority class and get high accuracy
      while completely missing churners.  'balanced' tells sklearn to
      internally up-weight minority class errors so the model tries harder
      to correctly classify churners.
    """
    return {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=RANDOM_STATE,
        ),
        "Decision Tree": DecisionTreeClassifier(
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            class_weight="balanced",
            n_estimators=200,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN TRAINING FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def train():
    logger.info("=" * 60)
    logger.info("CHURN MODEL TRAINING PIPELINE")
    logger.info("=" * 60)

    # ── 1. Load and clean ────────────────────────────────────────────────────
    logger.info("\n[1/8] Loading and cleaning data...")
    df_raw = load_data(str(RAW_DATA_PATH))
    df = full_clean(df_raw)
    logger.info(f"    Clean shape: {df.shape}")

    # ── 2. Feature engineering ───────────────────────────────────────────────
    logger.info("\n[2/8] Engineering features...")
    df = build_features_df(df)

    # ── 3. Separate features and target ──────────────────────────────────────
    logger.info("\n[3/8] Separating features and target...")
    y = encode_target(df[TARGET_COL])
    X = df.drop(columns=[TARGET_COL])

    logger.info(f"    Class distribution – 0 (No churn): {(y==0).sum()}, 1 (Churn): {(y==1).sum()}")

    # ── 4. Train / test split (stratified) ───────────────────────────────────
    # WHY stratify=y: ensures both sets have the same 73/27 ratio so neither
    # set is accidentally all non-churners or all churners.
    logger.info("\n[4/8] Splitting data (80/20, stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    logger.info(f"    Train: {len(X_train)} rows | Test: {len(X_test)} rows")

    # ── 5. Build preprocessor on training shapes ──────────────────────────────
    logger.info("\n[5/8] Training and evaluating all models...")
    preprocessor = build_preprocessor(X_train)
    models = build_models()

    results = {}
    fitted_pipelines = {}
    cv_splitter = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    for name, clf in models.items():
        logger.info(f"    → {name}")
        # Build the full pipeline: preprocessor → classifier
        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", clf),
        ])
        # Cross-validate on training data
        cv_scores = cross_validate(
            pipe, X_train, y_train,
            cv=cv_splitter,
            scoring=["f1", "recall", "roc_auc"],
            return_train_score=True,
        )
        # Fit on full training set
        pipe.fit(X_train, y_train)
        fitted_pipelines[name] = pipe

        # Hold-out metrics
        test_metrics = compute_metrics(pipe, X_test, y_test)

        results[name] = {
            "cv_f1_mean":     round(cv_scores["test_f1"].mean(), 4),
            "cv_recall_mean": round(cv_scores["test_recall"].mean(), 4),
            "cv_roc_mean":    round(cv_scores["test_roc_auc"].mean(), 4),
            **{f"test_{k}": v for k, v in test_metrics.items()},
        }
        logger.info(f"      Test → Acc: {test_metrics['accuracy']} | "
                    f"Recall: {test_metrics['recall']} | F1: {test_metrics['f1']} | "
                    f"ROC-AUC: {test_metrics['roc_auc']}")

    # ── 6. Hyperparameter tuning (best 2 by F1) ───────────────────────────────
    logger.info("\n[6/8] Hyperparameter tuning (GridSearchCV)...")

    # Sort by test F1 and tune top 2
    sorted_models = sorted(results.items(), key=lambda x: x[1]["test_f1"], reverse=True)
    top2_names = [name for name, _ in sorted_models[:2]]

    param_grids = {
        "Logistic Regression": {
            "classifier__C": [0.01, 0.1, 1, 10],
            "classifier__solver": ["lbfgs", "liblinear"],
        },
        "Decision Tree": {
            "classifier__max_depth": [3, 5, 10, None],
            "classifier__min_samples_leaf": [1, 5, 10],
        },
        "Random Forest": {
            "classifier__n_estimators": [100, 200],
            "classifier__max_depth": [5, 10, None],
            "classifier__min_samples_leaf": [1, 5],
        },
    }

    tuned_pipelines = {}
    for name in top2_names:
        if name not in param_grids:
            logger.info(f"    No grid defined for {name}, skipping tuning.")
            tuned_pipelines[name] = fitted_pipelines[name]
            continue

        logger.info(f"    Tuning {name}...")
        base_pipe = fitted_pipelines[name]
        grid = GridSearchCV(
            base_pipe,
            param_grid=param_grids[name],
            scoring="f1",
            cv=cv_splitter,
            n_jobs=-1,
            refit=True,
        )
        grid.fit(X_train, y_train)
        tuned_pipelines[name] = grid.best_estimator_
        tuned_metrics = compute_metrics(grid.best_estimator_, X_test, y_test)
        logger.info(f"      Best params: {grid.best_params_}")
        logger.info(f"      After tuning → F1: {tuned_metrics['f1']} | Recall: {tuned_metrics['recall']}")
        results[name]["tuned_f1"]     = tuned_metrics["f1"]
        results[name]["tuned_recall"] = tuned_metrics["recall"]
        results[name]["tuned_roc_auc"]= tuned_metrics["roc_auc"]

    # ── 7. Select final model ─────────────────────────────────────────────────
    logger.info("\n[7/8] Selecting final model...")

    # Score by: tuned recall (most important) + tuned F1
    # We prefer recall because the cost of missing a churner (false negative) is
    # higher than falsely flagging a loyal customer.
    best_name = None
    best_score = -1
    for name in top2_names:
        recall = results[name].get("tuned_recall", results[name]["test_recall"])
        f1     = results[name].get("tuned_f1",     results[name]["test_f1"])
        score  = 0.6 * recall + 0.4 * f1
        logger.info(f"    {name}: composite score = {score:.4f}")
        if score > best_score:
            best_score = score
            best_name = name

    final_model = tuned_pipelines.get(best_name, fitted_pipelines[best_name])
    final_metrics = compute_metrics(final_model, X_test, y_test)
    logger.info(f"    ✓ Selected: {best_name}")
    logger.info(f"      Metrics → {final_metrics}")

    # ── Measure inference time ───────────────────────────────────────────────
    t0 = time.perf_counter()
    for _ in range(100):
        final_model.predict(X_test.iloc[:1])
    inference_ms = round((time.perf_counter() - t0) / 100 * 1000, 3)
    logger.info(f"    Inference time per row: {inference_ms} ms")

    # ── 8. Save model and metadata ───────────────────────────────────────────
    logger.info("\n[8/8] Saving model and metadata...")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(final_model, MODEL_PATH)
    logger.info(f"    Saved model → {MODEL_PATH}")

    # Model size
    model_size_mb = round(MODEL_PATH.stat().st_size / 1024 / 1024, 3)

    metadata = {
        "model_version":   MODEL_VERSION,
        "model_name":      best_name,
        "trained_at":      datetime.utcnow().isoformat() + "Z",
        "metrics": {
            "accuracy":  final_metrics["accuracy"],
            "precision": final_metrics["precision"],
            "recall":    final_metrics["recall"],
            "f1":        final_metrics["f1"],
            "roc_auc":   final_metrics["roc_auc"],
        },
        "threshold":       DEFAULT_THRESHOLD,
        "feature_names":   [str(c) for c in X_train.columns.tolist()],
        "classes":         ["No Churn", "Churn"],
        "sklearn_version": sklearn.__version__,
        "python_version":  sys.version,
        "inference_time_ms_per_row": inference_ms,
        "model_size_mb":   model_size_mb,
        "all_model_results": results,
    }

    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"    Saved metadata → {METADATA_PATH}")

    # ── Print summary table ───────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 60)
    for name, r in results.items():
        logger.info(
            f"  {name:20s} | "
            f"Acc: {r['test_accuracy']} | "
            f"F1: {r['test_f1']} | "
            f"Recall: {r['test_recall']} | "
            f"ROC-AUC: {r['test_roc_auc']}"
        )
    logger.info(f"\n  ★  Final model: {best_name}")
    logger.info("=" * 60)

    return final_model, metadata


if __name__ == "__main__":
    train()
