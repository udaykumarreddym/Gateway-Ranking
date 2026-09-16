"""
======================================================================
STEP 02 - TRAIN FINAL GATEWAY FAULT MODEL
======================================================================

Purpose:
    Train the final HistGradientBoosting model using all historical
    gateway-week observations.

Input:
    outputs/gateway_week_features_reduced.csv

Output:
    models/gateway_fault_model_weight2.joblib

Model:
    HistGradientBoostingClassifier

Final configuration:
    max_iter=300
    learning_rate=0.05
    max_leaf_nodes=15
    l2_regularization=2.0
    random_state=42

Class weighting:
    Negative class = 1
    Positive class = 2

Training data:
    7,304 historical gateway-weeks
    91 positive
    7,213 negative

IMPORTANT:
    This is the final training model. No challenge-week data is used
    during training.
======================================================================
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import HistGradientBoostingClassifier


# =====================================================================
# PATHS
# =====================================================================

ROOT = Path(__file__).resolve().parent.parent

OUTPUT = ROOT / "outputs"
MODELS = ROOT / "models"

OUTPUT.mkdir(
    parents=True,
    exist_ok=True,
)

MODELS.mkdir(
    parents=True,
    exist_ok=True,
)


INPUT_FILE = (
    OUTPUT /
    "gateway_week_features_reduced.csv"
)

MODEL_FILE = (
    MODELS /
    "gateway_fault_model_weight2.joblib"
)


# =====================================================================
# FEATURE DEFINITIONS
# =====================================================================

ID_COLUMNS = [
    "gateway_id",
    "week_start",
]

TARGET_COLUMN = "fault_next_7d"


CATEGORICAL_COLUMNS = [
    "tenant",
    "site_type",
    "region",
    "hw_model",
    "antenna_type",
    "fw_version",
]


# =====================================================================
# LOAD DATA
# =====================================================================

def load_data():

    print("=" * 70)
    print("1. LOADING HISTORICAL DATASET")
    print("=" * 70)

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE
    )

    print(
        f"Rows: {len(df):,}"
    )

    print(
        f"Columns: {len(df.columns)}"
    )

    return df


# =====================================================================
# PREPARE FEATURES
# =====================================================================

def prepare_features(df):

    print()
    print("=" * 70)
    print("2. PREPARING MODEL FEATURES")
    print("=" * 70)

    if TARGET_COLUMN not in df.columns:

        raise ValueError(
            f"Missing target column: {TARGET_COLUMN}"
        )

    missing_categorical = [
        column
        for column in CATEGORICAL_COLUMNS
        if column not in df.columns
    ]

    if missing_categorical:

        raise ValueError(
            "Missing categorical columns: "
            f"{missing_categorical}"
        )

    # -------------------------------------------------------------
    # Remove identifiers and target.
    # -------------------------------------------------------------

    excluded_columns = (
        ID_COLUMNS
        + [TARGET_COLUMN]
    )

    feature_columns = [
        column
        for column in df.columns
        if column not in excluded_columns
    ]

    X = df[
        feature_columns
    ].copy()

    y = df[
        TARGET_COLUMN
    ].astype(int)

    # -------------------------------------------------------------
    # Identify numerical columns.
    # -------------------------------------------------------------

    numeric_columns = [
        column
        for column in feature_columns
        if column not in CATEGORICAL_COLUMNS
    ]

    print(
        f"Total model features: "
        f"{len(feature_columns)}"
    )

    print(
        f"Categorical features: "
        f"{len(CATEGORICAL_COLUMNS)}"
    )

    print(
        f"Numeric features: "
        f"{len(numeric_columns)}"
    )

    print()
    print("Categorical columns:")

    for column in CATEGORICAL_COLUMNS:
        print(f"  - {column}")

    print()
    print("Target distribution:")
    print(
        y.value_counts()
        .sort_index()
    )

    return (
        X,
        y,
        numeric_columns,
    )


# =====================================================================
# BUILD PREPROCESSOR
# =====================================================================

def build_preprocessor(
    numeric_columns,
):

    print()
    print("=" * 70)
    print("3. BUILDING PREPROCESSOR")
    print("=" * 70)

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                ),
            ),
            (
                "encoder",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_columns,
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_COLUMNS,
            ),
        ],
        remainder="drop",
    )

    return preprocessor


# =====================================================================
# BUILD FINAL MODEL
# =====================================================================

def build_model():

    print()
    print("=" * 70)
    print("4. BUILDING FINAL MODEL")
    print("=" * 70)

    model = HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.05,
        max_leaf_nodes=15,
        l2_regularization=2.0,
        random_state=42,
    )

    print(
        "Model: HistGradientBoostingClassifier"
    )

    print(
        "max_iter: 300"
    )

    print(
        "learning_rate: 0.05"
    )

    print(
        "max_leaf_nodes: 15"
    )

    print(
        "l2_regularization: 2.0"
    )

    print(
        "random_state: 42"
    )

    return model


# =====================================================================
# TRAIN
# =====================================================================

def train_model(
    X,
    y,
    numeric_columns,
):

    print()
    print("=" * 70)
    print("5. TRAINING FINAL MODEL")
    print("=" * 70)

    preprocessor = build_preprocessor(
        numeric_columns
    )

    model = build_model()

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                model,
            ),
        ]
    )

    # -------------------------------------------------------------
    # Positive-class weighting.
    #
    # Final validated model uses:
    #
    #     negative = 1
    #     positive = 2
    #
    # HistGradientBoosting does not use class_weight directly in the
    # version used for this project, so equivalent sample weights
    # are supplied during fitting.
    # -------------------------------------------------------------

    sample_weight = np.where(
        y == 1,
        2.0,
        1.0,
    )

    print(
        f"Training rows: {len(X):,}"
    )

    print(
        f"Positive rows: {int(y.sum()):,}"
    )

    print(
        f"Negative rows: "
        f"{int((y == 0).sum()):,}"
    )

    print(
        "Positive sample weight: 2.0"
    )

    print(
        "Negative sample weight: 1.0"
    )

    pipeline.fit(
        X,
        y,
        model__sample_weight=sample_weight,
    )

    print()
    print("[PASS] Model training complete.")

    return pipeline


# =====================================================================
# VERIFY MODEL
# =====================================================================

def verify_model(
    pipeline,
    X,
):

    print()
    print("=" * 70)
    print("6. VERIFYING TRAINED MODEL")
    print("=" * 70)

    probabilities = pipeline.predict_proba(
        X
    )[:, 1]

    if len(probabilities) != len(X):

        raise ValueError(
            "Prediction count does not match "
            "training row count."
        )

    if not np.isfinite(
        probabilities
    ).all():

        raise ValueError(
            "Model produced non-finite probabilities."
        )

    print(
        f"Prediction rows: "
        f"{len(probabilities):,}"
    )

    print(
        f"Minimum probability: "
        f"{probabilities.min():.8f}"
    )

    print(
        f"Maximum probability: "
        f"{probabilities.max():.8f}"
    )

    print(
        f"Mean probability: "
        f"{probabilities.mean():.8f}"
    )

    print()
    print("[PASS] Model verification complete.")


# =====================================================================
# SAVE
# =====================================================================

def save_model(
    pipeline,
):

    print()
    print("=" * 70)
    print("7. SAVING FINAL MODEL")
    print("=" * 70)

    joblib.dump(
        pipeline,
        MODEL_FILE,
    )

    if not MODEL_FILE.exists():

        raise RuntimeError(
            "Model file was not created."
        )

    size_mb = (
        MODEL_FILE.stat().st_size
        /
        (1024 * 1024)
    )

    print(
        f"Saved: "
        f"{MODEL_FILE.relative_to(ROOT)}"
    )

    print(
        f"Model size: "
        f"{size_mb:.2f} MB"
    )

    print()
    print("[PASS] Final model saved.")


# =====================================================================
# MAIN
# =====================================================================

def main():

    print()
    print("=" * 70)
    print("STEP 02 - TRAIN FINAL GATEWAY FAULT MODEL")
    print("=" * 70)

    df = load_data()

    X, y, numeric_columns = (
        prepare_features(df)
    )

    pipeline = train_model(
        X,
        y,
        numeric_columns,
    )

    verify_model(
        pipeline,
        X,
    )

    save_model(
        pipeline,
    )

    print()
    print("=" * 70)
    print("STEP 02 COMPLETE")
    print("=" * 70)

    print()
    print(
        "Final model:"
    )

    print(
        f"  {MODEL_FILE.relative_to(ROOT)}"
    )


if __name__ == "__main__":
    main()