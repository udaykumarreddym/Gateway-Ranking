"""
======================================================================
STEP 04 - GENERATE FINAL SUBMISSION
======================================================================

Purpose:
    Reproduce the frozen Step 48 final HGB + 3-sigma hybrid ranking
    and then generate the final predictions.csv submission.

Frozen configuration:

    Model:
        HistGradientBoostingClassifier

    Positive-class weight:
        2

    Hybrid:
        75% ML percentile
        25% 3-sigma baseline percentile

    Selection:
        Top 15 gateways per challenge week

Inputs:

    outputs/gateway_week_features_reduced.csv
    outputs/challenge_gateway_week_features.csv
    outputs/complete_baseline_scores.csv

Outputs:

    models/gateway_fault_model_weight2.joblib
    outputs/final_hybrid_weight2_ranking.csv
    predictions.csv

Final submission:

    week_start
    rank
    gateway_id
    score
    reason

======================================================================
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier


# =====================================================================
# PATHS
# =====================================================================

ROOT = Path(__file__).resolve().parents[1]

HISTORICAL_FILE = (
    ROOT
    / "outputs"
    / "gateway_week_features_reduced.csv"
)

CHALLENGE_FILE = (
    ROOT
    / "outputs"
    / "challenge_gateway_week_features.csv"
)

BASELINE_FILE = (
    ROOT
    / "outputs"
    / "complete_baseline_scores.csv"
)

MODEL_FILE = (
    ROOT
    / "models"
    / "gateway_fault_model_weight2.joblib"
)

RANKING_FILE = (
    ROOT
    / "outputs"
    / "final_hybrid_weight2_ranking.csv"
)

SUBMISSION_FILE = (
    ROOT
    / "predictions.csv"
)


# =====================================================================
# CONFIGURATION
# =====================================================================

TARGET = "fault_next_7d"

GATEWAY_COL = "gateway_id"

WEEK_COL = "week_start"

TOP_K = 15

ML_WEIGHT = 0.75

BASELINE_WEIGHT = 0.25


# =====================================================================
# MODEL PARAMETERS
# =====================================================================

MODEL_PARAMS = {
    "max_iter": 300,
    "learning_rate": 0.05,
    "max_leaf_nodes": 15,
    "l2_regularization": 2.0,
    "random_state": 42,
}


# =====================================================================
# CATEGORICAL FEATURES
# =====================================================================

CATEGORICAL_COLS = [
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

    print()
    print("=" * 75)
    print("1. LOADING DATA")
    print("=" * 75)

    # -------------------------------------------------------------
    # Historical
    # -------------------------------------------------------------

    print("\nLoading historical training dataset...")

    historical = pd.read_csv(
        HISTORICAL_FILE
    )

    historical[WEEK_COL] = pd.to_datetime(
        historical[WEEK_COL],
        utc=True,
    )

    print(
        f"Historical shape: "
        f"{historical.shape}"
    )

    print(
        f"Historical gateways: "
        f"{historical[GATEWAY_COL].nunique()}"
    )

    print(
        f"Historical positives: "
        f"{historical[TARGET].sum()}"
    )

    # -------------------------------------------------------------
    # Challenge
    # -------------------------------------------------------------

    print("\nLoading challenge features...")

    challenge = pd.read_csv(
        CHALLENGE_FILE
    )

    challenge[WEEK_COL] = pd.to_datetime(
        challenge[WEEK_COL],
        utc=True,
    )

    print(
        f"Challenge shape: "
        f"{challenge.shape}"
    )

    print(
        f"Challenge gateways: "
        f"{challenge[GATEWAY_COL].nunique()}"
    )

    print(
        f"Challenge weeks: "
        f"{challenge[WEEK_COL].nunique()}"
    )

    # -------------------------------------------------------------
    # Baseline
    # -------------------------------------------------------------

    print("\nLoading complete 3-sigma baseline...")

    baseline = pd.read_csv(
        BASELINE_FILE
    )

    baseline[WEEK_COL] = pd.to_datetime(
        baseline[WEEK_COL],
        utc=True,
    )

    print(
        f"Baseline shape: "
        f"{baseline.shape}"
    )

    return historical, challenge, baseline


# =====================================================================
# VALIDATE INPUT DATA
# =====================================================================

def validate_inputs(
    historical,
    challenge,
    baseline,
):

    print()
    print("=" * 75)
    print("2. VALIDATING INPUT DATA")
    print("=" * 75)

    # -------------------------------------------------------------
    # Historical target
    # -------------------------------------------------------------

    if TARGET not in historical.columns:
        raise ValueError(
            f"Historical dataset does not contain {TARGET}."
        )

    # -------------------------------------------------------------
    # Challenge size
    # -------------------------------------------------------------

    expected_challenge_rows = 332 * 8

    if len(challenge) != expected_challenge_rows:
        raise ValueError(
            f"Expected {expected_challenge_rows} challenge rows, "
            f"got {len(challenge)}."
        )

    # -------------------------------------------------------------
    # Challenge gateways
    # -------------------------------------------------------------

    if challenge[GATEWAY_COL].nunique() != 332:
        raise ValueError(
            "Challenge dataset does not contain 332 gateways."
        )

    # -------------------------------------------------------------
    # Challenge weeks
    # -------------------------------------------------------------

    if challenge[WEEK_COL].nunique() != 8:
        raise ValueError(
            "Challenge dataset does not contain 8 weeks."
        )

    # -------------------------------------------------------------
    # Duplicate gateway/week
    # -------------------------------------------------------------

    duplicates = challenge.duplicated(
        [
            GATEWAY_COL,
            WEEK_COL,
        ]
    ).sum()

    if duplicates:
        raise ValueError(
            f"Challenge dataset contains "
            f"{duplicates} duplicate gateway/week rows."
        )

    # -------------------------------------------------------------
    # Required baseline columns
    # -------------------------------------------------------------

    required_baseline = {
        GATEWAY_COL,
        WEEK_COL,
        "baseline_flagged_hours",
    }

    missing_baseline = (
        required_baseline
        -
        set(baseline.columns)
    )

    if missing_baseline:
        raise ValueError(
            "Baseline is missing required columns:\n"
            +
            "\n".join(
                f"  - {column}"
                for column in sorted(missing_baseline)
            )
        )

    # -------------------------------------------------------------
    # Baseline duplicates
    # -------------------------------------------------------------

    baseline_duplicates = baseline.duplicated(
        [
            GATEWAY_COL,
            WEEK_COL,
        ]
    ).sum()

    if baseline_duplicates:
        raise ValueError(
            f"Baseline contains "
            f"{baseline_duplicates} duplicate gateway/week rows."
        )

    print(
        "[PASS] Input validation complete."
    )


# =====================================================================
# BUILD FEATURE SCHEMA
# =====================================================================

def get_feature_schema(
    historical,
    challenge,
):

    print()
    print("=" * 75)
    print("3. BUILDING FEATURE SCHEMA")
    print("=" * 75)

    excluded = {
        GATEWAY_COL,
        WEEK_COL,
        TARGET,
    }

    historical_features = [
        column
        for column in historical.columns
        if column not in excluded
    ]

    challenge_features = [
        column
        for column in challenge.columns
        if column not in excluded
    ]

    print(
        f"Historical model features: "
        f"{len(historical_features)}"
    )

    print(
        f"Challenge model features: "
        f"{len(challenge_features)}"
    )

    # -------------------------------------------------------------
    # Exact schema check
    # -------------------------------------------------------------

    if historical_features != challenge_features:

        print("\nHistorical-only features:")

        print(
            sorted(
                set(historical_features)
                -
                set(challenge_features)
            )
        )

        print("\nChallenge-only features:")

        print(
            sorted(
                set(challenge_features)
                -
                set(historical_features)
            )
        )

        raise ValueError(
            "Historical and challenge feature schemas do not match."
        )

    feature_cols = historical_features

    # -------------------------------------------------------------
    # Categorical features
    # -------------------------------------------------------------

    categorical_cols = [
        column
        for column in CATEGORICAL_COLS
        if column in feature_cols
    ]

    numeric_cols = [
        column
        for column in feature_cols
        if column not in categorical_cols
    ]

    print()
    print(
        f"Total features : {len(feature_cols)}"
    )

    print(
        f"Categorical    : {len(categorical_cols)}"
    )

    print(
        f"Numeric        : {len(numeric_cols)}"
    )

    if len(feature_cols) != 51:
        raise ValueError(
            f"Expected 51 model features, "
            f"found {len(feature_cols)}."
        )

    if len(categorical_cols) != 6:
        raise ValueError(
            f"Expected 6 categorical features, "
            f"found {len(categorical_cols)}."
        )

    return feature_cols, categorical_cols, numeric_cols


# =====================================================================
# PREPARE FEATURES
# =====================================================================

def prepare_features(
    historical,
    challenge,
    feature_cols,
    categorical_cols,
):

    print()
    print("=" * 75)
    print("4. PREPARING FEATURES")
    print("=" * 75)

    historical_X = historical[
        feature_cols
    ].copy()

    challenge_X = challenge[
        feature_cols
    ].copy()

    # -------------------------------------------------------------
    # IMPORTANT:
    #
    # Exactly as in frozen Step 48:
    #
    # Historical + challenge are concatenated BEFORE categorical
    # encoding so both datasets use the same category mapping.
    # -------------------------------------------------------------

    combined_X = pd.concat(
        [
            historical_X,
            challenge_X,
        ],
        axis=0,
        ignore_index=True,
    )

    # -------------------------------------------------------------
    # Encode categorical features.
    # -------------------------------------------------------------

    for column in categorical_cols:

        combined_X[column] = (
            combined_X[column]
            .astype("category")
            .cat.codes
            .astype(float)
        )

    # -------------------------------------------------------------
    # Numeric conversion.
    # -------------------------------------------------------------

    for column in feature_cols:

        if column not in categorical_cols:

            combined_X[column] = pd.to_numeric(
                combined_X[column],
                errors="coerce",
            )

    # -------------------------------------------------------------
    # Split back.
    # -------------------------------------------------------------

    n_historical = len(
        historical
    )

    X_train = combined_X.iloc[
        :n_historical
    ].copy()

    X_challenge = combined_X.iloc[
        n_historical:
    ].copy()

    y_train = historical[
        TARGET
    ].astype(int)

    print(
        f"Training matrix: "
        f"{X_train.shape}"
    )

    print(
        f"Challenge matrix: "
        f"{X_challenge.shape}"
    )

    print(
        f"Target positives: "
        f"{int(y_train.sum())}"
    )

    print(
        f"Target negatives: "
        f"{int((y_train == 0).sum())}"
    )

    return X_train, X_challenge, y_train


# =====================================================================
# TRAIN FINAL MODEL
# =====================================================================

def train_model(
    X_train,
    y_train,
):

    print()
    print("=" * 75)
    print("5. TRAINING FINAL HGB MODEL")
    print("=" * 75)

    print(
        "Model: HistGradientBoostingClassifier"
    )

    print(
        "Positive-class weight: 2"
    )

    print(
        f"Training rows: "
        f"{len(X_train)}"
    )

    sample_weights = np.where(
        y_train == 1,
        2.0,
        1.0,
    )

    model = HistGradientBoostingClassifier(
        **MODEL_PARAMS
    )

    model.fit(
        X_train,
        y_train,
        sample_weight=sample_weights,
    )

    print(
        "[PASS] Final HGB model trained."
    )

    return model


# =====================================================================
# SAVE MODEL
# =====================================================================

def save_model(
    model,
    feature_cols,
    categorical_cols,
    numeric_cols,
):

    print()
    print("=" * 75)
    print("6. SAVING FINAL MODEL")
    print("=" * 75)

    MODEL_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_bundle = {
        "model": model,
        "feature_cols": feature_cols,
        "categorical_cols": categorical_cols,
        "numeric_cols": numeric_cols,
        "positive_weight": 2,
        "model_params": MODEL_PARAMS,
    }

    joblib.dump(
        model_bundle,
        MODEL_FILE,
    )

    print(
        f"Model saved:\n"
        f"{MODEL_FILE}"
    )


# =====================================================================
# GENERATE ML SCORES
# =====================================================================

def generate_ml_scores(
    model,
    X_challenge,
    challenge,
):

    print()
    print("=" * 75)
    print("7. GENERATING CHALLENGE ML SCORES")
    print("=" * 75)

    probabilities = (
        model
        .predict_proba(
            X_challenge
        )[:, 1]
    )

    scored = challenge[
        [
            GATEWAY_COL,
            WEEK_COL,
        ]
    ].copy()

    scored[
        "ml_probability"
    ] = probabilities

    print(
        f"Probability minimum: "
        f"{probabilities.min():.8f}"
    )

    print(
        f"Probability maximum: "
        f"{probabilities.max():.8f}"
    )

    print(
        f"Probability mean: "
        f"{probabilities.mean():.8f}"
    )

    return scored


# =====================================================================
# ADD BASELINE
# =====================================================================

def add_baseline(
    scored,
    baseline,
):

    print()
    print("=" * 75)
    print("8. ADDING 3-SIGMA BASELINE")
    print("=" * 75)

    baseline_use = baseline[
        [
            GATEWAY_COL,
            WEEK_COL,
            "baseline_flagged_hours",
        ]
    ].copy()

    baseline_use = baseline_use.rename(
        columns={
            "baseline_flagged_hours":
                "baseline_score"
        }
    )

    scored = scored.merge(
        baseline_use,
        on=[
            GATEWAY_COL,
            WEEK_COL,
        ],
        how="left",
        validate="one_to_one",
    )

    missing = (
        scored["baseline_score"]
        .isna()
        .sum()
    )

    print(
        f"Missing baseline scores: "
        f"{missing}"
    )

    if missing:
        raise ValueError(
            f"{missing} challenge rows have no "
            "3-sigma baseline score."
        )

    return scored


# =====================================================================
# BUILD HYBRID RANKING
# =====================================================================

def build_hybrid_ranking(
    scored,
):

    print()
    print("=" * 75)
    print("9. BUILDING 75/25 HYBRID RANKING")
    print("=" * 75)

    final_rows = []

    weeks = sorted(
        scored[WEEK_COL].unique()
    )

    for week in weeks:

        week_df = scored[
            scored[WEEK_COL] == week
        ].copy()

        if len(week_df) != 332:
            raise ValueError(
                f"{week}: expected 332 gateways, "
                f"got {len(week_df)}."
            )

        # ---------------------------------------------------------
        # ML percentile
        # ---------------------------------------------------------

        week_df[
            "ml_percentile"
        ] = (
            week_df[
                "ml_probability"
            ]
            .rank(
                method="average",
                pct=True,
            )
        )

        # ---------------------------------------------------------
        # Baseline percentile
        # ---------------------------------------------------------

        week_df[
            "baseline_percentile"
        ] = (
            week_df[
                "baseline_score"
            ]
            .rank(
                method="average",
                pct=True,
            )
        )

        # ---------------------------------------------------------
        # Frozen 75/25 hybrid.
        # ---------------------------------------------------------

        week_df[
            "hybrid_score"
        ] = (
            ML_WEIGHT
            *
            week_df[
                "ml_percentile"
            ]
            +
            BASELINE_WEIGHT
            *
            week_df[
                "baseline_percentile"
            ]
        )

        # ---------------------------------------------------------
        # Frozen deterministic sorting.
        # ---------------------------------------------------------

        week_df = (
            week_df
            .sort_values(
                [
                    "hybrid_score",
                    "ml_probability",
                    "baseline_score",
                    GATEWAY_COL,
                ],
                ascending=[
                    False,
                    False,
                    False,
                    True,
                ],
            )
            .reset_index(drop=True)
        )

        # ---------------------------------------------------------
        # Rank
        # ---------------------------------------------------------

        week_df["rank"] = (
            np.arange(
                len(week_df)
            )
            +
            1
        )

        # ---------------------------------------------------------
        # Select top 15.
        # ---------------------------------------------------------

        selected = week_df.head(
            TOP_K
        ).copy()

        if len(selected) != TOP_K:
            raise ValueError(
                f"{week}: expected {TOP_K} selections."
            )

        print(
            f"{week.strftime('%Y-%m-%d')}: "
            f"selected {len(selected)} gateways"
        )

        final_rows.append(
            selected[
                [
                    WEEK_COL,
                    "rank",
                    GATEWAY_COL,
                    "hybrid_score",
                    "ml_probability",
                    "baseline_score",
                    "ml_percentile",
                    "baseline_percentile",
                ]
            ]
        )

    final_ranking = pd.concat(
        final_rows,
        ignore_index=True,
    )

    return final_ranking


# =====================================================================
# VALIDATE RANKING
# =====================================================================

def validate_ranking(
    ranking,
):

    print()
    print("=" * 75)
    print("10. VALIDATING FINAL RANKING")
    print("=" * 75)

    # -------------------------------------------------------------
    # 120 rows
    # -------------------------------------------------------------

    if len(ranking) != 120:
        raise ValueError(
            f"Expected 120 rows, got {len(ranking)}."
        )

    print(
        "PASS 1: Exactly 120 rows"
    )

    # -------------------------------------------------------------
    # 8 weeks
    # -------------------------------------------------------------

    if ranking[WEEK_COL].nunique() != 8:
        raise ValueError(
            "Expected exactly 8 weeks."
        )

    print(
        "PASS 2: Exactly 8 weeks"
    )

    # -------------------------------------------------------------
    # 15/week
    # -------------------------------------------------------------

    counts = (
        ranking
        .groupby(WEEK_COL)
        .size()
    )

    if not counts.eq(15).all():
        raise ValueError(
            "Every week must contain exactly 15 selections."
        )

    print(
        "PASS 3: Exactly 15 selections per week"
    )

    # -------------------------------------------------------------
    # Rank 1-15
    # -------------------------------------------------------------

    if not ranking[
        "rank"
    ].between(
        1,
        15,
    ).all():

        raise ValueError(
            "Ranks outside 1-15 detected."
        )

    print(
        "PASS 4: Ranks are within 1-15"
    )

    # -------------------------------------------------------------
    # Every week has exactly 1-15.
    # -------------------------------------------------------------

    for week, group in ranking.groupby(
        WEEK_COL
    ):

        ranks = sorted(
            group["rank"].tolist()
        )

        if ranks != list(
            range(1, 16)
        ):
            raise ValueError(
                f"Invalid ranks for {week}: {ranks}"
            )

    print(
        "PASS 5: Every week contains ranks 1-15"
    )

    # -------------------------------------------------------------
    # No duplicate gateway/week.
    # -------------------------------------------------------------

    duplicates = ranking.duplicated(
        [
            WEEK_COL,
            GATEWAY_COL,
        ]
    ).sum()

    if duplicates:
        raise ValueError(
            f"Found {duplicates} duplicate gateway/week rows."
        )

    print(
        "PASS 6: No duplicate gateway/week"
    )

    # -------------------------------------------------------------
    # Numeric hybrid scores.
    # -------------------------------------------------------------

    if ranking["hybrid_score"].isna().any():
        raise ValueError(
            "Hybrid scores contain missing values."
        )

    print(
        "PASS 7: All hybrid scores are valid"
    )


# =====================================================================
# SAVE RANKING
# =====================================================================

def save_ranking(
    ranking,
):

    print()
    print("=" * 75)
    print("11. SAVING FINAL HYBRID RANKING")
    print("=" * 75)

    RANKING_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    ranking.to_csv(
        RANKING_FILE,
        index=False,
    )

    print(
        f"Saved:\n"
        f"{RANKING_FILE}"
    )


# =====================================================================
# GENERATE REASON
# =====================================================================

def generate_reason(
    row,
):

    ml_percentile = float(
        row["ml_percentile"]
    )

    baseline_percentile = float(
        row["baseline_percentile"]
    )

    baseline_score = float(
        row["baseline_score"]
    )

    signals = []

    # -------------------------------------------------------------
    # ML signal
    # -------------------------------------------------------------

    if ml_percentile >= 0.90:

        signals.append(
            "high ML fault-risk signal"
        )

    elif ml_percentile >= 0.75:

        signals.append(
            "elevated ML fault-risk signal"
        )

    else:

        signals.append(
            "ML fault-risk signal"
        )

    # -------------------------------------------------------------
    # Baseline signal
    # -------------------------------------------------------------

    if baseline_percentile >= 0.90:

        signals.append(
            "strong recent 3-sigma anomaly signal"
        )

    elif baseline_percentile >= 0.75:

        signals.append(
            "elevated recent 3-sigma anomaly signal"
        )

    elif baseline_score > 0:

        signals.append(
            "recent 3-sigma anomaly signal"
        )

    # -------------------------------------------------------------
    # Final reason
    # -------------------------------------------------------------

    if len(signals) == 2:

        reason = (
            "Selected due to "
            + signals[0]
            + " and "
            + signals[1]
            + "."
        )

    else:

        reason = (
            "Selected due to "
            + signals[0]
            + "."
        )

    return reason[:300]


# =====================================================================
# BUILD SUBMISSION
# =====================================================================

def build_submission(
    ranking,
):

    print()
    print("=" * 75)
    print("12. BUILDING predictions.csv")
    print("=" * 75)

    df = (
        ranking
        .sort_values(
            [
                WEEK_COL,
                "rank",
            ]
        )
        .reset_index(drop=True)
        .copy()
    )

    df["reason"] = df.apply(
        generate_reason,
        axis=1,
    )

    submission = df[
        [
            WEEK_COL,
            "rank",
            GATEWAY_COL,
            "hybrid_score",
            "reason",
        ]
    ].copy()

    submission = submission.rename(
        columns={
            "hybrid_score": "score",
        }
    )

    # -------------------------------------------------------------
    # Formatting
    # -------------------------------------------------------------

    submission[WEEK_COL] = (
        submission[WEEK_COL]
        .dt.strftime("%Y-%m-%d")
    )

    submission["rank"] = (
        submission["rank"]
        .astype(int)
    )

    submission[GATEWAY_COL] = (
        submission[GATEWAY_COL]
        .astype(str)
    )

    submission["score"] = pd.to_numeric(
        submission["score"],
        errors="raise",
    )

    submission["reason"] = (
        submission["reason"]
        .astype(str)
    )

    return submission


# =====================================================================
# FINAL SUBMISSION AUDIT
# =====================================================================

def audit_submission(
    submission,
):

    print()
    print("=" * 75)
    print("13. FINAL SUBMISSION AUDIT")
    print("=" * 75)

    # -------------------------------------------------------------
    # Exactly 120 rows
    # -------------------------------------------------------------

    assert len(submission) == 120

    print(
        "PASS 1: Exactly 120 rows"
    )

    # -------------------------------------------------------------
    # Exactly 8 weeks
    # -------------------------------------------------------------

    assert (
        submission[WEEK_COL]
        .nunique()
        ==
        8
    )

    print(
        "PASS 2: Exactly 8 challenge weeks"
    )

    # -------------------------------------------------------------
    # Exactly 15/week
    # -------------------------------------------------------------

    weekly_counts = (
        submission
        .groupby(WEEK_COL)
        .size()
    )

    assert weekly_counts.eq(15).all()

    print(
        "PASS 3: Exactly 15 selections per week"
    )

    # -------------------------------------------------------------
    # Ranks 1-15
    # -------------------------------------------------------------

    assert submission[
        "rank"
    ].between(
        1,
        15,
    ).all()

    print(
        "PASS 4: Ranks are within 1-15"
    )

    # -------------------------------------------------------------
    # Every week ranks 1-15.
    # -------------------------------------------------------------

    for week, group in submission.groupby(
        WEEK_COL
    ):

        ranks = sorted(
            group["rank"].tolist()
        )

        assert ranks == list(
            range(1, 16)
        ), (
            f"Invalid ranks for {week}: "
            f"{ranks}"
        )

    print(
        "PASS 5: Every week contains ranks 1-15"
    )

    # -------------------------------------------------------------
    # No duplicate gateway/week.
    # -------------------------------------------------------------

    duplicates = submission.duplicated(
        [
            WEEK_COL,
            GATEWAY_COL,
        ]
    ).sum()

    assert duplicates == 0

    print(
        "PASS 6: No duplicate gateway/week"
    )

    # -------------------------------------------------------------
    # Valid score.
    # -------------------------------------------------------------

    assert submission[
        "score"
    ].notna().all()

    assert np.isfinite(
        submission["score"]
        .to_numpy()
    ).all()

    print(
        "PASS 7: All scores are numeric"
    )

    # -------------------------------------------------------------
    # Reasons.
    # -------------------------------------------------------------

    assert submission[
        "reason"
    ].str.strip().ne("").all()

    print(
        "PASS 8: No empty reasons"
    )

    # -------------------------------------------------------------
    # Reason length.
    # -------------------------------------------------------------

    max_reason_length = (
        submission[
            "reason"
        ]
        .str.len()
        .max()
    )

    assert max_reason_length <= 300

    print(
        f"PASS 9: Maximum reason length = "
        f"{max_reason_length}"
    )

    # -------------------------------------------------------------
    # Required columns.
    # -------------------------------------------------------------

    expected_columns = [
        "week_start",
        "rank",
        "gateway_id",
        "score",
        "reason",
    ]

    assert list(
        submission.columns
    ) == expected_columns

    print(
        "PASS 10: Submission columns are correct"
    )

    return max_reason_length


# =====================================================================
# SAVE SUBMISSION
# =====================================================================

def save_submission(
    submission,
):

    submission.to_csv(
        SUBMISSION_FILE,
        index=False,
    )

    print()
    print(
        f"Saved final submission:\n"
        f"{SUBMISSION_FILE}"
    )


# =====================================================================
# MAIN
# =====================================================================

def main():

    print()
    print("=" * 75)
    print("STEP 04 - FINAL SUBMISSION GENERATION")
    print("=" * 75)

    # -------------------------------------------------------------
    # 1. Load
    # -------------------------------------------------------------

    historical, challenge, baseline = (
        load_data()
    )

    # -------------------------------------------------------------
    # 2. Validate
    # -------------------------------------------------------------

    validate_inputs(
        historical,
        challenge,
        baseline,
    )

    # -------------------------------------------------------------
    # 3. Feature schema
    # -------------------------------------------------------------

    (
        feature_cols,
        categorical_cols,
        numeric_cols,
    ) = get_feature_schema(
        historical,
        challenge,
    )

    # -------------------------------------------------------------
    # 4. Prepare features
    # -------------------------------------------------------------

    (
        X_train,
        X_challenge,
        y_train,
    ) = prepare_features(
        historical,
        challenge,
        feature_cols,
        categorical_cols,
    )

    # -------------------------------------------------------------
    # 5. Train final HGB
    # -------------------------------------------------------------

    model = train_model(
        X_train,
        y_train,
    )

    # -------------------------------------------------------------
    # 6. Save model
    # -------------------------------------------------------------

    save_model(
        model,
        feature_cols,
        categorical_cols,
        numeric_cols,
    )

    # -------------------------------------------------------------
    # 7. ML scores
    # -------------------------------------------------------------

    scored = generate_ml_scores(
        model,
        X_challenge,
        challenge,
    )

    # -------------------------------------------------------------
    # 8. Baseline
    # -------------------------------------------------------------

    scored = add_baseline(
        scored,
        baseline,
    )

    # -------------------------------------------------------------
    # 9. Hybrid ranking
    # -------------------------------------------------------------

    ranking = build_hybrid_ranking(
        scored,
    )

    # -------------------------------------------------------------
    # 10. Validate ranking
    # -------------------------------------------------------------

    validate_ranking(
        ranking,
    )

    # -------------------------------------------------------------
    # Save intermediate ranking
    # -------------------------------------------------------------

    save_ranking(
        ranking,
    )

    # -------------------------------------------------------------
    # 11. Build submission
    # -------------------------------------------------------------

    submission = build_submission(
        ranking,
    )

    # -------------------------------------------------------------
    # 12. Audit
    # -------------------------------------------------------------

    max_reason_length = audit_submission(
        submission,
    )

    # -------------------------------------------------------------
    # 13. Save predictions.csv
    # -------------------------------------------------------------

    save_submission(
        submission,
    )

    # -------------------------------------------------------------
    # Display first week
    # -------------------------------------------------------------

    print()
    print("=" * 75)
    print("FINAL SUBMISSION SAMPLE")
    print("=" * 75)

    print(
        submission
        .head(15)
        .to_string(index=False)
    )

    # -------------------------------------------------------------
    # Final summary
    # -------------------------------------------------------------

    print()
    print("=" * 75)
    print("STEP 04 COMPLETE")
    print("=" * 75)

    print(
        f"\nFinal submission:"
    )

    print(
        f"  Rows: {len(submission)}"
    )

    print(
        f"  Weeks: "
        f"{submission[WEEK_COL].nunique()}"
    )

    print(
        f"  Selections/week: 15"
    )

    print(
        f"  Maximum reason length: "
        f"{max_reason_length}"
    )

    print(
        f"\nRanking:"
    )

    print(
        f"  {RANKING_FILE}"
    )

    print(
        f"\nSubmission:"
    )

    print(
        f"  {SUBMISSION_FILE}"
    )

    print(
        "\nPASS: predictions.csv is ready for final audit."
    )


# =====================================================================
# ENTRY POINT
# =====================================================================

if __name__ == "__main__":
    main()