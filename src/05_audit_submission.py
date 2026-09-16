"""
STEP 05 - FINAL SUBMISSION AUDIT

Final validation of:
    predictions.csv
    outputs/final_hybrid_weight2_ranking.csv
    gateway_master.csv
    historical/challenge feature datasets

This script does not modify the submission.
It only audits the final artifacts.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================================
# PATHS
# ============================================================================

ROOT = Path(__file__).resolve().parents[1]

PREDICTIONS_PATH = ROOT / "predictions.csv"
RANKING_PATH = ROOT / "outputs" / "final_hybrid_weight2_ranking.csv"
MASTER_PATH = ROOT / "data" / "gateway_master.csv"

HISTORICAL_PATH = ROOT / "outputs" / "gateway_week_features_reduced.csv"
CHALLENGE_PATH = ROOT / "outputs" / "challenge_gateway_week_features.csv"


EXPECTED_WEEKS = pd.to_datetime([
    "2026-02-02",
    "2026-02-09",
    "2026-02-16",
    "2026-02-23",
    "2026-03-02",
    "2026-03-09",
    "2026-03-16",
    "2026-03-23",
])

TOP_K = 15
EXPECTED_ROWS = len(EXPECTED_WEEKS) * TOP_K


# ============================================================================
# HELPERS
# ============================================================================

def normalize_gateway_id(value):
    """
    Normalize gateway IDs consistently with the feature pipeline.
    """
    if pd.isna(value):
        return None

    return (
        str(value)
        .strip()
        .upper()
        .replace(":", "")
        .replace("-", "")
        .replace(" ", "")
    )


def check(condition, message):
    """
    Print PASS/FAIL and raise on failure.
    """
    if condition:
        print(f"[PASS] {message}")
    else:
        print(f"[FAIL] {message}")
        raise AssertionError(message)


# ============================================================================
# LOAD FILES
# ============================================================================

def load_files():

    print("\n" + "=" * 75)
    print("1. LOADING FINAL ARTIFACTS")
    print("=" * 75)

    check(
        PREDICTIONS_PATH.exists(),
        f"predictions.csv exists: {PREDICTIONS_PATH}"
    )

    check(
        RANKING_PATH.exists(),
        f"final ranking exists: {RANKING_PATH}"
    )

    check(
        MASTER_PATH.exists(),
        f"gateway master exists: {MASTER_PATH}"
    )

    predictions = pd.read_csv(PREDICTIONS_PATH)
    ranking = pd.read_csv(RANKING_PATH)
    master = pd.read_csv(MASTER_PATH, encoding="latin1")

    print(f"Predictions shape : {predictions.shape}")
    print(f"Ranking shape     : {ranking.shape}")
    print(f"Master shape      : {master.shape}")

    return predictions, ranking, master


# ============================================================================
# PREDICTION STRUCTURE
# ============================================================================

def audit_predictions(predictions):

    print("\n" + "=" * 75)
    print("2. PREDICTIONS STRUCTURE")
    print("=" * 75)

    expected_columns = [
        "week_start",
        "rank",
        "gateway_id",
        "score",
        "reason",
    ]

    check(
        list(predictions.columns) == expected_columns,
        f"Submission columns are exactly {expected_columns}"
    )

    check(
        len(predictions) == EXPECTED_ROWS,
        f"Exactly {EXPECTED_ROWS} rows"
    )

    predictions["week_start"] = pd.to_datetime(
        predictions["week_start"],
        errors="coerce"
    )

    check(
        predictions["week_start"].notna().all(),
        "All week_start values are valid dates"
    )

    actual_weeks = sorted(predictions["week_start"].unique())

    check(
        actual_weeks == list(EXPECTED_WEEKS),
        "Exactly the eight expected challenge weeks"
    )

    check(
        predictions["rank"].notna().all(),
        "No missing ranks"
    )

    check(
        pd.api.types.is_numeric_dtype(predictions["rank"]),
        "Rank column is numeric"
    )

    check(
        predictions["rank"].between(1, TOP_K).all(),
        "All ranks are between 1 and 15"
    )

    for week in EXPECTED_WEEKS:

        week_rows = predictions[
            predictions["week_start"] == week
        ]

        check(
            len(week_rows) == TOP_K,
            f"{week.date()}: exactly 15 selections"
        )

        check(
            sorted(week_rows["rank"].tolist())
            == list(range(1, TOP_K + 1)),
            f"{week.date()}: ranks are exactly 1-15"
        )

    duplicates = predictions.duplicated(
        subset=["week_start", "gateway_id"]
    )

    check(
        not duplicates.any(),
        "No duplicate gateway/week combinations"
    )


# ============================================================================
# GATEWAY ID VALIDATION
# ============================================================================

def audit_gateway_ids(predictions, master):

    print("\n" + "=" * 75)
    print("3. GATEWAY ID VALIDATION")
    print("=" * 75)

    master_id_column = None

    for candidate in ["gateway_id", "Gateway ID", "gatewayId"]:
        if candidate in master.columns:
            master_id_column = candidate
            break

    check(
        master_id_column is not None,
        "Gateway master contains a gateway ID column"
    )

    submission_ids = set(
        predictions["gateway_id"]
        .map(normalize_gateway_id)
    )

    master_ids = set(
        master[master_id_column]
        .map(normalize_gateway_id)
    )

    invalid_ids = submission_ids - master_ids

    check(
        len(invalid_ids) == 0,
        "Every submitted gateway exists in gateway_master.csv"
    )

    print(f"Master gateways           : {len(master_ids)}")
    print(f"Unique submitted gateways : {len(submission_ids)}")


# ============================================================================
# SCORE VALIDATION
# ============================================================================

def audit_scores(predictions):

    print("\n" + "=" * 75)
    print("4. SCORE VALIDATION")
    print("=" * 75)

    scores = pd.to_numeric(
        predictions["score"],
        errors="coerce"
    )

    check(
        scores.notna().all(),
        "All scores are numeric"
    )

    check(
        np.isfinite(scores).all(),
        "All scores are finite"
    )

    check(
        ((scores >= 0) & (scores <= 1)).all(),
        "All scores are between 0 and 1"
    )

    print(f"Minimum score : {scores.min():.6f}")
    print(f"Maximum score : {scores.max():.6f}")
    print(f"Mean score    : {scores.mean():.6f}")


# ============================================================================
# REASON VALIDATION
# ============================================================================

def audit_reasons(predictions):

    print("\n" + "=" * 75)
    print("5. REASON VALIDATION")
    print("=" * 75)

    reasons = predictions["reason"]

    check(
        reasons.notna().all(),
        "No missing reasons"
    )

    check(
        reasons.astype(str).str.strip().ne("").all(),
        "No empty reasons"
    )

    reason_lengths = reasons.astype(str).str.len()

    check(
        (reason_lengths <= 300).all(),
        "Every reason is <= 300 characters"
    )

    print(f"Maximum reason length : {reason_lengths.max()}")
    print(f"Average reason length : {reason_lengths.mean():.2f}")


# ============================================================================
# RANKING FILE VALIDATION
# ============================================================================

def audit_ranking(ranking):

    print("\n" + "=" * 75)
    print("6. INTERNAL RANKING VALIDATION")
    print("=" * 75)

    required_columns = [
        "week_start",
        "rank",
        "gateway_id",
        "hybrid_score",
        "ml_probability",
        "baseline_score",
        "ml_percentile",
        "baseline_percentile",
    ]

    check(
        list(ranking.columns) == required_columns,
        "Internal ranking columns are correct"
    )

    ranking["week_start"] = (
        pd.to_datetime(
            ranking["week_start"],
            errors="coerce",
            utc=True
        )
        .dt.tz_localize(None)
        .dt.normalize()
    )

    expected_weeks = EXPECTED_WEEKS.normalize()

    check(
        len(ranking) == EXPECTED_ROWS,
        f"Internal ranking contains exactly {EXPECTED_ROWS} rows"
    )

    check(
        ranking["week_start"].notna().all(),
        "All internal ranking week_start values are valid dates"
    )

    check(
        ranking["week_start"].isin(expected_weeks).all(),
        "Internal ranking contains only challenge weeks"
    )

    check(
        ranking["rank"].between(1, TOP_K).all(),
        "Internal ranking ranks are between 1 and 15"
    )

    check(
        not ranking.duplicated(
            subset=["week_start", "gateway_id"]
        ).any(),
        "Internal ranking has no duplicate gateway/week"
    )

    numeric_columns = [
        "hybrid_score",
        "ml_probability",
        "baseline_score",
        "ml_percentile",
        "baseline_percentile",
    ]

    for column in numeric_columns:

        values = pd.to_numeric(
            ranking[column],
            errors="coerce"
        )

        check(
            values.notna().all(),
            f"{column} contains only numeric values"
        )

        check(
            np.isfinite(values).all(),
            f"{column} contains no NaN/inf values"
        )


# ============================================================================
# CONSISTENCY BETWEEN RANKING AND SUBMISSION
# ============================================================================

def audit_consistency(predictions, ranking):

    print("\n" + "=" * 75)
    print("7. SUBMISSION / RANKING CONSISTENCY")
    print("=" * 75)

    pred = predictions.copy()
    rank = ranking.copy()

    pred["week_start"] = pd.to_datetime(pred["week_start"])
    rank["week_start"] = pd.to_datetime(rank["week_start"])

    pred = pred.sort_values(
        ["week_start", "rank"]
    ).reset_index(drop=True)

    rank = rank.sort_values(
        ["week_start", "rank"]
    ).reset_index(drop=True)

    check(
        pred["week_start"].equals(rank["week_start"]),
        "Submission weeks match internal ranking"
    )

    check(
        pred["rank"].equals(rank["rank"]),
        "Submission ranks match internal ranking"
    )

    check(
        pred["gateway_id"].astype(str).equals(
            rank["gateway_id"].astype(str)
        ),
        "Submission gateway IDs match internal ranking"
    )

    check(
        np.allclose(
            pred["score"].astype(float),
            rank["hybrid_score"].astype(float),
            rtol=1e-10,
            atol=1e-10,
        ),
        "Submission scores match hybrid ranking scores"
    )


# ============================================================================
# FEATURE DATASET SANITY CHECK
# ============================================================================

def audit_feature_datasets():

    print("\n" + "=" * 75)
    print("8. FEATURE DATASET SANITY CHECK")
    print("=" * 75)

    check(
        CHALLENGE_PATH.exists(),
        "Challenge feature dataset exists"
    )

    challenge = pd.read_csv(CHALLENGE_PATH)

    print(f"Challenge rows : {len(challenge)}")

    check(
        len(challenge) == 2656,
        "Challenge dataset contains 2,656 gateway-week rows"
    )

    check(
        challenge["gateway_id"].nunique() == 332,
        "Challenge dataset contains 332 gateways"
    )

    check(
        challenge["week_start"].nunique() == 8,
        "Challenge dataset contains 8 weeks"
    )

    challenge["week_start"] = (
        pd.to_datetime(
            challenge["week_start"],
            errors="coerce",
            utc=True
        )
        .dt.tz_localize(None)
        .dt.normalize()
    )

    expected_weeks = EXPECTED_WEEKS.normalize()

    check(
        challenge["week_start"].notna().all(),
        "All challenge feature week_start values are valid dates"
    )

    check(
        sorted(challenge["week_start"].unique())
        == list(expected_weeks),
        "Challenge feature dates are correct"
    )

    duplicate_count = challenge.duplicated(
        subset=["gateway_id", "week_start"]
    ).sum()

    check(
        duplicate_count == 0,
        "Challenge features have no duplicate gateway/week rows"
    )


# ============================================================================
# FINAL SUMMARY
# ============================================================================

def print_summary(predictions):

    print("\n" + "=" * 75)
    print("9. FINAL SUBMISSION SUMMARY")
    print("=" * 75)

    print(f"Rows                 : {len(predictions)}")
    print(f"Weeks                : {predictions['week_start'].nunique()}")
    print(f"Selections per week  : {TOP_K}")
    print(
        f"Unique gateways      : "
        f"{predictions['gateway_id'].nunique()}"
    )

    scores = predictions["score"].astype(float)

    print(f"Score minimum        : {scores.min():.6f}")
    print(f"Score maximum        : {scores.max():.6f}")

    print("\nSelections by week:")

    counts = (
        predictions
        .groupby("week_start")
        .size()
    )

    for week, count in counts.items():
        print(f"  {week.date()} : {count}")

    print("\n" + "=" * 75)
    print("FINAL RESULT")
    print("=" * 75)

    print("[PASS] FINAL SUBMISSION AUDIT COMPLETE")
    print()
    print("predictions.csv is structurally valid and ready for")
    print("the challenge submission package.")


# ============================================================================
# MAIN
# ============================================================================

def main():

    print("\n" + "=" * 75)
    print("STEP 05 - FINAL SUBMISSION AUDIT")
    print("=" * 75)

    predictions, ranking, master = load_files()

    audit_predictions(predictions)
    audit_gateway_ids(predictions, master)
    audit_scores(predictions)
    audit_reasons(predictions)
    audit_ranking(ranking)
    audit_consistency(predictions, ranking)
    audit_feature_datasets()

    print_summary(predictions)


if __name__ == "__main__":
    main()