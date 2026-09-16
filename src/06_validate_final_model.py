"""
STEP 06 - FINAL MODEL VALIDATION

Purpose
-------
Evaluate the final ML solution using historical forward-in-time validation.

Methods compared:
    1. Supplied 3-sigma telemetry baseline
    2. HGB positive-class weight = 2
    3. 75% HGB + 25% 3-sigma hybrid

IMPORTANT
---------
The official 3-sigma baseline is reconstructed directly from telemetry.

For each validation Monday:

    - trailing 28 days strictly before Monday
    - per-gateway mean/std for:
        offline_duration_sec
        disconnection_cnt
        reboot_cnt
    - trailing 7 days
    - flag hours where any metric exceeds:
        mean + 3 * std

This follows the supplied baseline implementation. 

The validation is strictly forward in time:
training data ends before each validation week.

The challenge weeks are NOT used as validation labels.

Outputs:
    outputs/final_model_validation.csv
"""

from operator import index
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier


# ============================================================================
# PATHS
# ============================================================================

ROOT = Path(__file__).resolve().parents[1]

DATASET_PATH = (
    ROOT
    / "outputs"
    / "gateway_week_features_reduced.csv"
)

TELEMETRY_PATH = (
    ROOT
    / "data"
    / "telemetry"
)

OUTPUT_PATH = (
    ROOT
    / "outputs"
    / "final_model_validation.csv"
)


# ============================================================================
# CONFIGURATION
# ============================================================================

TARGET = "fault_next_7d"

GATEWAY_COL = "gateway_id"

WEEK_COL = "week_start"

TOP_K = 15

BASELINE_DAYS = 28

RECENT_DAYS = 7

SIGMA = 3.0

ML_WEIGHT = 0.75

BASELINE_WEIGHT = 0.25

VISIT_COST = 380

MISSED_FAULT_COST = 600


# ============================================================================
# MODEL
# ============================================================================

CATEGORICAL_COLUMNS = [
    "tenant",
    "site_type",
    "region",
    "hw_model",
    "antenna_type",
    "fw_version",
]

MODEL_PARAMS = {
    "max_iter": 300,
    "learning_rate": 0.05,
    "max_leaf_nodes": 15,
    "l2_regularization": 2.0,
    "random_state": 42,
}

POSITIVE_WEIGHT = 2.0


# ============================================================================
# BASELINE METRICS
# ============================================================================

BASELINE_METRICS = [
    "offline_duration_sec",
    "disconnection_cnt",
    "reboot_cnt",
]


# ============================================================================
# HELPERS
# ============================================================================

def normalize_gateway_id(value):

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


def normalize_datetime(series):

    return (
        pd.to_datetime(
            series,
            errors="coerce",
            utc=True,
        )
        .dt.tz_localize(None)
        .dt.normalize()
    )


def calculate_cost(selected_gateways, actual_faults):

    selected = set(selected_gateways)

    faults = set(actual_faults)

    caught = len(selected & faults)

    missed = len(faults - selected)

    cost = (
        len(selected) * VISIT_COST
        + missed * MISSED_FAULT_COST
    )

    return cost, caught, missed


# ============================================================================
# LOAD HISTORICAL DATASET
# ============================================================================

def load_dataset():

    print("\n" + "=" * 75)
    print("1. LOADING HISTORICAL DATASET")
    print("=" * 75)

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Missing dataset: {DATASET_PATH}"
        )

    dataset = pd.read_csv(
        DATASET_PATH
    )

    dataset[WEEK_COL] = normalize_datetime(
        dataset[WEEK_COL]
    )

    dataset[GATEWAY_COL] = (
        dataset[GATEWAY_COL]
        .map(normalize_gateway_id)
    )

    dataset[TARGET] = (
        pd.to_numeric(
            dataset[TARGET],
            errors="coerce",
        )
        .fillna(0)
        .astype(int)
    )

    print(
        f"Dataset shape : {dataset.shape}"
    )

    print(
        f"Historical gateways : "
        f"{dataset[GATEWAY_COL].nunique()}"
    )

    print(
        f"Historical weeks : "
        f"{dataset[WEEK_COL].nunique()}"
    )

    print(
        f"Historical positives : "
        f"{dataset[TARGET].sum()}"
    )

    return dataset


# ============================================================================
# LOAD TELEMETRY
# ============================================================================

def load_telemetry():

    print("\n" + "=" * 75)
    print("2. LOADING TELEMETRY FOR 3-SIGMA BASELINE")
    print("=" * 75)

    if not TELEMETRY_PATH.exists():
        raise FileNotFoundError(
            f"Missing telemetry directory: {TELEMETRY_PATH}"
        )

    columns = [
        "gateway_id",
        "ts_utc",
        *BASELINE_METRICS,
    ]

    telemetry = pd.read_parquet(
        TELEMETRY_PATH,
        columns=columns,
    )

    telemetry[GATEWAY_COL] = (
        telemetry[GATEWAY_COL]
        .map(normalize_gateway_id)
    )

    telemetry["ts"] = pd.to_datetime(
        telemetry["ts_utc"],
        errors="coerce",
        utc=True,
    )

    telemetry = telemetry.drop(
        columns=["ts_utc"]
    )

    for column in BASELINE_METRICS:

        telemetry[column] = pd.to_numeric(
            telemetry[column],
            errors="coerce",
        )

    telemetry = telemetry.dropna(
        subset=[
            GATEWAY_COL,
            "ts",
        ]
    )

    telemetry = telemetry.sort_values(
        [
            GATEWAY_COL,
            "ts",
        ]
    )

    print(
        f"Telemetry rows : {len(telemetry):,}"
    )

    print(
        f"Telemetry gateways : "
        f"{telemetry[GATEWAY_COL].nunique()}"
    )

    print(
        f"Telemetry range : "
        f"{telemetry['ts'].min()} -> "
        f"{telemetry['ts'].max()}"
    )

    return telemetry


# ============================================================================
# OFFICIAL 3-SIGMA BASELINE
# ============================================================================

def calculate_3sigma_scores(
    telemetry,
    monday,
):

    """
    Reproduce the supplied baseline logic.

    For a Monday:

        baseline window:
            [Monday - 28 days, Monday)

        recent window:
            [Monday - 7 days, Monday)

    Per gateway:
        mean/std over full 28-day window.

    A recent hour is flagged if:
        value - mean > 3 * std

    If std == 0:
        the metric cannot create a breach.
    """

    end = pd.Timestamp(
        monday,
        tz="UTC",
    )

    window_start = (
        end
        - pd.Timedelta(
            days=BASELINE_DAYS
        )
    )

    recent_start = (
        end
        - pd.Timedelta(
            days=RECENT_DAYS
        )
    )

    window = telemetry[
        (
            telemetry["ts"] >= window_start
        )
        &
        (
            telemetry["ts"] < end
        )
    ].copy()

    if window.empty:

        raise RuntimeError(
            f"No telemetry available before {monday}"
        )

    stats = (
        window
        .groupby(GATEWAY_COL)[BASELINE_METRICS]
        .agg(
            [
                "mean",
                "std",
            ]
        )
    )

    recent = window[
        window["ts"] >= recent_start
    ].copy()

    flags = pd.Series(
        0,
        index=recent.index,
        dtype=int,
    )

    worst_metric = pd.Series(
        "",
        index=recent.index,
        dtype=object,
    )

    for metric in BASELINE_METRICS:

        means = recent[
            GATEWAY_COL
        ].map(
            stats[
                (
                    metric,
                    "mean",
                )
            ]
        )

        stds = recent[
            GATEWAY_COL
        ].map(
            stats[
                (
                    metric,
                    "std",
                )
            ]
        )

        stds = stds.replace(
            0,
            np.nan,
        )

        exceeded = (
            recent[metric]
            - means
        ) > (
            SIGMA
            * stds
        )

        exceeded = exceeded.fillna(
            False
        )

        flags = (
            flags
            + exceeded.astype(int)
        )

        worst_metric = (
            worst_metric
            .where(
                ~exceeded
                | (
                    worst_metric != ""
                ),
                metric,
            )
        )

    recent["flagged"] = flags

    recent["worst_metric"] = (
        worst_metric
    )

    grouped = (
        recent
        .groupby(GATEWAY_COL)
        .agg(
            baseline_score=(
                "flagged",
                "sum",
            ),
            worst_metric=(
                "worst_metric",
                lambda values:
                    next(
                        (
                            value
                            for value in values
                            if value
                        ),
                        "",
                    ),
            ),
        )
        .reset_index()
    )

    return grouped


# ============================================================================
# FEATURE PREPARATION
# ============================================================================

def prepare_features(
    train_df,
    test_df,
):

    feature_columns = [
        column
        for column in train_df.columns
        if column not in [
            GATEWAY_COL,
            WEEK_COL,
            TARGET,
        ]
    ]

    train_x = (
        train_df[
            feature_columns
        ]
        .copy()
    )

    test_x = (
        test_df[
            feature_columns
        ]
        .copy()
    )

    for column in CATEGORICAL_COLUMNS:

        combined = pd.concat(
            [
                train_x[column],
                test_x[column],
            ],
            ignore_index=True,
        )

        combined = (
            combined
            .astype("category")
            .cat.codes
            .astype(float)
        )

        train_x[column] = (
            combined
            .iloc[:len(train_x)]
            .to_numpy()
        )

        test_x[column] = (
            combined
            .iloc[len(train_x):]
            .to_numpy()
        )

    numeric_columns = [
        column
        for column in feature_columns
        if column not in CATEGORICAL_COLUMNS
    ]

    for column in numeric_columns:

        train_x[column] = (
            pd.to_numeric(
                train_x[column],
                errors="coerce",
            )
        )

        test_x[column] = (
            pd.to_numeric(
                test_x[column],
                errors="coerce",
            )
        )

    return (
        train_x,
        test_x,
    )


# ============================================================================
# TRAIN HGB
# ============================================================================

def train_and_predict(
    train_df,
    test_df,
):

    train_x, test_x = (
        prepare_features(
            train_df,
            test_df,
        )
    )

    y_train = (
        train_df[TARGET]
        .astype(int)
    )

    sample_weights = np.where(
        y_train == 1,
        POSITIVE_WEIGHT,
        1.0,
    )

    model = (
        HistGradientBoostingClassifier(
            **MODEL_PARAMS
        )
    )

    model.fit(
        train_x,
        y_train,
        sample_weight=sample_weights,
    )

    probabilities = (
        model
        .predict_proba(
            test_x
        )[:, 1]
    )

    return probabilities


# ============================================================================
# SELECT TOP 15
# ============================================================================

def select_top_15(
    dataframe,
    score_column,
    secondary_columns,
):

    selected_rows = []

    for week, week_df in (
        dataframe
        .groupby(
            WEEK_COL,
            sort=True,
        )
    ):

        week_df = (
            week_df
            .copy()
        )

        week_df["selection_percentile"] = (
            week_df[score_column]
            .rank(
                method="average",
                pct=True,
            )
        )

        sort_columns = [
            "selection_percentile",
            score_column,
            *secondary_columns,
            GATEWAY_COL,
        ]

        ascending = [
            False,
            False,
            *(
                [False]
                * len(secondary_columns)
            ),
            True,
        ]

        selected = (
            week_df
            .sort_values(
                sort_columns,
                ascending=ascending,
            )
            .head(TOP_K)
        )

        selected_rows.append(
            selected
        )

    if not selected_rows:

        return dataframe.iloc[0:0].copy()

    return pd.concat(
        selected_rows,
        ignore_index=True,
    )


# ============================================================================
# BUILD HYBRID
# ============================================================================

def build_hybrid(
    validation_df,
):

    result_rows = []

    for week, week_df in (
        validation_df
        .groupby(
            WEEK_COL,
            sort=True,
        )
    ):

        week_df = (
            week_df
            .copy()
        )

        week_df["ml_percentile"] = (
            week_df[
                "ml_probability"
            ]
            .rank(
                method="average",
                pct=True,
            )
        )

        week_df["baseline_percentile"] = (
            week_df[
                "baseline_score"
            ]
            .rank(
                method="average",
                pct=True,
            )
        )

        week_df["hybrid_score"] = (
            ML_WEIGHT
            * week_df[
                "ml_percentile"
            ]
            +
            BASELINE_WEIGHT
            * week_df[
                "baseline_percentile"
            ]
        )

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
        )

        result_rows.append(
            week_df.head(TOP_K)
        )

    return pd.concat(
        result_rows,
        ignore_index=True,
    )


# ============================================================================
# EVALUATE
# ============================================================================

def evaluate_method(
    full_validation,
    selected,
    method_name,
):

    rows = []

    for week, full_week in (
        full_validation
        .groupby(
            WEEK_COL,
            sort=True,
        )
    ):

        selected_week = (
            selected[
                selected[WEEK_COL] == week
            ]
        )

        selected_gateways = set(
            selected_week[
                GATEWAY_COL
            ]
        )

        actual_faults = set(
            full_week.loc[
                full_week[TARGET] == 1,
                GATEWAY_COL,
            ]
        )

        cost, caught, missed = (
            calculate_cost(
                selected_gateways,
                actual_faults,
            )
        )

        rows.append(
            {
                "week_start": week,
                "method": method_name,
                "selected": len(
                    selected_gateways
                ),
                "actual_faults": len(
                    actual_faults
                ),
                "caught_faults": caught,
                "missed_faults": missed,
                "cost": cost,
            }
        )

    return pd.DataFrame(rows)


# ============================================================================
# FORWARD VALIDATION
# ============================================================================

def run_forward_validation(
    dataset,
    telemetry,
):

    print("\n" + "=" * 75)
    print("3. FORWARD-IN-TIME VALIDATION")
    print("=" * 75)

    all_weeks = sorted(
        dataset[WEEK_COL]
        .dropna()
        .unique()
    )

    validation_start = pd.Timestamp("2025-11-10")
    validation_end = pd.Timestamp("2026-01-26")

    weeks = [
        week
        for week in all_weeks
        if validation_start <= week <= validation_end
    ]

    print(
        f"Validation window : "
        f"{validation_start.date()} -> "
        f"{validation_end.date()}"
    )

    print(
        f"Historical weeks available : "
        f"{len(weeks)}"
    )

    all_results = []

    # ------------------------------------------------------------
    # Each validation week is predicted using only earlier weeks.
    # ------------------------------------------------------------

    for index in range(len(weeks)):

        train_end = weeks[index - 1]

        validation_week = weeks[index]

        train_df = dataset[
            dataset[WEEK_COL] < validation_week
        ].copy()

        test_df = dataset[
            dataset[WEEK_COL] == validation_week
        ].copy()

        # --------------------------------------------------------
        # Skip weeks with no positive examples.
        # --------------------------------------------------------

        actual_fault_count = int(
            test_df[TARGET].sum()
        )

        if actual_fault_count == 0:

            continue

        if train_df.empty:
            print(
                f"\nSkipping {validation_week.date()}: "
                "no earlier training data."
            )
            continue

        train_end = train_df[WEEK_COL].max()

        print(
            f"\nFold {index + 1}: "
            f"train <= {train_end.date()} "
            f"-> validate "
            f"{validation_week.date()}"
        )

        # --------------------------------------------------------
        # ML
        # --------------------------------------------------------

        probabilities = train_and_predict(
            train_df,
            test_df,
        )

        validation = test_df[
            [
                GATEWAY_COL,
                WEEK_COL,
                TARGET,
            ]
        ].copy()

        validation[
            "ml_probability"
        ] = probabilities

        # --------------------------------------------------------
        # OFFICIAL 3-SIGMA
        # --------------------------------------------------------

        baseline_scores = (
            calculate_3sigma_scores(
                telemetry,
                validation_week,
            )
        )

        validation = validation.merge(
            baseline_scores[
                [
                    GATEWAY_COL,
                    "baseline_score",
                ]
            ],
            on=GATEWAY_COL,
            how="left",
            validate="one_to_one",
        )

        # Gateways without telemetry in the recent
        # window cannot have a flagged hour.
        validation["baseline_score"] = (
            pd.to_numeric(
                validation[
                    "baseline_score"
                ],
                errors="coerce",
            )
            .fillna(0)
        )

        # --------------------------------------------------------
        # HGB selection
        # --------------------------------------------------------

        ml_selected = select_top_15(
            validation,
            "ml_probability",
            [],
        )

        # --------------------------------------------------------
        # Baseline selection
        # --------------------------------------------------------

        baseline_selected = (
            select_top_15(
                validation,
                "baseline_score",
                [],
            )
        )

        # --------------------------------------------------------
        # Hybrid
        # --------------------------------------------------------

        hybrid_selected = build_hybrid(
            validation
        )

        # --------------------------------------------------------
        # Evaluate
        # --------------------------------------------------------

        ml_result = evaluate_method(
            validation,
            ml_selected,
            "HGB_weight2",
        )

        baseline_result = evaluate_method(
            validation,
            baseline_selected,
            "3sigma_baseline",
        )

        hybrid_result = evaluate_method(
            validation,
            hybrid_selected,
            "Hybrid_75ML_25Baseline",
        )

        fold_results = pd.concat(
            [
                baseline_result,
                ml_result,
                hybrid_result,
            ],
            ignore_index=True,
        )

        fold_results["fold"] = index

        all_results.append(
            fold_results
        )

        print(
            f"  Faults : {actual_fault_count}"
        )

        print(
            f"  Baseline caught : "
            f"{int(baseline_result['caught_faults'].iloc[0])}"
        )

        print(
            f"  HGB caught : "
            f"{int(ml_result['caught_faults'].iloc[0])}"
        )

        print(
            f"  Hybrid caught : "
            f"{int(hybrid_result['caught_faults'].iloc[0])}"
        )

    if not all_results:

        raise RuntimeError(
            "No validation results were generated."
        )

    return pd.concat(
        all_results,
        ignore_index=True,
    )


# ============================================================================
# SUMMARY
# ============================================================================

def summarize(results):

    print("\n" + "=" * 75)
    print("4. FINAL VALIDATION RESULTS")
    print("=" * 75)

    summary = (
        results
        .groupby("method")
        .agg(
            validation_weeks=(
                "week_start",
                "nunique",
            ),
            total_selected=(
                "selected",
                "sum",
            ),
            total_faults=(
                "actual_faults",
                "sum",
            ),
            caught_faults=(
                "caught_faults",
                "sum",
            ),
            missed_faults=(
                "missed_faults",
                "sum",
            ),
            total_cost=(
                "cost",
                "sum",
            ),
        )
        .reset_index()
    )

    summary["catch_rate"] = (
        summary["caught_faults"]
        / summary["total_faults"]
    )

    summary["cost_per_fault_caught"] = (
        summary["total_cost"]
        / summary["caught_faults"]
        .replace(0, np.nan)
    )

    print(
        summary.to_string(
            index=False
        )
    )

    return summary


# ============================================================================
# SAVE
# ============================================================================

def save_results(
    results,
    summary,
):

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Save a combined report.
    #
    # The first rows contain the summary followed by
    # the detailed weekly evaluation.

    summary_output = summary.copy()

    summary_output.insert(
        0,
        "report_type",
        "summary",
    )

    detail_output = results.copy()

    detail_output.insert(
        0,
        "report_type",
        "weekly",
    )

    combined = pd.concat(
        [
            summary_output,
            detail_output,
        ],
        ignore_index=True,
        sort=False,
    )

    combined.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("\n" + "=" * 75)
    print("5. SAVING VALIDATION REPORT")
    print("=" * 75)

    print(
        f"Saved:\n{OUTPUT_PATH}"
    )


# ============================================================================
# MAIN
# ============================================================================

def main():

    print("\n" + "=" * 75)
    print("STEP 06 - FINAL MODEL VALIDATION")
    print("=" * 75)

    dataset = load_dataset()

    telemetry = load_telemetry()

    results = run_forward_validation(
        dataset,
        telemetry,
    )

    summary = summarize(
        results
    )

    save_results(
        results,
        summary,
    )

    print("\n" + "=" * 75)
    print("STEP 06 COMPLETE")
    print("=" * 75)

    print(
        "\nHistorical validation compares:"
    )

    print(
        "  1. Official 3-sigma baseline"
    )

    print(
        "  2. HGB positive-class weight = 2"
    )

    print(
        "  3. 75% HGB + 25% 3-sigma"
    )

    print(
        "\nThe challenge weeks remain inference-only."
    )


if __name__ == "__main__":
    main()