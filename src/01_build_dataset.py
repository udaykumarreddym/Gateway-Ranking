"""
======================================================================
STEP 01 - BUILD HISTORICAL DATASET
======================================================================

Purpose:
    Build the complete cutoff-safe historical gateway-week dataset
    used to train the final gateway fault-risk model.

Pipeline:
    1. Load gateway master
    2. Load field visits
    3. Create confirmed-fault events
    4. Create 332 x 22 historical gateway-week grid
    5. Create fault_next_7d target
    6. Load telemetry
    7. Create cutoff-safe 7/14/28-day telemetry features
    8. Create previous-week meter features
    9. Add gateway metadata
    10. Select final model features
    11. Remove constant / >90% missing features
    12. Save final historical dataset

Output:
    outputs/gateway_week_features_reduced.csv

Expected:
    Rows: 7304
    Historical gateways: 332
    Historical weeks: 22
    Positive labels: 91
    Final columns: approximately 54
    Model features: 51

IMPORTANT:
    Feature values only use information available BEFORE each
    gateway-week cutoff.
======================================================================
"""

from pathlib import Path

import numpy as np
import pandas as pd


# =====================================================================
# CONFIGURATION
# =====================================================================

ROOT = Path(__file__).resolve().parent.parent

DATA = ROOT / "data"
OUTPUT = ROOT / "outputs"

OUTPUT.mkdir(parents=True, exist_ok=True)

HISTORICAL_START = pd.Timestamp(
    "2025-09-01",
    tz="UTC"
)

HISTORICAL_END = pd.Timestamp(
    "2026-01-26",
    tz="UTC"
)

OUTPUT_FILE = (
    OUTPUT /
    "gateway_week_features_reduced.csv"
)


# =====================================================================
# GATEWAY ID NORMALIZATION
# =====================================================================

def normalize_gateway_id(value):
    """
    Normalize gateway IDs so identifiers from different files
    can be matched reliably.
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


# =====================================================================
# TELEMETRY COLUMNS
# =====================================================================

TELEMETRY_COLUMNS = [
    "offline_duration_sec",
    "disconnection_cnt",
    "reboot_cnt",
    "reboot_duration_sec",
    "r_cnt_power_cycle",
    "r_cnt_reboot",
    "r_cnt_unknown",
    "r_dur_power_cycle",
    "r_dur_reboot",
    "r_dur_unknown",
    "avg_reboot_duration",
    "reboot_importance",
    "avg_offline_duration",
    "online_duration_mins",
    "no_conn_importance",
    "avg_load1",
    "load1_bigger1",
    "load1_bigger2",
    "avg_memfree",
    "avg_uptime",
    "avg_activeproccess",
    "avg_totalproccess",
    "rx_nr_pkts",
    "rx_crc_bad",
    "tx_success",
    "tx_busy",
    "tx_override",
    "number_of_messages",
    "network_2g",
    "network_3g",
    "network_4g",
    "network_unknown",
    "operator_unknown",
    "operator_3AT",
    "operator_A1",
    "operator_Eplus",
    "operator_O2DE",
    "operator_OrangeLU",
    "operator_Salt",
    "operator_Swisscom",
    "operator_TmobileA",
    "operator_TelekomDE",
    "operator_VodafoneDE",
    "rssi_good",
    "rssi_normal",
    "rssi_bad",
    "rscp_rsrp_good",
    "rscp_rsrp_normal",
    "rscp_rsrp_bad",
    "ecio_rsrq_good",
    "ecio_rsrq_normal",
    "ecio_rsrq_bad",
]


# =====================================================================
# TELEMETRY FEATURES USED FOR AGGREGATION
# =====================================================================

BASE_FEATURES = [
    "offline_duration_sec",
    "disconnection_cnt",
    "reboot_cnt",
    "reboot_duration_sec",
    "r_cnt_power_cycle",
    "r_cnt_reboot",
    "r_cnt_unknown",
    "r_dur_power_cycle",
    "r_dur_reboot",
    "r_dur_unknown",
    "avg_reboot_duration",
    "reboot_importance",
    "avg_offline_duration",
    "online_duration_mins",
    "no_conn_importance",
    "avg_load1",
    "load1_bigger1",
    "load1_bigger2",
    "avg_memfree",
    "avg_uptime",
    "avg_activeproccess",
    "avg_totalproccess",
    "rx_nr_pkts",
    "rx_crc_bad",
    "tx_success",
    "tx_busy",
    "tx_override",
    "number_of_messages",
    "network_2g",
    "network_3g",
    "network_4g",
    "network_unknown",
    "rssi_good",
    "rssi_normal",
    "rssi_bad",
    "rscp_rsrp_good",
    "rscp_rsrp_normal",
    "rscp_rsrp_bad",
    "ecio_rsrq_good",
    "ecio_rsrq_normal",
    "ecio_rsrq_bad",
    "rssi_bad_ratio",
    "rscp_rsrp_bad_ratio",
    "ecio_rsrq_bad_ratio",
    "crc_error_ratio",
    "tx_busy_ratio",
]


WINDOWS = {
    "7d": 7,
    "14d": 14,
    "28d": 28,
}


# =====================================================================
# FINAL FEATURE SELECTION
# =====================================================================

CATEGORICAL_COLUMNS = [
    "tenant",
    "site_type",
    "region",
    "hw_model",
    "antenna_type",
    "fw_version",
]


METER_FEATURES = [
    "meter_success_2w_mean",
    "meter_success_4w_mean",
    "meter_success_last_week",
    "meter_success_trend",
    "meter_expected_4w_mean",
]


TELEMETRY_BASES_FOR_SELECTION = [
    "offline_duration_sec",
    "disconnection_cnt",
    "reboot_cnt",
    "load_mean",
    "rssi",
    "rscp",
    "rsrp",
    "ecio",
    "rsrq",
    "crc_error",
    "tx_busy",
]


STAT_SUFFIXES = [
    "_7d_mean",
    "_7d_sum",
    "_7d_max",
    "_7d_std",
    "_14d_mean",
    "_14d_sum",
    "_14d_max",
    "_28d_mean",
    "_28d_sum",
    "_28d_max",
    "_trend_7d",
]


RATIO_FEATURES = [
    "rssi_bad_ratio_7d",
    "rssi_bad_ratio_14d",
    "rssi_bad_ratio_28d",

    "rscp_rsrp_bad_ratio_7d",
    "rscp_rsrp_bad_ratio_14d",
    "rscp_rsrp_bad_ratio_28d",

    "ecio_rsrq_bad_ratio_7d",
    "ecio_rsrq_bad_ratio_14d",
    "ecio_rsrq_bad_ratio_28d",

    "crc_error_ratio_7d",
    "crc_error_ratio_14d",
    "crc_error_ratio_28d",

    "tx_busy_ratio_7d",
    "tx_busy_ratio_14d",
    "tx_busy_ratio_28d",
]


# =====================================================================
# LOAD MASTER
# =====================================================================

def load_master():

    print("=" * 70)
    print("1. LOADING GATEWAY MASTER")
    print("=" * 70)

    master = pd.read_csv(
        DATA / "gateway_master.csv",
        encoding="latin1",
    )

    master["gateway_id"] = (
        master["gateway_id"]
        .apply(normalize_gateway_id)
    )

    gateway_ids = (
        master["gateway_id"]
        .dropna()
        .unique()
    )

    print(f"Master gateways: {len(gateway_ids)}")

    if len(gateway_ids) != 332:
        print(
            f"WARNING: expected 332 gateways, "
            f"found {len(gateway_ids)}"
        )

    return master, gateway_ids


# =====================================================================
# BUILD FAULT EVENTS
# =====================================================================

def build_fault_events():

    print()
    print("=" * 70)
    print("2. BUILDING CONFIRMED FAULT EVENTS")
    print("=" * 70)

    visits = pd.read_csv(
        DATA / "field_visits.csv",
        encoding="latin1",
    )

    visits["requested_on"] = pd.to_datetime(
        visits["requested_on"],
        utc=True,
    )

    visits["visited_on"] = pd.to_datetime(
        visits["visited_on"],
        utc=True,
    )

    visits["gateway_id"] = (
        visits["gateway_id"]
        .apply(normalize_gateway_id)
    )

    fixed = visits[
        visits["outcome"] == "Fehler behoben"
    ].copy()

    fixed["fault_date"] = (
        fixed["requested_on"]
        .dt.normalize()
    )

    faults = (
        fixed[
            [
                "gateway_id",
                "fault_date",
            ]
        ]
        .drop_duplicates()
    )

    print(f"Confirmed repair events: {len(faults)}")
    print(
        f"Unique gateways with confirmed repairs: "
        f"{faults['gateway_id'].nunique()}"
    )

    return faults


# =====================================================================
# BUILD HISTORICAL GATEWAY-WEEK GRID
# =====================================================================

def build_gateway_week(
    master,
    gateway_ids,
    faults,
):

    print()
    print("=" * 70)
    print("3. BUILDING HISTORICAL GATEWAY-WEEK DATASET")
    print("=" * 70)

    weeks = pd.date_range(
        HISTORICAL_START,
        HISTORICAL_END,
        freq="7D",
    )

    print(f"Historical weeks: {len(weeks)}")

    gateway_week = (
        pd.MultiIndex.from_product(
            [
                gateway_ids,
                weeks,
            ],
            names=[
                "gateway_id",
                "week_start",
            ],
        )
        .to_frame(index=False)
    )

    print(
        f"Gateway-week rows: {len(gateway_week)}"
    )

    # -------------------------------------------------------------
    # Create future 7-day target
    # -------------------------------------------------------------

    target = gateway_week.merge(
        faults,
        on="gateway_id",
        how="left",
    )

    target["fault_next_7d"] = (
        (target["fault_date"] >= target["week_start"])
        &
        (
            target["fault_date"]
            <
            target["week_start"]
            + pd.Timedelta(days=7)
        )
    )

    target = (
        target
        .groupby(
            [
                "gateway_id",
                "week_start",
            ],
            as_index=False,
        )["fault_next_7d"]
        .max()
    )

    target["fault_next_7d"] = (
        target["fault_next_7d"]
        .astype(int)
    )

    # -------------------------------------------------------------
    # Add metadata
    # -------------------------------------------------------------

    metadata_columns = [
        "gateway_id",
        "tenant",
        "site_type",
        "region",
        "hw_model",
        "antenna_type",
        "fw_version",
        "fw_updated_on",
        "installed_on",
        "decommissioned_on",
        "n_meters_installed",
    ]

    metadata = master[
        [
            c
            for c in metadata_columns
            if c in master.columns
        ]
    ].copy()

    target = target.merge(
        metadata,
        on="gateway_id",
        how="left",
    )

    print()
    print("Target distribution:")
    print(
        target["fault_next_7d"]
        .value_counts()
    )

    print(
        f"\nPositive examples: "
        f"{target['fault_next_7d'].sum()}"
    )

    print(
        f"Positive rate: "
        f"{target['fault_next_7d'].mean():.4%}"
    )

    return target


# =====================================================================
# LOAD TELEMETRY
# =====================================================================

def load_telemetry():

    print()
    print("=" * 70)
    print("4. LOADING TELEMETRY")
    print("=" * 70)

    telemetry_files = sorted(
        (
            DATA / "telemetry"
        ).glob(
            "month=*/part-*.parquet"
        )
    )

    print(
        f"Telemetry parquet files: "
        f"{len(telemetry_files)}"
    )

    if not telemetry_files:
        raise FileNotFoundError(
            "No telemetry parquet files found under "
            "data/telemetry/"
        )

    frames = []

    columns = [
        "gateway_id",
        "ts_utc",
        *TELEMETRY_COLUMNS,
    ]

    for file in telemetry_files:

        print(
            f"  Loading: {file.relative_to(ROOT)}"
        )

        df = pd.read_parquet(
            file,
            columns=columns,
        )

        df["gateway_id"] = (
            df["gateway_id"]
            .apply(normalize_gateway_id)
        )

        df["ts"] = pd.to_datetime(
            df["ts_utc"],
            utc=True,
        )

        df = df.drop(
            columns=["ts_utc"]
        )

        frames.append(df)

    telemetry = pd.concat(
        frames,
        ignore_index=True,
    )

    for col in TELEMETRY_COLUMNS:

        telemetry[col] = pd.to_numeric(
            telemetry[col],
            errors="coerce",
        )

    telemetry = (
        telemetry
        .sort_values(
            [
                "gateway_id",
                "ts",
            ]
        )
        .reset_index(drop=True)
    )

    print(
        f"\nTelemetry rows: "
        f"{len(telemetry):,}"
    )

    print(
        f"Telemetry gateways: "
        f"{telemetry['gateway_id'].nunique()}"
    )

    print(
        f"Telemetry range: "
        f"{telemetry['ts'].min()} "
        f"→ "
        f"{telemetry['ts'].max()}"
    )

    return telemetry


# =====================================================================
# DERIVED TELEMETRY FEATURES
# =====================================================================

def create_derived_telemetry_features(
    telemetry,
):

    print()
    print("=" * 70)
    print("5. CREATING DERIVED TELEMETRY FEATURES")
    print("=" * 70)

    # -------------------------------------------------------------
    # RSSI bad ratio
    # -------------------------------------------------------------

    rssi_total = telemetry[
        [
            "rssi_good",
            "rssi_normal",
            "rssi_bad",
        ]
    ].sum(axis=1)

    telemetry["rssi_bad_ratio"] = np.where(
        rssi_total > 0,
        telemetry["rssi_bad"] / rssi_total,
        np.nan,
    )

    # -------------------------------------------------------------
    # RSCP / RSRP bad ratio
    # -------------------------------------------------------------

    rsrp_total = telemetry[
        [
            "rscp_rsrp_good",
            "rscp_rsrp_normal",
            "rscp_rsrp_bad",
        ]
    ].sum(axis=1)

    telemetry["rscp_rsrp_bad_ratio"] = np.where(
        rsrp_total > 0,
        telemetry["rscp_rsrp_bad"] / rsrp_total,
        np.nan,
    )

    # -------------------------------------------------------------
    # ECIO / RSRQ bad ratio
    # -------------------------------------------------------------

    rsrq_total = telemetry[
        [
            "ecio_rsrq_good",
            "ecio_rsrq_normal",
            "ecio_rsrq_bad",
        ]
    ].sum(axis=1)

    telemetry["ecio_rsrq_bad_ratio"] = np.where(
        rsrq_total > 0,
        telemetry["ecio_rsrq_bad"] / rsrq_total,
        np.nan,
    )

    # -------------------------------------------------------------
    # CRC error ratio
    # -------------------------------------------------------------

    telemetry["crc_error_ratio"] = np.where(
        telemetry["rx_nr_pkts"] > 0,
        telemetry["rx_crc_bad"]
        / telemetry["rx_nr_pkts"],
        np.nan,
    )

    # -------------------------------------------------------------
    # TX busy ratio
    # -------------------------------------------------------------

    tx_total = (
        telemetry["tx_success"]
        + telemetry["tx_busy"]
    )

    telemetry["tx_busy_ratio"] = np.where(
        tx_total > 0,
        telemetry["tx_busy"] / tx_total,
        np.nan,
    )

    return telemetry


# =====================================================================
# CREATE CUTOFF-SAFE TELEMETRY FEATURES
# =====================================================================

def create_telemetry_features(
    gateway_week,
    telemetry,
):

    print()
    print("=" * 70)
    print("6. CREATING CUTOFF-SAFE TELEMETRY FEATURES")
    print("=" * 70)

    feature_frames = []

    cutoffs = (
        gateway_week["week_start"]
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    total_cutoffs = len(cutoffs)

    for index, cutoff in enumerate(
        cutoffs,
        start=1,
    ):

        print(
            f"[{index}/{total_cutoffs}] "
            f"Cutoff: {cutoff.date()}"
        )

        # ---------------------------------------------------------
        # CRITICAL LEAKAGE RULE
        #
        # Only observations strictly before cutoff.
        # ---------------------------------------------------------

        history = telemetry[
            telemetry["ts"] < cutoff
        ].copy()

        history = history[
            history["ts"]
            >=
            cutoff - pd.Timedelta(days=28)
        ]

        if history.empty:
            continue

        gateway_subset = gateway_week[
            gateway_week["week_start"] == cutoff
        ][
            ["gateway_id"]
        ]

        weekly_rows = []

        for gateway_id in gateway_subset[
            "gateway_id"
        ]:

            gateway_history = history[
                history["gateway_id"] == gateway_id
            ]

            row = {
                "gateway_id": gateway_id,
                "week_start": cutoff,
            }

            # -----------------------------------------------------
            # 7 / 14 / 28 DAY STATISTICS
            # -----------------------------------------------------

            for window_name, days in WINDOWS.items():

                window_start = (
                    cutoff
                    - pd.Timedelta(days=days)
                )

                window_data = gateway_history[
                    gateway_history["ts"]
                    >= window_start
                ]

                for column in BASE_FEATURES:

                    values = window_data[column]

                    row[
                        f"{column}_{window_name}_mean"
                    ] = values.mean()

                    row[
                        f"{column}_{window_name}_sum"
                    ] = values.sum()

                    row[
                        f"{column}_{window_name}_max"
                    ] = values.max()

                    row[
                        f"{column}_{window_name}_std"
                    ] = values.std()

            # -----------------------------------------------------
            # 7-DAY TREND
            #
            # Recent 7-day mean minus previous 7-day mean.
            # -----------------------------------------------------

            recent_start = (
                cutoff
                - pd.Timedelta(days=7)
            )

            previous_start = (
                cutoff
                - pd.Timedelta(days=14)
            )

            recent = gateway_history[
                gateway_history["ts"]
                >= recent_start
            ]

            previous = gateway_history[
                (
                    gateway_history["ts"]
                    >= previous_start
                )
                &
                (
                    gateway_history["ts"]
                    < recent_start
                )
            ]

            for column in BASE_FEATURES:

                recent_mean = recent[column].mean()
                previous_mean = previous[column].mean()

                row[
                    f"{column}_trend"
                ] = (
                    recent_mean
                    - previous_mean
                )

            weekly_rows.append(row)

        if weekly_rows:
            feature_frames.append(
                pd.DataFrame(weekly_rows)
            )

    telemetry_features = pd.concat(
        feature_frames,
        ignore_index=True,
    )

    print()
    print(
        f"Telemetry feature rows: "
        f"{len(telemetry_features):,}"
    )

    print(
        f"Telemetry feature columns: "
        f"{len(telemetry_features.columns)}"
    )

    return telemetry_features


# =====================================================================
# CREATE METER FEATURES
# =====================================================================

def create_meter_features():

    print()
    print("=" * 70)
    print("7. CREATING CUTOFF-SAFE METER FEATURES")
    print("=" * 70)

    meter = pd.read_csv(
        DATA / "meter_read_success.csv",
        encoding="latin1",
    )

    meter["gateway_id"] = (
        meter["gateway_id"]
        .apply(normalize_gateway_id)
    )

    meter["week_start"] = pd.to_datetime(
        meter["week_start"],
        utc=True,
    )

    meter["meter_read_success"] = np.where(
        meter["meters_expected"] > 0,
        (
            meter["meters_read"]
            /
            meter["meters_expected"]
        ),
        np.nan,
    )

    meter = (
        meter
        .sort_values(
            [
                "gateway_id",
                "week_start",
            ]
        )
        .copy()
    )

    feature_frames = []

    for gateway_id, group in meter.groupby(
        "gateway_id"
    ):

        group = (
            group
            .sort_values("week_start")
            .copy()
        )

        group[
            "meter_success_2w_mean"
        ] = (
            group["meter_read_success"]
            .rolling(
                2,
                min_periods=1,
            )
            .mean()
        )

        group[
            "meter_success_4w_mean"
        ] = (
            group["meter_read_success"]
            .rolling(
                4,
                min_periods=1,
            )
            .mean()
        )

        group[
            "meter_success_last_week"
        ] = group[
            "meter_read_success"
        ]

        group[
            "meter_success_trend"
        ] = (
            group["meter_success_2w_mean"]
            -
            group["meter_success_4w_mean"]
        )

        group[
            "meter_expected_4w_mean"
        ] = (
            group["meters_expected"]
            .rolling(
                4,
                min_periods=1,
            )
            .mean()
        )

        feature_frames.append(
            group[
                [
                    "gateway_id",
                    "week_start",
                    "meter_success_2w_mean",
                    "meter_success_4w_mean",
                    "meter_success_last_week",
                    "meter_success_trend",
                    "meter_expected_4w_mean",
                ]
            ]
        )

    meter_features = pd.concat(
        feature_frames,
        ignore_index=True,
    )

    # -------------------------------------------------------------
    # CRITICAL CUTOFF SHIFT
    #
    # A meter row for week T is the result of week T.
    # It cannot be used at the Monday cutoff for week T.
    #
    # Shift the source week forward by 7 days so that the features
    # attached to prediction week T originate from T-7.
    # -------------------------------------------------------------

    meter_features["week_start"] = (
        meter_features["week_start"]
        + pd.Timedelta(days=7)
    )

    print(
        f"Meter feature rows: "
        f"{len(meter_features):,}"
    )

    return meter_features


# =====================================================================
# MERGE ALL FEATURES
# =====================================================================

def merge_features(
    gateway_week,
    telemetry_features,
    meter_features,
):

    print()
    print("=" * 70)
    print("8. MERGING ALL FEATURES")
    print("=" * 70)

    result = gateway_week.merge(
        telemetry_features,
        on=[
            "gateway_id",
            "week_start",
        ],
        how="left",
    )

    result = result.merge(
        meter_features,
        on=[
            "gateway_id",
            "week_start",
        ],
        how="left",
    )

    print(
        f"Rows after merge: {len(result):,}"
    )

    print(
        f"Columns after merge: "
        f"{len(result.columns)}"
    )

    return result


# =====================================================================
# SELECT FINAL MODEL FEATURES
# =====================================================================

def select_final_features(df):

    print()
    print("=" * 70)
    print("9. SELECTING FINAL MODEL FEATURES")
    print("=" * 70)

    id_columns = [
        "gateway_id",
        "week_start",
    ]

    target_columns = [
        "fault_next_7d",
    ]

    # -------------------------------------------------------------
    # Telemetry
    # -------------------------------------------------------------

    selected_telemetry = []

    existing_columns = set(df.columns)

    for base in TELEMETRY_BASES_FOR_SELECTION:

        for suffix in STAT_SUFFIXES:

            column = base + suffix

            if column in existing_columns:
                selected_telemetry.append(column)

    # -------------------------------------------------------------
    # Ratios
    # -------------------------------------------------------------

    selected_ratios = [
        column
        for column in RATIO_FEATURES
        if column in existing_columns
    ]

    # -------------------------------------------------------------
    # Baseline-inspired features
    #
    # Preserve the original selection behavior.
    # -------------------------------------------------------------

    baseline_keywords = [
        "zscore",
        "sigma",
        "anomaly",
        "flag",
        "threshold",
    ]

    selected_baseline = []

    for column in df.columns:

        column_lower = column.lower()

        if any(
            keyword in column_lower
            for keyword in baseline_keywords
        ):

            if column not in (
                id_columns
                + target_columns
            ):

                selected_baseline.append(column)

    selected_columns = (
        id_columns
        + target_columns
        + CATEGORICAL_COLUMNS
        + METER_FEATURES
        + selected_telemetry
        + selected_ratios
        + selected_baseline
    )

    # Remove duplicates while preserving order.
    selected_columns = list(
        dict.fromkeys(selected_columns)
    )

    # Keep only columns that actually exist.
    selected_columns = [
        column
        for column in selected_columns
        if column in df.columns
    ]

    reduced = df[
        selected_columns
    ].copy()

    print(
        f"Selected columns before cleanup: "
        f"{len(reduced.columns)}"
    )

    # -------------------------------------------------------------
    # Remove constant features
    # -------------------------------------------------------------

    protected = (
        id_columns
        + target_columns
        + CATEGORICAL_COLUMNS
    )

    feature_columns = [
        column
        for column in reduced.columns
        if column not in protected
    ]

    constant_columns = [
        column
        for column in feature_columns
        if reduced[column]
        .nunique(dropna=False)
        <= 1
    ]

    if constant_columns:

        print()
        print("Removing constant features:")

        for column in constant_columns:
            print(f"  - {column}")

        reduced.drop(
            columns=constant_columns,
            inplace=True,
        )

    # -------------------------------------------------------------
    # Remove features with >90% missing values
    # -------------------------------------------------------------

    feature_columns = [
        column
        for column in reduced.columns
        if column not in protected
    ]

    missing_ratio = (
        reduced[feature_columns]
        .isna()
        .mean()
    )

    sparse_columns = (
        missing_ratio[
            missing_ratio > 0.90
        ]
        .index
        .tolist()
    )

    if sparse_columns:

        print()
        print(
            "Removing features with >90% missing:"
        )

        for column in sparse_columns:

            print(
                f"  - {column}: "
                f"{missing_ratio[column]:.2%}"
            )

        reduced.drop(
            columns=sparse_columns,
            inplace=True,
        )

    return reduced


# =====================================================================
# VALIDATE DATASET
# =====================================================================

def validate_dataset(df):

    print()
    print("=" * 70)
    print("10. VALIDATING HISTORICAL DATASET")
    print("=" * 70)

    expected_rows = 332 * 22

    if len(df) != expected_rows:

        raise ValueError(
            f"Expected {expected_rows} rows, "
            f"got {len(df)}"
        )

    if df["gateway_id"].nunique() != 332:

        raise ValueError(
            "Expected 332 unique gateways."
        )

    if df["week_start"].nunique() != 22:

        raise ValueError(
            "Expected 22 historical weeks."
        )

    duplicate_count = df.duplicated(
        [
            "gateway_id",
            "week_start",
        ]
    ).sum()

    if duplicate_count:

        raise ValueError(
            f"Found {duplicate_count} "
            "duplicate gateway/week rows."
        )

    positives = int(
        df["fault_next_7d"].sum()
    )

    print(
        f"Rows: {len(df):,}"
    )

    print(
        f"Unique gateways: "
        f"{df['gateway_id'].nunique()}"
    )

    print(
        f"Unique weeks: "
        f"{df['week_start'].nunique()}"
    )

    print(
        f"Positive labels: "
        f"{positives}"
    )

    print(
        f"Positive rate: "
        f"{df['fault_next_7d'].mean():.4%}"
    )

    print(
        f"Duplicate gateway/week rows: "
        f"{duplicate_count}"
    )

    # -------------------------------------------------------------
    # Verify required categorical columns.
    # -------------------------------------------------------------

    missing_categorical = [
        column
        for column in CATEGORICAL_COLUMNS
        if column not in df.columns
    ]

    if missing_categorical:

        raise ValueError(
            "Missing categorical features: "
            f"{missing_categorical}"
        )

    # -------------------------------------------------------------
    # Model feature count
    # -------------------------------------------------------------

    model_features = [
        column
        for column in df.columns
        if column not in (
            [
                "gateway_id",
                "week_start",
                "fault_next_7d",
            ]
            + CATEGORICAL_COLUMNS
        )
    ]

    print(
        f"Model feature columns: "
        f"{len(model_features) + len(CATEGORICAL_COLUMNS)}"
    )

    print()
    print("[PASS] Dataset validation complete.")


# =====================================================================
# MAIN
# =====================================================================

def main():

    print()
    print("=" * 70)
    print("STEP 01 - BUILD HISTORICAL DATASET")
    print("=" * 70)
    print()

    # -------------------------------------------------------------
    # 1. Master
    # -------------------------------------------------------------

    master, gateway_ids = load_master()

    # -------------------------------------------------------------
    # 2. Fault events
    # -------------------------------------------------------------

    faults = build_fault_events()

    # -------------------------------------------------------------
    # 3. Gateway-week + target
    # -------------------------------------------------------------

    gateway_week = build_gateway_week(
        master,
        gateway_ids,
        faults,
    )

    # -------------------------------------------------------------
    # 4. Telemetry
    # -------------------------------------------------------------

    telemetry = load_telemetry()

    # -------------------------------------------------------------
    # 5. Derived telemetry
    # -------------------------------------------------------------

    telemetry = (
        create_derived_telemetry_features(
            telemetry
        )
    )

    # -------------------------------------------------------------
    # 6. Cutoff-safe telemetry features
    # -------------------------------------------------------------

    telemetry_features = (
        create_telemetry_features(
            gateway_week,
            telemetry,
        )
    )

    # -------------------------------------------------------------
    # 7. Meter features
    # -------------------------------------------------------------

    meter_features = (
        create_meter_features()
    )

    # -------------------------------------------------------------
    # 8. Merge
    # -------------------------------------------------------------

    dataset = merge_features(
        gateway_week,
        telemetry_features,
        meter_features,
    )

    # -------------------------------------------------------------
    # 9. Feature selection
    # -------------------------------------------------------------

    dataset = select_final_features(
        dataset
    )

    # -------------------------------------------------------------
    # 10. Validate
    # -------------------------------------------------------------

    validate_dataset(dataset)

    # -------------------------------------------------------------
    # Save ONLY the final training dataset.
    #
    # We deliberately do not save:
    #   confirmed_fault_events.csv
    #   gateway_week_target.csv
    #   gateway_week_features.csv
    #   gateway_week_features_v2.csv
    #
    # They are intermediate artifacts and can be rebuilt by this
    # script whenever required.
    # -------------------------------------------------------------

    dataset.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print("=" * 70)
    print("STEP 01 COMPLETE")
    print("=" * 70)

    print(
        f"Final dataset shape: "
        f"{dataset.shape}"
    )

    print(
        f"Saved: "
        f"{OUTPUT_FILE.relative_to(ROOT)}"
    )

    print()
    print("Final columns:")

    for column in dataset.columns:
        print(f"  {column}")

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()