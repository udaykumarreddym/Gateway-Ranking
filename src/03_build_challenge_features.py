"""
======================================================================
STEP 03 - BUILD CHALLENGE FEATURES
======================================================================

Purpose:
    Build cutoff-safe features for all challenge gateway-weeks.

Challenge period:
    2026-02-02 through 2026-03-23

Rows:
    332 gateways x 8 weeks = 2,656

IMPORTANT:
    Only information available before each challenge cutoff is used.

Output:
    outputs/challenge_gateway_week_features.csv

Expected final schema:
    gateway_id
    week_start
    51 model features

The 51 model features consist of:
    6 categorical gateway metadata features
    5 meter features
    40 telemetry features

======================================================================
"""

from pathlib import Path

import numpy as np
import pandas as pd


# =====================================================================
# PATHS
# =====================================================================

ROOT = Path(__file__).resolve().parent.parent

DATA = ROOT / "data"
OUTPUT = ROOT / "outputs"

OUTPUT.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_FILE = (
    OUTPUT /
    "challenge_gateway_week_features.csv"
)

MODEL_FEATURE_FILE = (
    OUTPUT /
    "gateway_week_features_reduced.csv"
)


# =====================================================================
# CHALLENGE WEEKS
# =====================================================================

CHALLENGE_WEEKS = pd.date_range(
    "2026-02-02",
    "2026-03-23",
    freq="7D",
    tz="UTC",
)


# =====================================================================
# TELEMETRY WINDOWS
# =====================================================================

WINDOWS = {
    "7d": 7,
    "14d": 14,
    "28d": 28,
}


# =====================================================================
# CATEGORICAL MODEL FEATURES
# =====================================================================

CATEGORICAL_FEATURES = [
    "tenant",
    "site_type",
    "region",
    "hw_model",
    "antenna_type",
    "fw_version",
]


# =====================================================================
# METER MODEL FEATURES
# =====================================================================

METER_FEATURES = [
    "meter_success_2w_mean",
    "meter_success_4w_mean",
    "meter_success_last_week",
    "meter_success_trend",
    "meter_expected_4w_mean",
]


# =====================================================================
# TELEMETRY SOURCE COLUMNS
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
# TELEMETRY FEATURES USED AS BASES
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


# =====================================================================
# GATEWAY ID NORMALIZATION
# =====================================================================

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


# =====================================================================
# LOAD MASTER
# =====================================================================

def load_master():

    print()
    print("=" * 70)
    print("1. LOADING GATEWAY MASTER")
    print("=" * 70)

    master = pd.read_csv(
        DATA / "gateway_master.csv",
        encoding="latin1",
    )

    if "gateway_id" not in master.columns:
        raise ValueError(
            "gateway_master.csv does not contain gateway_id."
        )

    master["gateway_id"] = (
        master["gateway_id"]
        .apply(normalize_gateway_id)
    )

    master = (
        master
        .dropna(subset=["gateway_id"])
        .drop_duplicates(
            subset=["gateway_id"],
            keep="first",
        )
        .reset_index(drop=True)
    )

    gateway_ids = (
        master["gateway_id"]
        .unique()
    )

    print(
        f"Master gateways: {len(gateway_ids)}"
    )

    if len(gateway_ids) != 332:
        raise ValueError(
            f"Expected 332 gateways, "
            f"found {len(gateway_ids)}."
        )

    missing = [
        column
        for column in CATEGORICAL_FEATURES
        if column not in master.columns
    ]

    if missing:
        raise ValueError(
            "Gateway master is missing metadata columns:\n"
            +
            "\n".join(
                f"  - {column}"
                for column in missing
            )
        )

    return master, gateway_ids


# =====================================================================
# LOAD TELEMETRY
# =====================================================================

def load_telemetry():

    print()
    print("=" * 70)
    print("2. LOADING TELEMETRY")
    print("=" * 70)

    telemetry_files = sorted(
        (
            DATA / "telemetry"
        ).glob(
            "month=*/part-*.parquet"
        )
    )

    if not telemetry_files:
        raise FileNotFoundError(
            "No telemetry parquet files found."
        )

    print(
        f"Parquet files: "
        f"{len(telemetry_files)}"
    )

    columns = [
        "gateway_id",
        "ts_utc",
        *TELEMETRY_COLUMNS,
    ]

    frames = []

    for file in telemetry_files:

        print(
            f"  Loading: "
            f"{file.relative_to(ROOT)}"
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
            errors="coerce",
            utc=True,
        )

        df.drop(
            columns=["ts_utc"],
            inplace=True,
        )

        frames.append(df)

    telemetry = pd.concat(
        frames,
        ignore_index=True,
    )

    for column in TELEMETRY_COLUMNS:

        telemetry[column] = pd.to_numeric(
            telemetry[column],
            errors="coerce",
        )

    telemetry = (
        telemetry
        .dropna(
            subset=[
                "gateway_id",
                "ts",
            ]
        )
        .sort_values(
            [
                "gateway_id",
                "ts",
            ]
        )
        .reset_index(drop=True)
    )

    print(
        f"Telemetry rows: "
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
# CREATE DERIVED TELEMETRY FEATURES
# =====================================================================

def create_derived_features(
    telemetry,
):

    print()
    print("=" * 70)
    print("3. CREATING DERIVED TELEMETRY FEATURES")
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
    ].sum(
        axis=1,
        min_count=1,
    )

    telemetry["rssi_bad_ratio"] = np.where(
        rssi_total > 0,
        telemetry["rssi_bad"] / rssi_total,
        np.nan,
    )

    # -------------------------------------------------------------
    # RSRP bad ratio
    # -------------------------------------------------------------

    rsrp_total = telemetry[
        [
            "rscp_rsrp_good",
            "rscp_rsrp_normal",
            "rscp_rsrp_bad",
        ]
    ].sum(
        axis=1,
        min_count=1,
    )

    telemetry["rscp_rsrp_bad_ratio"] = np.where(
        rsrp_total > 0,
        telemetry["rscp_rsrp_bad"] / rsrp_total,
        np.nan,
    )

    # -------------------------------------------------------------
    # RSRQ bad ratio
    # -------------------------------------------------------------

    rsrq_total = telemetry[
        [
            "ecio_rsrq_good",
            "ecio_rsrq_normal",
            "ecio_rsrq_bad",
        ]
    ].sum(
        axis=1,
        min_count=1,
    )

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
        /
        telemetry["rx_nr_pkts"],
        np.nan,
    )

    # -------------------------------------------------------------
    # TX busy ratio
    # -------------------------------------------------------------

    tx_total = (
        telemetry["tx_success"]
        +
        telemetry["tx_busy"]
    )

    telemetry["tx_busy_ratio"] = np.where(
        tx_total > 0,
        telemetry["tx_busy"] / tx_total,
        np.nan,
    )

    return telemetry


# =====================================================================
# BUILD TELEMETRY FEATURES
# =====================================================================

def build_features(
    gateway_ids,
    telemetry,
):

    print()
    print("=" * 70)
    print("4. BUILDING CUTOFF-SAFE CHALLENGE FEATURES")
    print("=" * 70)

    all_rows = []

    for index, cutoff in enumerate(
        CHALLENGE_WEEKS,
        start=1,
    ):

        print(
            f"[{index}/{len(CHALLENGE_WEEKS)}] "
            f"Cutoff: {cutoff.date()}"
        )

        # ---------------------------------------------------------
        # STRICT CUTOFF
        # ---------------------------------------------------------

        history = telemetry[
            telemetry["ts"] < cutoff
        ].copy()

        # ---------------------------------------------------------
        # Maximum 28-day lookback.
        # ---------------------------------------------------------

        history = history[
            history["ts"]
            >=
            cutoff - pd.Timedelta(days=28)
        ]

        print(
            f"    Lookback rows: "
            f"{len(history):,}"
        )

        for gateway_id in gateway_ids:

            gateway_history = history[
                history["gateway_id"] == gateway_id
            ]

            row = {
                "gateway_id": gateway_id,
                "week_start": cutoff,
            }

            # -----------------------------------------------------
            # WINDOW STATISTICS
            # -----------------------------------------------------

            for window_name, days in WINDOWS.items():

                window_start = (
                    cutoff
                    -
                    pd.Timedelta(days=days)
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
            # TREND
            # -----------------------------------------------------

            recent_start = (
                cutoff
                -
                pd.Timedelta(days=7)
            )

            previous_start = (
                cutoff
                -
                pd.Timedelta(days=14)
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

                row[
                    f"{column}_trend"
                ] = (
                    recent[column].mean()
                    -
                    previous[column].mean()
                )

            all_rows.append(row)

    return pd.DataFrame(all_rows)


# =====================================================================
# LOAD METER DATA
# =====================================================================

def load_meter():

    print()
    print("=" * 70)
    print("5. LOADING METER DATA")
    print("=" * 70)

    file = DATA / "meter_read_success.csv"

    if not file.exists():
        raise FileNotFoundError(
            f"Meter file not found:\n{file}"
        )

    meter = pd.read_csv(
        file,
    )

    print(
        f"Meter rows: "
        f"{len(meter):,}"
    )

    print(
        f"Meter columns: "
        f"{list(meter.columns)}"
    )

    required = [
        "week_start",
        "gateway_id",
        "meters_expected",
        "meters_read",
    ]

    missing = [
        column
        for column in required
        if column not in meter.columns
    ]

    if missing:
        raise ValueError(
            "meter_read_success.csv is missing columns:\n"
            +
            "\n".join(
                f"  - {column}"
                for column in missing
            )
        )

    meter["gateway_id"] = (
        meter["gateway_id"]
        .apply(normalize_gateway_id)
    )

    meter["week_start"] = pd.to_datetime(
        meter["week_start"],
        errors="coerce",
        utc=True,
    )

    meter["meters_expected"] = pd.to_numeric(
        meter["meters_expected"],
        errors="coerce",
    )

    meter["meters_read"] = pd.to_numeric(
        meter["meters_read"],
        errors="coerce",
    )

    # -------------------------------------------------------------
    # Meter success rate.
    #
    # When meters_expected == 0, success is undefined.
    # -------------------------------------------------------------

    meter["meter_success"] = np.where(
        meter["meters_expected"] > 0,
        meter["meters_read"]
        /
        meter["meters_expected"],
        np.nan,
    )

    meter = meter.dropna(
        subset=[
            "gateway_id",
            "week_start",
        ]
    )

    meter = (
        meter
        .sort_values(
            [
                "gateway_id",
                "week_start",
            ]
        )
        .reset_index(drop=True)
    )

    print(
        f"Clean meter rows: "
        f"{len(meter):,}"
    )

    print(
        f"Meter gateways: "
        f"{meter['gateway_id'].nunique()}"
    )

    print(
        f"Meter date range: "
        f"{meter['week_start'].min()} "
        f"→ "
        f"{meter['week_start'].max()}"
    )

    return meter


# =====================================================================
# BUILD METER FEATURES
# =====================================================================

def build_meter_features(
    meter,
    gateway_week_grid,
):

    print()
    print("=" * 70)
    print("6. BUILDING CUTOFF-SAFE METER FEATURES")
    print("=" * 70)

    # -------------------------------------------------------------
    # Aggregate to gateway/week.
    # -------------------------------------------------------------

    weekly = (
        meter
        .groupby(
            [
                "gateway_id",
                "week_start",
            ],
            as_index=False,
        )
        .agg(
            meter_success=(
                "meter_success",
                "mean",
            ),
            meters_expected=(
                "meters_expected",
                "sum",
            ),
        )
    )

    weekly = weekly.sort_values(
        [
            "gateway_id",
            "week_start",
        ]
    )

    # -------------------------------------------------------------
    # IMPORTANT:
    #
    # A source meter week is assigned to the following target week.
    #
    # Source:
    #   2026-01-26
    #
    # Target:
    #   2026-02-02
    #
    # Therefore the target week only receives completed historical
    # meter information.
    # -------------------------------------------------------------

    weekly["target_week"] = (
        weekly["week_start"]
        +
        pd.Timedelta(days=7)
    )

    # -------------------------------------------------------------
    # Group by gateway.
    # -------------------------------------------------------------

    grouped_success = (
        weekly
        .groupby("gateway_id")["meter_success"]
    )

    grouped_expected = (
        weekly
        .groupby("gateway_id")["meters_expected"]
    )

    # -------------------------------------------------------------
    # Previous completed week.
    # -------------------------------------------------------------

    weekly["meter_last_week"] = (
        grouped_success.shift(1)
    )

    weekly["meter_two_weeks_ago"] = (
        grouped_success.shift(2)
    )

    # -------------------------------------------------------------
    # 2-week mean.
    # -------------------------------------------------------------

    weekly["meter_success_2w_mean"] = (
        weekly
        .groupby("gateway_id")[
            "meter_success"
        ]
        .transform(
            lambda x:
            x.shift(1)
            .rolling(
                2,
                min_periods=1,
            )
            .mean()
        )
    )

    # -------------------------------------------------------------
    # 4-week mean.
    # -------------------------------------------------------------

    weekly["meter_success_4w_mean"] = (
        weekly
        .groupby("gateway_id")[
            "meter_success"
        ]
        .transform(
            lambda x:
            x.shift(1)
            .rolling(
                4,
                min_periods=1,
            )
            .mean()
        )
    )

    # -------------------------------------------------------------
    # Last completed week.
    # -------------------------------------------------------------

    weekly["meter_success_last_week"] = (
        weekly["meter_last_week"]
    )

    # -------------------------------------------------------------
    # Trend.
    # -------------------------------------------------------------

    weekly["meter_success_trend"] = (
        weekly["meter_last_week"]
        -
        weekly["meter_two_weeks_ago"]
    )

    # -------------------------------------------------------------
    # Expected meter count, 4-week historical mean.
    #
    # This is intentionally based on meters_expected, not the
    # success ratio.
    # -------------------------------------------------------------

    weekly["meter_expected_4w_mean"] = (
        grouped_expected
        .transform(
            lambda x:
            x.shift(1)
            .rolling(
                4,
                min_periods=1,
            )
            .mean()
        )
    )

    # -------------------------------------------------------------
    # Keep model meter features.
    # -------------------------------------------------------------

    meter_features = weekly[
        [
            "gateway_id",
            "target_week",
            "meter_success_2w_mean",
            "meter_success_4w_mean",
            "meter_success_last_week",
            "meter_success_trend",
            "meter_expected_4w_mean",
        ]
    ].copy()

    meter_features = meter_features.rename(
        columns={
            "target_week": "week_start",
        }
    )

    meter_features = (
        meter_features
        .drop_duplicates(
            subset=[
                "gateway_id",
                "week_start",
            ]
        )
    )

    # -------------------------------------------------------------
    # Merge onto complete 332 x 8 grid.
    # -------------------------------------------------------------

    result = gateway_week_grid.merge(
        meter_features,
        on=[
            "gateway_id",
            "week_start",
        ],
        how="left",
        validate="one_to_one",
    )

    print(
        f"Meter feature rows: "
        f"{len(result):,}"
    )

    print()
    print("Meter feature missing values:")

    for column in METER_FEATURES:

        print(
            f"  {column}: "
            f"{result[column].isna().sum()}"
        )

    return result


# =====================================================================
# ADD GATEWAY METADATA
# =====================================================================

def add_gateway_metadata(
    challenge,
    master,
):

    print()
    print("=" * 70)
    print("7. ADDING GATEWAY METADATA")
    print("=" * 70)

    metadata = master[
        [
            "gateway_id",
            *CATEGORICAL_FEATURES,
        ]
    ].copy()

    metadata = metadata.drop_duplicates(
        subset=["gateway_id"]
    )

    challenge = challenge.merge(
        metadata,
        on="gateway_id",
        how="left",
        validate="many_to_one",
    )

    for column in CATEGORICAL_FEATURES:

        missing = (
            challenge[column]
            .isna()
            .sum()
        )

        print(
            f"  {column}: "
            f"missing={missing}"
        )

        if missing:
            raise ValueError(
                f"{column} contains "
                f"{missing} missing values."
            )

    return challenge


# =====================================================================
# MATCH MODEL SCHEMA
# =====================================================================

def match_model_schema(
    challenge,
):

    print()
    print("=" * 70)
    print("8. MATCHING MODEL FEATURE SCHEMA")
    print("=" * 70)

    if not MODEL_FEATURE_FILE.exists():

        raise FileNotFoundError(
            f"Historical model dataset not found:\n"
            f"{MODEL_FEATURE_FILE}"
        )

    historical = pd.read_csv(
        MODEL_FEATURE_FILE,
        nrows=1,
    )

    excluded = {
        "gateway_id",
        "week_start",
        "fault_next_7d",
    }

    model_features = [
        column
        for column in historical.columns
        if column not in excluded
    ]

    print(
        f"Historical model features: "
        f"{len(model_features)}"
    )

    if len(model_features) != 51:

        raise ValueError(
            f"Expected exactly 51 model features, "
            f"found {len(model_features)}."
        )

    # -------------------------------------------------------------
    # Check missing features.
    # -------------------------------------------------------------

    missing = [
        column
        for column in model_features
        if column not in challenge.columns
    ]

    if missing:

        print()
        print("Missing features:")

        for column in missing:
            print(
                f"  - {column}"
            )

        raise ValueError(
            "Challenge dataset is missing model features."
        )

    # -------------------------------------------------------------
    # Exact historical order.
    # -------------------------------------------------------------

    challenge = challenge[
        [
            "gateway_id",
            "week_start",
            *model_features,
        ]
    ].copy()

    print(
        f"Challenge model features: "
        f"{len(model_features)}"
    )

    # -------------------------------------------------------------
    # Feature groups.
    # -------------------------------------------------------------

    categorical_found = [
        column
        for column in CATEGORICAL_FEATURES
        if column in model_features
    ]

    meter_found = [
        column
        for column in METER_FEATURES
        if column in model_features
    ]

    numeric_features = [
        column
        for column in model_features
        if column not in CATEGORICAL_FEATURES
    ]

    telemetry_found = [
        column
        for column in numeric_features
        if column not in METER_FEATURES
    ]

    print(
        f"Categorical features: "
        f"{len(categorical_found)}"
    )

    print(
        f"Meter features: "
        f"{len(meter_found)}"
    )

    print(
        f"Telemetry features: "
        f"{len(telemetry_found)}"
    )

    if len(categorical_found) != 6:
        raise ValueError(
            "Expected 6 categorical features."
        )

    if len(meter_found) != 5:
        raise ValueError(
            "Expected 5 meter features."
        )

    if len(telemetry_found) != 40:
        raise ValueError(
            "Expected 40 telemetry features."
        )

    return challenge


# =====================================================================
# VALIDATE
# =====================================================================

def validate(
    challenge,
    gateway_ids,
):

    print()
    print("=" * 70)
    print("9. VALIDATING CHALLENGE DATASET")
    print("=" * 70)

    expected_rows = (
        len(gateway_ids)
        *
        len(CHALLENGE_WEEKS)
    )

    # -------------------------------------------------------------
    # Row count
    # -------------------------------------------------------------

    if len(challenge) != expected_rows:

        raise ValueError(
            f"Expected {expected_rows} rows, "
            f"got {len(challenge)}."
        )

    # -------------------------------------------------------------
    # Gateway count
    # -------------------------------------------------------------

    gateway_count = (
        challenge["gateway_id"]
        .nunique()
    )

    if gateway_count != len(gateway_ids):

        raise ValueError(
            f"Expected {len(gateway_ids)} gateways, "
            f"got {gateway_count}."
        )

    # -------------------------------------------------------------
    # Week count
    # -------------------------------------------------------------

    week_count = (
        challenge["week_start"]
        .nunique()
    )

    if week_count != len(CHALLENGE_WEEKS):

        raise ValueError(
            f"Expected {len(CHALLENGE_WEEKS)} weeks, "
            f"got {week_count}."
        )

    # -------------------------------------------------------------
    # Duplicate gateway/week.
    # -------------------------------------------------------------

    duplicates = (
        challenge
        .duplicated(
            [
                "gateway_id",
                "week_start",
            ]
        )
        .sum()
    )

    if duplicates:

        raise ValueError(
            f"Found {duplicates} duplicate "
            "gateway/week rows."
        )

    # -------------------------------------------------------------
    # Rows per week.
    # -------------------------------------------------------------

    rows_per_week = (
        challenge
        .groupby("week_start")
        .size()
    )

    if not (
        rows_per_week == len(gateway_ids)
    ).all():

        raise ValueError(
            "Not every challenge week contains "
            "all gateways."
        )

    # -------------------------------------------------------------
    # Model feature count.
    # -------------------------------------------------------------

    model_features = [
        column
        for column in challenge.columns
        if column not in {
            "gateway_id",
            "week_start",
        }
    ]

    if len(model_features) != 51:

        raise ValueError(
            f"Expected 51 model features, "
            f"got {len(model_features)}."
        )

    # -------------------------------------------------------------
    # Target must NOT be present.
    # -------------------------------------------------------------

    if "fault_next_7d" in challenge.columns:

        raise ValueError(
            "fault_next_7d must not be present "
            "in challenge inference data."
        )

    # -------------------------------------------------------------
    # Metadata completeness.
    # -------------------------------------------------------------

    for column in CATEGORICAL_FEATURES:

        missing = (
            challenge[column]
            .isna()
            .sum()
        )

        if missing:

            raise ValueError(
                f"{column} contains "
                f"{missing} missing values."
            )

    # -------------------------------------------------------------
    # Print summary.
    # -------------------------------------------------------------

    print(
        f"Rows: "
        f"{len(challenge):,}"
    )

    print(
        f"Gateways: "
        f"{gateway_count}"
    )

    print(
        f"Weeks: "
        f"{week_count}"
    )

    print(
        f"Model features: "
        f"{len(model_features)}"
    )

    print(
        f"Duplicate gateway/week rows: "
        f"{duplicates}"
    )

    print()
    print("Rows per week:")

    print(rows_per_week)

    print()
    print(
        "[PASS] Challenge dataset validation complete."
    )


# =====================================================================
# MAIN
# =====================================================================

def main():

    print()
    print("=" * 70)
    print("STEP 03 - BUILD CHALLENGE FEATURES")
    print("=" * 70)

    # -------------------------------------------------------------
    # 1. Gateway master
    # -------------------------------------------------------------

    master, gateway_ids = load_master()

    # -------------------------------------------------------------
    # 2. Telemetry
    # -------------------------------------------------------------

    telemetry = load_telemetry()

    # -------------------------------------------------------------
    # 3. Derived telemetry
    # -------------------------------------------------------------

    telemetry = create_derived_features(
        telemetry
    )

    # -------------------------------------------------------------
    # 4. Challenge telemetry features
    # -------------------------------------------------------------

    telemetry_features = build_features(
        gateway_ids,
        telemetry,
    )

    print()
    print(
        f"Telemetry feature shape: "
        f"{telemetry_features.shape}"
    )

    # -------------------------------------------------------------
    # 5. Meter data
    # -------------------------------------------------------------

    meter = load_meter()

    # -------------------------------------------------------------
    # Complete gateway-week grid.
    # -------------------------------------------------------------

    gateway_week_grid = (
        pd.MultiIndex.from_product(
            [
                gateway_ids,
                CHALLENGE_WEEKS,
            ],
            names=[
                "gateway_id",
                "week_start",
            ],
        )
        .to_frame(
            index=False
        )
    )

    print()
    print(
        f"Gateway-week grid: "
        f"{gateway_week_grid.shape}"
    )

    # -------------------------------------------------------------
    # 6. Meter features.
    # -------------------------------------------------------------

    meter_features = build_meter_features(
        meter,
        gateway_week_grid[
            [
                "gateway_id",
                "week_start",
            ]
        ],
    )

    # -------------------------------------------------------------
    # Merge telemetry and meter.
    # -------------------------------------------------------------

    challenge = telemetry_features.merge(
        meter_features,
        on=[
            "gateway_id",
            "week_start",
        ],
        how="left",
        validate="one_to_one",
    )

    print()
    print(
        f"After telemetry + meter merge: "
        f"{challenge.shape}"
    )

    # -------------------------------------------------------------
    # 7. Metadata.
    # -------------------------------------------------------------

    challenge = add_gateway_metadata(
        challenge,
        master,
    )

    print()
    print(
        f"After metadata merge: "
        f"{challenge.shape}"
    )

    # -------------------------------------------------------------
    # 8. Match exact model schema.
    # -------------------------------------------------------------

    challenge = match_model_schema(
        challenge
    )

    # -------------------------------------------------------------
    # 9. Validate.
    # -------------------------------------------------------------

    validate(
        challenge,
        gateway_ids,
    )

    # -------------------------------------------------------------
    # Save.
    # -------------------------------------------------------------

    challenge.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # -------------------------------------------------------------
    # Final output.
    # -------------------------------------------------------------

    print()
    print("=" * 70)
    print("STEP 03 COMPLETE")
    print("=" * 70)

    print(
        f"Final shape: "
        f"{challenge.shape}"
    )

    print(
        f"Saved: "
        f"{OUTPUT_FILE.relative_to(ROOT)}"
    )

    print()
    print(
        "Challenge features are ready for Step 04."
    )


# =====================================================================
# ENTRY POINT
# =====================================================================

if __name__ == "__main__":
    main()