#!/usr/bin/env python3
"""
STEP 07 - RUN FINAL MODEL ON NEW / UNSEEN MONTH DATA

Purpose
-------
Run the frozen gateway-ranking pipeline on a newly supplied telemetry period.

The script:
1. Loads the gateway master and all available telemetry parquet files.
2. Automatically determines the newest complete calendar month in telemetry.
3. Builds cutoff-safe features for each Monday scoring week whose 7-day
   scoring window lies inside that newest month.
4. Loads the frozen HGB weight=2 model.
5. Reproduces the 3-sigma anomaly signal directly from telemetry.
6. Combines 75% ML percentile + 25% baseline percentile.
7. Selects and ranks the top 15 gateways for each scoring week.
8. Writes a submission-style CSV.

Important
---------
This script is for NEW INFERENCE DATA only. It does not create labels and
does not use future data relative to a scoring cutoff.

Expected telemetry layout:
    data/telemetry/month=*/part-0.parquet

Expected model:
    models/gateway_fault_model_weight2.joblib

Output:
    outputs/new_month_predictions.csv
"""

from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
TELEMETRY_DIR = DATA_DIR / "telemetry"
MASTER_FILE = DATA_DIR / "gateway_master.csv"
MODEL_FILE = ROOT / "models" / "gateway_fault_model_weight2.joblib"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_FILE = OUTPUT_DIR / "new_month_predictions.csv"

WEEKDAY_MONDAY = 0
TOP_K = 15
ML_WEIGHT = 0.75
BASELINE_WEIGHT = 0.25


def normalize_gateway_id(value):
    if pd.isna(value):
        return np.nan
    return (
        str(value)
        .strip()
        .upper()
        .replace(":", "")
        .replace("-", "")
        .replace(" ", "")
    )


def find_column(df, candidates):
    lower = {str(c).lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in lower:
            return lower[candidate.lower()]
    return None


def load_master():
    master = pd.read_csv(MASTER_FILE, encoding="latin1")
    gateway_col = find_column(master, ["gateway_id", "gatewayid", "gateway"])
    if gateway_col is None:
        raise ValueError("gateway_master.csv does not contain a gateway ID column.")

    master["gateway_id"] = master[gateway_col].map(normalize_gateway_id)
    master = master.dropna(subset=["gateway_id"]).drop_duplicates("gateway_id")

    print(f"Master gateways : {len(master)}")
    return master


def find_telemetry_files():
    files = sorted(TELEMETRY_DIR.glob("month=*/part-*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"No telemetry parquet files found under {TELEMETRY_DIR}"
        )
    return files


def inspect_newest_month(files):
    """
    Read only the timestamp/gateway columns first so the newest month can be
    identified without loading all telemetry columns into memory.
    """
    frames = []

    for file in files:
        try:
            sample = pd.read_parquet(file)
        except Exception as exc:
            raise RuntimeError(f"Could not read {file}: {exc}") from exc

        date_col = find_column(
            sample,
            ["ts_utc", "timestamp", "datetime", "time", "ts", "event_time"],
        )
        gateway_col = find_column(
            sample,
            ["gateway_id", "gatewayid", "gateway"],
        )

        if date_col is None or gateway_col is None:
            raise ValueError(
                f"{file} must contain timestamp and gateway ID columns."
            )

        temp = sample[[date_col, gateway_col]].copy()
        temp.columns = ["timestamp", "gateway_id"]
        temp["timestamp"] = pd.to_datetime(
            temp["timestamp"], errors="coerce", utc=True
        )
        temp["gateway_id"] = temp["gateway_id"].map(normalize_gateway_id)
        temp = temp.dropna(subset=["timestamp", "gateway_id"])

        frames.append(temp)

    meta = pd.concat(frames, ignore_index=True)

    if meta.empty:
        raise ValueError("No valid telemetry timestamps were found.")

    meta["month"] = meta["timestamp"].dt.to_period("M").astype(str)
    newest_month = meta["month"].max()

    month_meta = meta[meta["month"] == newest_month].copy()

    print(f"Telemetry rows scanned : {len(meta):,}")
    print(f"Newest telemetry month : {newest_month}")
    print(
        f"Newest month range     : "
        f"{month_meta['timestamp'].min()} -> {month_meta['timestamp'].max()}"
    )

    return newest_month


def load_month_telemetry(files, month):
    """
    Load the detected month plus the preceding 28 days.

    The model uses 7/14/28-day cutoff-safe telemetry features. Therefore a
    new month's scoring week needs telemetry from the previous calendar month
    as well. Loading only the new month would make early-month 28-day features
    incomplete.
    """
    month_start = pd.Timestamp(f"{month}-01", tz="UTC")
    lookback_start = month_start - pd.Timedelta(days=28)
    month_end = month_start + pd.offsets.MonthBegin(1)

    frames = []

    for file in files:
        df = pd.read_parquet(file)

        date_col = find_column(
            df,
            ["ts_utc", "timestamp", "datetime", "time", "ts", "event_time"],
        )
        gateway_col = find_column(
            df,
            ["gateway_id", "gatewayid", "gateway"],
        )

        if date_col is None or gateway_col is None:
            continue

        df = df.rename(
            columns={
                date_col: "timestamp",
                gateway_col: "gateway_id",
            }
        )

        df["timestamp"] = pd.to_datetime(
            df["timestamp"], errors="coerce", utc=True
        )
        df["gateway_id"] = df["gateway_id"].map(normalize_gateway_id)

        df = df.dropna(subset=["timestamp", "gateway_id"])

        # Include the newest month and exactly the preceding 28 days.
        df = df[
            (df["timestamp"] >= lookback_start)
            & (df["timestamp"] < month_end)
        ]

        if not df.empty:
            frames.append(df)

    if not frames:
        raise ValueError(
            f"No telemetry rows found for {month} or its 28-day lookback."
        )

    telemetry = pd.concat(frames, ignore_index=True)

    new_month_rows = telemetry[
        (telemetry["timestamp"] >= month_start)
        & (telemetry["timestamp"] < month_end)
    ]

    print(f"New-month telemetry rows : {len(new_month_rows):,}")
    print(f"Lookback + month rows    : {len(telemetry):,}")
    print(f"New-month gateways       : {new_month_rows['gateway_id'].nunique()}")
    print(
        f"Feature telemetry range  : "
        f"{telemetry['timestamp'].min()} -> {telemetry['timestamp'].max()}"
    )

    return telemetry


def numeric_column(df, names):
    for name in names:
        if name in df.columns:
            return name
    return None


def make_telemetry_features(telemetry, gateway_ids, scoring_week):
    """
    Construct the same telemetry feature families used by the final model.

    Only observations strictly before scoring_week are used.
    """
    cutoff = pd.Timestamp(scoring_week)

    t = telemetry[
        (telemetry["timestamp"] < cutoff)
        & (telemetry["gateway_id"].isin(gateway_ids))
    ].copy()

    # The new month may be short at its beginning. The model therefore uses
    # whatever pre-cutoff observations exist in the available telemetry.
    feature_specs = {
        "offline_duration_sec": ["offline_duration_sec"],
        "disconnection_cnt": ["disconnection_cnt"],
        "reboot_cnt": ["reboot_cnt"],
        "tx_busy": ["tx_busy"],
    }

    result = pd.DataFrame({"gateway_id": gateway_ids})

    for output_name, candidates in feature_specs.items():
        source = numeric_column(t, candidates)

        if source is None:
            # Preserve model schema if a source field is absent.
            values = pd.DataFrame(
                {"gateway_id": gateway_ids, output_name: np.nan}
            )
        else:
            values = t[["gateway_id", "timestamp", source]].copy()
            values[source] = pd.to_numeric(values[source], errors="coerce")
            values[output_name] = values[source]
            values = values[["gateway_id", "timestamp", output_name]]

        for days in [7, 14, 28]:
            start = cutoff - pd.Timedelta(days=days)

            if source is None:
                agg = pd.DataFrame(
                    {
                        "gateway_id": gateway_ids,
                        f"{output_name}_{days}d_mean": np.nan,
                        f"{output_name}_{days}d_sum": np.nan,
                        f"{output_name}_{days}d_max": np.nan,
                        f"{output_name}_{days}d_std": np.nan,
                    }
                )
            else:
                window = values[
                    (values["timestamp"] >= start)
                    & (values["timestamp"] < cutoff)
                ]

                grouped = window.groupby("gateway_id")[output_name]

                # Reindex each statistic to ALL master gateways. A gateway
                # may have no observations in a particular lookback window.
                # In that case its feature must be NaN, not a shorter Series.
                agg = pd.DataFrame(
                    {
                        "gateway_id": gateway_ids,
                        f"{output_name}_{days}d_mean":
                            grouped.mean().reindex(gateway_ids).to_numpy(),
                        f"{output_name}_{days}d_sum":
                            grouped.sum().reindex(gateway_ids).to_numpy(),
                        f"{output_name}_{days}d_max":
                            grouped.max().reindex(gateway_ids).to_numpy(),
                        f"{output_name}_{days}d_std":
                            grouped.std().reindex(gateway_ids).to_numpy(),
                    }
                )

            result = result.merge(agg, on="gateway_id", how="left")

        # Seven-day trend: recent 7d mean minus preceding 7d mean.
        if source is None:
            result[f"{output_name}_7d_trend"] = np.nan
        else:
            recent_start = cutoff - pd.Timedelta(days=7)
            previous_start = cutoff - pd.Timedelta(days=14)

            recent = values[
                (values["timestamp"] >= recent_start)
                & (values["timestamp"] < cutoff)
            ].groupby("gateway_id")[output_name].mean()

            previous = values[
                (values["timestamp"] >= previous_start)
                & (values["timestamp"] < recent_start)
            ].groupby("gateway_id")[output_name].mean()

            trend = (
                recent.reindex(gateway_ids)
                - previous.reindex(gateway_ids)
            )
            trend = pd.DataFrame(
                {
                    "gateway_id": gateway_ids,
                    f"{output_name}_7d_trend": trend.to_numpy(),
                }
            )

            result = result.merge(trend, on="gateway_id", how="left")

    return result


def build_baseline_scores(telemetry, gateway_ids, scoring_week):
    """
    Exact Part 1 / supplied 3-sigma baseline, adapted for arbitrary scoring
    Mondays.

    For each gateway:
      1. trailing 28 days strictly before Monday -> mean/std
      2. trailing 7 days -> count hourly metric breaches
      3. breach if value - gateway mean > 3 * gateway std
      4. total flagged hours is the baseline score

    Metrics:
      offline_duration_sec
      disconnection_cnt
      reboot_cnt
    """
    metrics = [
        "offline_duration_sec",
        "disconnection_cnt",
        "reboot_cnt",
    ]

    end = pd.Timestamp(scoring_week)
    if end.tzinfo is None:
        end = end.tz_localize("UTC")
    else:
        end = end.tz_convert("UTC")

    window = telemetry[
        (telemetry["timestamp"] >= end - pd.Timedelta(days=28))
        & (telemetry["timestamp"] < end)
        & (telemetry["gateway_id"].isin(gateway_ids))
    ].copy()

    if window.empty:
        return pd.DataFrame(
            {
                "gateway_id": gateway_ids,
                "baseline_score": 0.0,
            }
        )

    stats = window.groupby("gateway_id")[metrics].agg(["mean", "std"])

    recent = window[
        window["timestamp"] >= end - pd.Timedelta(days=7)
    ].copy()

    # Start with zero for every recent telemetry row.
    flags = pd.Series(0, index=recent.index, dtype=int)

    for metric in metrics:
        mean = recent["gateway_id"].map(stats[(metric, "mean")])
        std = (
            recent["gateway_id"]
            .map(stats[(metric, "std")])
            .replace(0, np.nan)
        )

        exceeded = (recent[metric] - mean) > 3.0 * std
        exceeded = exceeded.fillna(False)

        flags = flags + exceeded.astype(int)

    recent["flagged"] = flags

    grouped = recent.groupby("gateway_id").agg(
        flagged_hours=("flagged", "sum")
    )

    # The supplied baseline ranks only gateways that have telemetry in the
    # recent window. Gateways with no recent observations receive zero when
    # we align the result to the complete master gateway universe.
    scores = (
        grouped["flagged_hours"]
        .reindex(gateway_ids)
        .fillna(0.0)
    )

    return pd.DataFrame(
        {
            "gateway_id": gateway_ids,
            "baseline_score": scores.to_numpy(dtype=float),
        }
    )

def encode_features(df, bundle):
    feature_cols = bundle["feature_cols"]
    categorical_cols = bundle["categorical_cols"]
    numeric_cols = bundle["numeric_cols"]

    X = df.copy()

    for col in feature_cols:
        if col not in X.columns:
            X[col] = np.nan

    X = X[feature_cols].copy()

    for col in categorical_cols:
        X[col] = X[col].astype("category").cat.codes.astype(float)

    for col in numeric_cols:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    return X


def load_metadata(master):
    """
    Supply the six categorical model features from gateway_master.csv.
    """
    column_map = {}

    for target in [
        "tenant",
        "site_type",
        "region",
        "hw_model",
        "antenna_type",
        "fw_version",
    ]:
        source = find_column(master, [target])
        if source is None:
            raise ValueError(
                f"gateway_master.csv is missing required model column: {target}"
            )
        column_map[target] = source

    metadata = master[["gateway_id"] + list(column_map.values())].copy()
    metadata = metadata.rename(
        columns={source: target for target, source in column_map.items()}
    )

    return metadata


def generate_week_predictions(
    telemetry,
    master,
    bundle,
    scoring_week,
):
    gateway_ids = master["gateway_id"].tolist()

    features = make_telemetry_features(
        telemetry,
        gateway_ids,
        scoring_week,
    )

    metadata = load_metadata(master)
    features = features.merge(metadata, on="gateway_id", how="left")

    X = encode_features(features, bundle)
    ml_probability = bundle["model"].predict_proba(X)[:, 1]

    baseline = build_baseline_scores(
        telemetry,
        gateway_ids,
        scoring_week,
    )

    result = pd.DataFrame(
        {
            "gateway_id": gateway_ids,
            "ml_probability": ml_probability,
        }
    ).merge(baseline, on="gateway_id", how="left")

    result["ml_percentile"] = result["ml_probability"].rank(
        method="average",
        pct=True,
    )

    result["baseline_percentile"] = result["baseline_score"].rank(
        method="average",
        pct=True,
    )

    result["hybrid_score"] = (
        ML_WEIGHT * result["ml_percentile"]
        + BASELINE_WEIGHT * result["baseline_percentile"]
    )

    result = result.sort_values(
        [
            "hybrid_score",
            "ml_probability",
            "baseline_score",
            "gateway_id",
        ],
        ascending=[False, False, False, True],
    ).head(TOP_K).copy()

    result["rank"] = np.arange(1, len(result) + 1)
    result["week_start"] = pd.Timestamp(scoring_week).strftime("%Y-%m-%d")

    return result[
        [
            "week_start",
            "rank",
            "gateway_id",
            "hybrid_score",
            "ml_probability",
            "baseline_score",
        ]
    ]


def main():
    print("=" * 75)
    print("STEP 07 - NEW MONTH INFERENCE")
    print("=" * 75)

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Frozen model not found: {MODEL_FILE}"
        )

    files = find_telemetry_files()
    master = load_master()
    newest_month = inspect_newest_month(files)
    telemetry = load_month_telemetry(files, newest_month)

    bundle = joblib.load(MODEL_FILE)

    print("\nModel loaded")
    print(f"Model features : {len(bundle['feature_cols'])}")
    print(f"ML weight      : {ML_WEIGHT}")
    print(f"Baseline weight: {BASELINE_WEIGHT}")

    # Generate Monday scoring weeks whose 7-day forward result window fits
    # completely inside the detected month.
    month_start = pd.Timestamp(f"{newest_month}-01", tz="UTC")
    month_end = month_start + pd.offsets.MonthBegin(1)

    # First Monday on/after the beginning of the month.
    first_monday = month_start + pd.Timedelta(
        days=(WEEKDAY_MONDAY - month_start.weekday()) % 7
    )

    weeks = []
    current = first_monday

    while current + pd.Timedelta(days=7) <= month_end:
        # Need sufficient pre-cutoff telemetry for the model's 28-day
        # lookback. The script still permits missing history as NaN.
        weeks.append(current)
        current += pd.Timedelta(days=7)

    if not weeks:
        raise ValueError(
            f"No complete Monday scoring weeks found in {newest_month}."
        )

    print(f"\nScoring weeks in newest month : {len(weeks)}")
    for week in weeks:
        print(f"  {week.date()}")

    all_results = []

    for week in weeks:
        print(f"\nProcessing {week.date()}")

        result = generate_week_predictions(
            telemetry,
            master,
            bundle,
            week,
        )

        print(
            f"  Selected : {len(result)} gateways | "
            f"score range "
            f"{result['hybrid_score'].min():.6f} -> "
            f"{result['hybrid_score'].max():.6f}"
        )

        all_results.append(result)

    final = pd.concat(all_results, ignore_index=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    final.to_csv(OUTPUT_FILE, index=False)

    print("\n" + "=" * 75)
    print("NEW MONTH INFERENCE COMPLETE")
    print("=" * 75)
    print(f"Month           : {newest_month}")
    print(f"Weeks processed  : {final['week_start'].nunique()}")
    print(f"Rows             : {len(final)}")
    print(
        f"Rows per week    : "
        f"{final.groupby('week_start').size().unique().tolist()}"
    )
    print(f"Output           : {OUTPUT_FILE}")
    print("\nNote: no future labels or challenge labels are used.")


if __name__ == "__main__":
    main()
