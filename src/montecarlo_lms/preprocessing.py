import pandas as pd
import numpy as np
from datetime import datetime

def preprocess_data(train_df, building_df, weather_df, selected_site_id=None):
    """
    Preprocess the data: select site, merge datasets, handle missing values,
    feature engineering, and split into train/validation.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training data.
    building_df : pd.DataFrame
        Building metadata.
    weather_df : pd.DataFrame
        Weather data.
    selected_site_id : int, optional
        Pre-selected site ID. If None, select automatically.

    Returns
    -------
    X_train_norm : np.ndarray
        Normalized training features.
    y_train : np.ndarray
        Training targets.
    X_val_norm : np.ndarray
        Normalized validation features.
    y_val : np.ndarray
        Validation targets.
    FEATURES : list
        List of feature names.
    TARGET : str
        Target name.
    X_mean : np.ndarray
        Feature means for normalization.
    X_std : np.ndarray
        Feature stds for normalization.
    site_id : int
        Selected site ID.
    """
    # Timestamp parsing
    train_df["timestamp"] = pd.to_datetime(train_df["timestamp"])
    weather_df["timestamp"] = pd.to_datetime(weather_df["timestamp"])

    if selected_site_id is None:
        # Site selection logic
        site_counts = (
            train_df.merge(building_df[["building_id", "site_id"]], on="building_id", how="left")
            ["site_id"]
            .value_counts()
            .sort_index()
        )
        weather_site_counts = weather_df["site_id"].value_counts().sort_index()
        site_summary = pd.DataFrame({
            "train_records": site_counts,
            "weather_records": weather_site_counts
        }).dropna()

        MIN_TRAIN_RECORDS = site_summary["train_records"].quantile(0.5)
        MIN_WEATHER_RECORDS = site_summary["weather_records"].quantile(0.5)
        site_summary_filtered = site_summary[
            (site_summary["train_records"] >= MIN_TRAIN_RECORDS) &
            (site_summary["weather_records"] >= MIN_WEATHER_RECORDS)
        ]
        site_summary_filtered = site_summary_filtered.copy()
        site_summary_filtered.loc[:, "coverage_ratio"] = (
            site_summary_filtered["train_records"] / site_summary_filtered["weather_records"]
        )
        site_summary_filtered.loc[:, "balanced_score"] = np.abs(np.log(site_summary_filtered["coverage_ratio"]))
        selected_site_id = site_summary_filtered.sort_values(
            by=["balanced_score", "train_records"],
            ascending=[True, False]
        ).index[0]

    site_id = int(selected_site_id)

    # Restrict to electricity meter
    train_site_df = train_df[train_df["meter"] == 0].copy()

    # Join building metadata
    building_features = [
        "building_id",
        "site_id",
        "square_feet",
        "year_built",
        "primary_use"
    ]
    train_site_df = train_site_df.merge(
        building_df[building_features],
        on="building_id",
        how="left"
    )

    # Join weather data
    weather_features = [
        "site_id",
        "timestamp",
        "air_temperature",
        "dew_temperature",
        "wind_speed"
    ]
    weather_site_df = weather_df[weather_features].copy()

    data_df = train_site_df.merge(
        weather_site_df,
        on=["site_id", "timestamp"],
        how="left"
    )

    # Handle missing values
    required_cols = [
        "meter_reading",
        "air_temperature",
        "dew_temperature",
        "wind_speed",
        "square_feet"
    ]
    data_df = data_df.dropna(subset=required_cols).copy()

    # Feature transformations
    data_df["log_meter_reading"] = np.log1p(data_df["meter_reading"])
    data_df["log_square_feet"] = np.log1p(data_df["square_feet"])

    # Time-based features
    data_df["hour"] = data_df["timestamp"].dt.hour
    data_df["dayofweek"] = data_df["timestamp"].dt.dayofweek

    # Final feature set
    FEATURES = [
        "air_temperature",
        "dew_temperature",
        "wind_speed",
        "hour",
        "dayofweek",
        "log_square_feet"
    ]
    TARGET = "log_meter_reading"

    X = data_df[FEATURES].values
    y = data_df[TARGET].values

    # Time-aware train/validation split
    data_df = data_df.sort_values("timestamp")
    split_ratio = 0.8
    split_idx = int(len(data_df) * split_ratio)

    X_train = X[:split_idx]
    y_train = y[:split_idx]
    X_val = X[split_idx:]
    y_val = y[split_idx:]

    # Feature normalization
    X_mean = X_train.mean(axis=0)
    X_std = X_train.std(axis=0) + 1e-8

    X_train_norm = (X_train - X_mean) / X_std
    X_val_norm = (X_val - X_mean) / X_std

    return X_train_norm, y_train, X_val_norm, y_val, FEATURES, TARGET, X_mean, X_std, site_id