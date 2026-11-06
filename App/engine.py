from __future__ import annotations

from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
import streamlit as st


MODEL_FEATURES = [
    "Global_reactive_power",
    "Voltage",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",
    "Year",
    "Month",
    "Day",
    "Hour",
    "Minute",
    "Is_Weekend",
    "Peak_Hour",
    "Rolling_Mean_60",
    "Rolling_STD_60",
    "Rolling_Max_60",
    "Rolling_Min_60",
    "Lag_1",
    "Lag_60",
    "Lag_1440",
    "Time_Period_Evening",
    "Time_Period_Morning",
    "Time_Period_Night",
    "Weekday_Monday",
    "Weekday_Saturday",
    "Weekday_Sunday",
    "Weekday_Thursday",
    "Weekday_Tuesday",
    "Weekday_Wednesday",
]

NUMERIC_COLUMNS = [
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",
]

RAW_REQUIRED_COLUMNS = [
    "Date",
    "Time",
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",
]


@st.cache_resource(show_spinner=False)
def load_model(model_path: str):
    return joblib.load(model_path)


@st.cache_data(show_spinner="Loading dataset...")
def load_raw_data(data_path: str) -> pd.DataFrame:
    return pd.read_csv(data_path, sep=";", low_memory=False)


def validate_raw_columns(data: pd.DataFrame) -> list[str]:
    return [column for column in RAW_REQUIRED_COLUMNS if column not in data.columns]


@st.cache_data(show_spinner="Preparing analytics...")
def prepare_analytics_data(data_path: str) -> pd.DataFrame:
    data = pd.read_csv(data_path, sep=";", low_memory=False)

    for column in NUMERIC_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data["Datetime"] = pd.to_datetime(
        data["Date"].astype(str) + " " + data["Time"].astype(str),
        dayfirst=True,
        errors="coerce",
    )

    return data.dropna(subset=["Datetime", "Global_active_power"])


@st.cache_data(show_spinner=False)
def build_analytics_tables(data_path: str) -> dict[str, pd.DataFrame]:
    data = prepare_analytics_data(data_path).copy()
    data["Hour"] = data["Datetime"].dt.hour
    data["Month"] = data["Datetime"].dt.month
    data["Weekday"] = data["Datetime"].dt.day_name()
    data["Is_Weekend"] = (data["Datetime"].dt.dayofweek >= 5).astype(int)

    weekday_order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]

    hourly = data.groupby("Hour", as_index=False)["Global_active_power"].mean()
    monthly = data.groupby("Month", as_index=False)["Global_active_power"].mean()
    weekday = (
        data.groupby("Weekday")["Global_active_power"]
        .mean()
        .reindex(weekday_order)
        .reset_index()
    )
    weekend = data.groupby("Is_Weekend", as_index=False)["Global_active_power"].mean()
    daily = (
        data.set_index("Datetime")["Global_active_power"]
        .resample("D")
        .mean()
        .dropna()
        .reset_index()
    )
    daily["Rolling_30_Day"] = daily["Global_active_power"].rolling(30).mean()

    return {
        "hourly": hourly,
        "monthly": monthly,
        "weekday": weekday,
        "weekend": weekend,
        "daily": daily,
    }


def _prepare_model_data_from_frame(raw_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    missing = validate_raw_columns(raw_data)
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))

    data = raw_data.copy()

    for column in NUMERIC_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data["Datetime"] = pd.to_datetime(
        data["Date"].astype(str) + " " + data["Time"].astype(str),
        dayfirst=True,
        errors="coerce",
    )

    data = data.sort_values("Datetime").reset_index(drop=True)

    data["Year"] = data["Datetime"].dt.year
    data["Month"] = data["Datetime"].dt.month
    data["Day"] = data["Datetime"].dt.day
    data["Hour"] = data["Datetime"].dt.hour
    data["Minute"] = data["Datetime"].dt.minute
    data["Weekday"] = data["Datetime"].dt.day_name()
    data["Is_Weekend"] = (data["Datetime"].dt.dayofweek >= 5).astype(int)
    data["Peak_Hour"] = data["Hour"].between(18, 22).astype(int)

    conditions = [
        data["Hour"].between(5, 11),
        data["Hour"].between(12, 16),
        data["Hour"].between(17, 21),
    ]
    data["Time_Period"] = np.select(
        conditions,
        ["Morning", "Afternoon", "Evening"],
        default="Night",
    )

    data["Lag_1"] = data["Global_active_power"].shift(1)
    data["Lag_60"] = data["Global_active_power"].shift(60)
    data["Lag_1440"] = data["Global_active_power"].shift(1440)

    shifted = data["Global_active_power"].shift(1)
    rolling = shifted.rolling(window=60)
    data["Rolling_Mean_60"] = rolling.mean()
    data["Rolling_STD_60"] = rolling.std()
    data["Rolling_Max_60"] = rolling.max()
    data["Rolling_Min_60"] = rolling.min()

    data = pd.get_dummies(
        data,
        columns=["Time_Period", "Weekday"],
        dtype=int,
    )

    for feature in MODEL_FEATURES:
        if feature not in data.columns:
            data[feature] = 0

    features = data[MODEL_FEATURES].copy()
    valid_mask = (
        features.notna().all(axis=1)
        & data["Global_active_power"].notna()
        & data["Datetime"].notna()
    )

    features = features.loc[valid_mask].reset_index(drop=True)
    metadata = data.loc[
        valid_mask,
        ["Datetime", "Global_active_power"],
    ].reset_index(drop=True)

    return features, metadata


@st.cache_data(show_spinner="Preparing model features...")
def prepare_model_data(data_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    raw_data = pd.read_csv(data_path, sep=";", low_memory=False)
    return _prepare_model_data_from_frame(raw_data)


def prepare_uploaded_model_data(raw_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    return _prepare_model_data_from_frame(raw_data)


def consumption_status(value: float, high_threshold: float = 1.528, critical_threshold: float = 3.0) -> str:
    if value < high_threshold:
        return "🟢 Normal"
    if value < critical_threshold:
        return "🟡 High"
    return "🔴 Critical"


def model_path_exists(path: str) -> bool:
    return Path(path).is_file()
