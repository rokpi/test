from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

DEFAULT_FEATURE_COLUMNS = [
    "hour",
    "day_of_week",
    "avg_last_7_days",
    "avg_last_28_days",
    "same_dow_last_week",
]
DEFAULT_MODEL_TYPE = "hist_gradient_boosting"

CACHE_DIR = Path(__file__).resolve().parents[1] / "cache" / "models"


@dataclass(frozen=True)
class ModelMetadata:
    feature_columns: list[str]
    target_columns: list[str]
    params: dict[str, Any]
    h3_res: int
    date_range: tuple[date, date]
    model_type: str


@dataclass(frozen=True)
class ModelPayload:
    model: MultiOutputRegressor
    metadata: ModelMetadata
    model_hash: str


def aggregate_incidents(df: pd.DataFrame) -> pd.DataFrame:
    required_columns = {"cell_id", "date", "hour", "incident_type"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    filtered = df.dropna(subset=["cell_id", "date", "hour"]).copy()
    grouped = (
        filtered.groupby(["cell_id", "date", "hour", "incident_type"], dropna=False)
        .size()
        .reset_index(name="count")
    )
    pivot = grouped.pivot_table(
        index=["cell_id", "date", "hour"],
        columns="incident_type",
        values="count",
        fill_value=0,
        aggfunc="sum",
    ).reset_index()
    pivot.columns.name = None
    target_columns = [col for col in pivot.columns if col not in {"cell_id", "date", "hour"}]
    pivot["total_count"] = pivot[target_columns].sum(axis=1)
    return pivot


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    features = df.copy()
    features["date"] = pd.to_datetime(features["date"])
    features = features.sort_values(["cell_id", "hour", "date"])
    grouped = features.groupby(["cell_id", "hour"], sort=False)

    features["avg_last_7_days"] = grouped["total_count"].transform(
        lambda series: series.shift(1).rolling(7, min_periods=1).mean()
    )
    features["avg_last_28_days"] = grouped["total_count"].transform(
        lambda series: series.shift(1).rolling(28, min_periods=1).mean()
    )

    lookup = features[["cell_id", "hour", "date", "total_count"]].copy()
    lookup["date"] = lookup["date"] + pd.Timedelta(days=7)
    lookup = lookup.rename(columns={"total_count": "same_dow_last_week"})

    features = features.merge(lookup, on=["cell_id", "hour", "date"], how="left")
    features["avg_last_7_days"] = features["avg_last_7_days"].fillna(0)
    features["avg_last_28_days"] = features["avg_last_28_days"].fillna(0)
    features["same_dow_last_week"] = features["same_dow_last_week"].fillna(0)
    features["day_of_week"] = features["date"].dt.dayofweek
    features["date"] = features["date"].dt.date
    return features


def build_feature_target_frames(
    df: pd.DataFrame, feature_columns: list[str] | None = None
) -> tuple[pd.DataFrame, list[str]]:
    feature_columns = feature_columns or DEFAULT_FEATURE_COLUMNS
    aggregated = aggregate_incidents(df)
    features = add_rolling_features(aggregated)
    target_columns = [
        col
        for col in aggregated.columns
        if col not in {"cell_id", "date", "hour", "total_count"}
    ]
    return features, target_columns


def build_model(model_type: str, params: dict[str, Any] | None = None) -> MultiOutputRegressor:
    params = params or {}
    if model_type == "hist_gradient_boosting":
        base_model = HistGradientBoostingRegressor(**params)
    elif model_type == "random_forest":
        base_model = RandomForestRegressor(**params)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    return MultiOutputRegressor(base_model)


def _metadata_hash(metadata: ModelMetadata) -> str:
    payload = {
        "feature_columns": metadata.feature_columns,
        "target_columns": metadata.target_columns,
        "params": metadata.params,
        "h3_res": metadata.h3_res,
        "date_range": [metadata.date_range[0].isoformat(), metadata.date_range[1].isoformat()],
        "model_type": metadata.model_type,
    }
    return sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def persist_model(payload: ModelPayload) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"model_{payload.model_hash}.joblib"
    joblib.dump(
        {
            "model": payload.model,
            "metadata": payload.metadata,
            "model_hash": payload.model_hash,
        },
        path,
    )
    return path


def load_model(model_hash: str) -> ModelPayload:
    path = CACHE_DIR / f"model_{model_hash}.joblib"
    stored = joblib.load(path)
    return ModelPayload(
        model=stored["model"],
        metadata=stored["metadata"],
        model_hash=stored["model_hash"],
    )


def train_model(
    df: pd.DataFrame,
    h3_res: int,
    model_type: str = DEFAULT_MODEL_TYPE,
    params: dict[str, Any] | None = None,
    feature_columns: list[str] | None = None,
) -> ModelPayload:
    feature_columns = feature_columns or DEFAULT_FEATURE_COLUMNS
    features, target_columns = build_feature_target_frames(df, feature_columns)
    date_series = pd.to_datetime(features["date"])
    date_range = (date_series.min().date(), date_series.max().date())

    model = build_model(model_type, params=params)
    x_values = features[feature_columns]
    y_values = features[target_columns]
    model.fit(x_values, y_values)

    metadata = ModelMetadata(
        feature_columns=feature_columns,
        target_columns=target_columns,
        params=params or {},
        h3_res=h3_res,
        date_range=date_range,
        model_type=model_type,
    )
    model_hash = _metadata_hash(metadata)
    payload = ModelPayload(model=model, metadata=metadata, model_hash=model_hash)
    persist_model(payload)
    return payload


def build_forecast_features(
    aggregated: pd.DataFrame,
    forecast_date: date,
    candidate_cells: list[str],
) -> pd.DataFrame:
    history = aggregated.copy()
    history["date"] = pd.to_datetime(history["date"])
    forecast_ts = pd.to_datetime(forecast_date)

    start_7 = forecast_ts - timedelta(days=7)
    start_28 = forecast_ts - timedelta(days=28)

    mask_7 = (history["date"] >= start_7) & (history["date"] < forecast_ts)
    mask_28 = (history["date"] >= start_28) & (history["date"] < forecast_ts)

    avg_7 = (
        history.loc[mask_7]
        .groupby(["cell_id", "hour"])["total_count"]
        .mean()
        .rename("avg_last_7_days")
    )
    avg_28 = (
        history.loc[mask_28]
        .groupby(["cell_id", "hour"])["total_count"]
        .mean()
        .rename("avg_last_28_days")
    )
    same_week = (
        history.loc[history["date"] == forecast_ts - timedelta(days=7)]
        .set_index(["cell_id", "hour"])["total_count"]
        .rename("same_dow_last_week")
    )

    base = pd.MultiIndex.from_product(
        [candidate_cells, range(24)], names=["cell_id", "hour"]
    ).to_frame(index=False)
    base = base.merge(avg_7.reset_index(), on=["cell_id", "hour"], how="left")
    base = base.merge(avg_28.reset_index(), on=["cell_id", "hour"], how="left")
    base = base.merge(same_week.reset_index(), on=["cell_id", "hour"], how="left")
    base["avg_last_7_days"] = base["avg_last_7_days"].fillna(0)
    base["avg_last_28_days"] = base["avg_last_28_days"].fillna(0)
    base["same_dow_last_week"] = base["same_dow_last_week"].fillna(0)
    base["day_of_week"] = forecast_ts.dayofweek
    base["date"] = forecast_date
    return base


__all__ = [
    "DEFAULT_FEATURE_COLUMNS",
    "DEFAULT_MODEL_TYPE",
    "ModelMetadata",
    "ModelPayload",
    "aggregate_incidents",
    "add_rolling_features",
    "build_feature_target_frames",
    "build_model",
    "load_model",
    "persist_model",
    "train_model",
    "build_forecast_features",
]
