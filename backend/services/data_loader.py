from __future__ import annotations

from pathlib import Path

import h3
import pandas as pd


NORMALIZED_COLUMNS = {
    "lat": "lat",
    "latitude": "lat",
    "lng": "lng",
    "lon": "lng",
    "longitude": "lng",
    "title": "title",
    "timestamp": "timeStamp",
    "timeStamp": "timeStamp",
}


def load_911_csv(path: str | Path, **read_csv_kwargs: object) -> pd.DataFrame:
    """Load the 911 CSV dataset from disk."""
    return pd.read_csv(path, **read_csv_kwargs)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names for downstream processing."""
    rename_map: dict[str, str] = {}
    for column in df.columns:
        normalized = NORMALIZED_COLUMNS.get(column.lower())
        if normalized and column != normalized:
            rename_map[column] = normalized
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def add_datetime_features(
    df: pd.DataFrame, timestamp_column: str = "timeStamp"
) -> pd.DataFrame:
    """Parse timestamp column and add date-related features."""
    if timestamp_column not in df.columns:
        return df

    timestamps = pd.to_datetime(df[timestamp_column], errors="coerce")
    df = df.copy()
    df[timestamp_column] = timestamps
    df["date"] = timestamps.dt.date
    df["hour"] = timestamps.dt.hour
    df["day_of_week"] = timestamps.dt.dayofweek
    df["month"] = timestamps.dt.month
    df["is_weekend"] = timestamps.dt.dayofweek >= 5
    return df


def add_incident_type(df: pd.DataFrame, title_column: str = "title") -> pd.DataFrame:
    """Map incident type based on the title prefix."""
    if title_column not in df.columns:
        return df

    title_prefix = (
        df[title_column]
        .astype("string")
        .str.split(":", n=1)
        .str[0]
        .str.strip()
        .str.upper()
    )
    mapping = {"EMS": "EMS", "FIRE": "Fire", "TRAFFIC": "Traffic"}
    df = df.copy()
    df["incident_type"] = title_prefix.map(mapping)
    return df


def add_cell_id(
    df: pd.DataFrame, lat_column: str = "lat", lng_column: str = "lng", res: int = 8
) -> pd.DataFrame:
    """Add H3 cell IDs based on latitude/longitude."""
    if lat_column not in df.columns or lng_column not in df.columns:
        return df

    df = df.copy()
    df["cell_id"] = pd.NA
    valid_mask = df[lat_column].notna() & df[lng_column].notna()
    if valid_mask.any():
        df.loc[valid_mask, "cell_id"] = df.loc[valid_mask].apply(
            lambda row: h3.latlng_to_cell(row[lat_column], row[lng_column], res=res),
            axis=1,
        )
    return df


def prepare_911_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize, enrich, and add derived columns for 911 dataset."""
    df = normalize_columns(df)
    df = add_datetime_features(df)
    df = add_incident_type(df)
    df = add_cell_id(df)
    return df


def load_and_prepare_911_csv(
    path: str | Path, **read_csv_kwargs: object
) -> pd.DataFrame:
    """Load and prepare the 911 CSV dataset in one step."""
    df = load_911_csv(path, **read_csv_kwargs)
    return prepare_911_dataframe(df)
