from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable

import pandas as pd

from backend.models.registry import build_forecast_features, load_model

CACHE_DIR = Path(__file__).resolve().parents[1] / "cache" / "forecasts"
CANDIDATE_CACHE = Path(__file__).resolve().parents[1] / "cache" / "candidates_res8.parquet"


@dataclass(frozen=True)
class DailyForecastCube:
    forecast_date: date
    model_hash: str
    res: int
    path: Path

    @classmethod
    def _path_for(cls, forecast_date: date, model_hash: str, res: int) -> Path:
        date_dir = CACHE_DIR / forecast_date.isoformat()
        return date_dir / f"cube_{model_hash}_res{res}.parquet"

    @classmethod
    def load(cls, forecast_date: date, model_hash: str, res: int) -> pd.DataFrame:
        path = cls._path_for(forecast_date, model_hash, res)
        if not path.exists():
            raise FileNotFoundError(f"Forecast cube not found at {path}")
        return pd.read_parquet(path)

    @classmethod
    def compute(
        cls,
        forecast_date: date,
        model_hash: str,
        res: int,
        aggregated_history: pd.DataFrame,
        candidate_cells: Iterable[str] | None = None,
    ) -> pd.DataFrame:
        payload = load_model(model_hash)
        candidates = list(candidate_cells) if candidate_cells is not None else _load_candidates()

        feature_frame = build_forecast_features(
            aggregated_history,
            forecast_date=forecast_date,
            candidate_cells=candidates,
        )
        feature_columns = payload.metadata.feature_columns
        prediction_matrix = payload.model.predict(feature_frame[feature_columns])
        predictions = pd.DataFrame(
            prediction_matrix, columns=payload.metadata.target_columns
        )

        cube = pd.concat(
            [feature_frame[["cell_id", "hour", "date"]].reset_index(drop=True), predictions],
            axis=1,
        )
        cube["model_hash"] = model_hash
        cube["res"] = res

        path = cls._path_for(forecast_date, model_hash, res)
        path.parent.mkdir(parents=True, exist_ok=True)
        cube.to_parquet(path, index=False)
        return cube


def _load_candidates() -> list[str]:
    if not CANDIDATE_CACHE.exists():
        return []
    try:
        candidates = pd.read_parquet(CANDIDATE_CACHE)
    except Exception:
        return []
    if "cell_id" not in candidates.columns:
        return []
    return candidates["cell_id"].dropna().astype(str).tolist()


__all__ = ["DailyForecastCube"]
