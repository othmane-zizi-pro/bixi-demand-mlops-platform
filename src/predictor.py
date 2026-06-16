"""Shared prediction logic for Streamlit and FastAPI."""

from __future__ import annotations

import datetime as dt
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.s3_io import read_bytes_from_s3, read_pickle_from_s3


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class PredictionInput:
    station: str
    date: dt.date
    hour: int
    is_holiday: int
    temperature: float
    feels_like: float
    wind_speed: float
    bad_weather: int


class BixiDemandPredictor:
    def __init__(self, model: lgb.Booster, meta: dict[str, Any], source: str):
        self.model = model
        self.meta = meta
        self.source = source

    @classmethod
    def load(cls) -> "BixiDemandPredictor":
        source = os.getenv("MODEL_SOURCE", "local").lower()
        if source == "s3":
            return cls._load_from_s3()
        return cls._load_from_local()

    @classmethod
    def _load_from_local(cls) -> "BixiDemandPredictor":
        model_path = Path(os.getenv("LOCAL_MODEL_PATH", PROJECT_ROOT / "model_lightgbm.txt"))
        meta_path = Path(os.getenv("LOCAL_META_PATH", PROJECT_ROOT / "meta_lightgbm.pkl"))

        model = lgb.Booster(model_file=str(model_path))
        with meta_path.open("rb") as f:
            meta = pickle.load(f)
        return cls(model=model, meta=meta, source="local")

    @classmethod
    def _load_from_s3(cls) -> "BixiDemandPredictor":
        model_key = os.getenv("MODEL_KEY")
        meta_key = os.getenv("META_KEY")
        if not model_key or not meta_key:
            raise RuntimeError("MODEL_KEY and META_KEY are required when MODEL_SOURCE=s3.")

        model_text = read_bytes_from_s3(model_key).decode("utf-8")
        model = lgb.Booster(model_str=model_text)
        meta = read_pickle_from_s3(meta_key)
        return cls(model=model, meta=meta, source="s3")

    @property
    def stations(self) -> list[str]:
        return list(self.meta["station"])

    def build_features(self, req: PredictionInput) -> dict[str, Any]:
        dow = req.date.weekday() + 1
        month = req.date.month

        key_hour = (req.station, req.hour)
        key_dow = (req.station, dow)
        key_month = (req.station, month)

        return {
            "station": req.station,
            "hour": req.hour,
            "dow": dow,
            "month": month,
            "is_holiday": req.is_holiday,
            "bad_weather": req.bad_weather,
            "station_hour_demand_24": self.meta["station_hour_demand_24"].get(
                key_hour, self.meta["global_hour_demand_24"]
            ),
            "station_dow_demand_24": self.meta["station_dow_demand_24"].get(
                key_dow, self.meta["global_dow_demand_24"]
            ),
            "station_month_demand_24": self.meta["station_month_demand_24"].get(
                key_month, self.meta["global_month_demand_24"]
            ),
            "temperature": req.temperature,
            "feels_like": req.feels_like,
            "wind_speed": req.wind_speed,
        }

    def predict(self, req: PredictionInput) -> tuple[float, dict[str, Any]]:
        feature_values = self.build_features(req)
        all_features = self.meta["all_features"]
        categorical_features = self.meta["categorical_features"]

        row = {col: feature_values.get(col, np.nan) for col in all_features}
        x_input = pd.DataFrame([row], columns=all_features)

        for col in categorical_features:
            x_input[col] = x_input[col].astype("string").astype("category")

        numeric_cols = [col for col in all_features if col not in categorical_features]
        x_input[numeric_cols] = x_input[numeric_cols].apply(pd.to_numeric, errors="coerce")

        prediction = float(self.model.predict(x_input)[0])
        return max(prediction, 0.0), feature_values
