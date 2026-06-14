"""FastAPI backend for BIXI demand prediction."""

from __future__ import annotations

import datetime as dt
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.predictor import BixiDemandPredictor, PredictionInput


app = FastAPI(title="BIXI Demand Forecast API", version="0.1.0")


class ForecastRequest(BaseModel):
    station: str
    date: dt.date
    hour: int = Field(ge=0, le=23)
    is_holiday: int = Field(default=0, ge=0, le=1)
    temperature: float
    feels_like: float
    wind_speed: float = Field(ge=0)
    bad_weather: int = Field(default=0, ge=0, le=1)


@lru_cache(maxsize=1)
def get_predictor() -> BixiDemandPredictor:
    return BixiDemandPredictor.load()


@app.get("/health")
def health():
    predictor = get_predictor()
    return {
        "status": "ok",
        "model_source": predictor.source,
        "station_count": len(predictor.stations),
    }


@app.get("/stations")
def stations():
    return {"stations": get_predictor().stations}


@app.post("/predict")
def predict(request: ForecastRequest):
    predictor = get_predictor()
    if request.station not in predictor.stations:
        raise HTTPException(status_code=400, detail="Unknown station.")

    prediction, features = predictor.predict(
        PredictionInput(
            station=request.station,
            date=request.date,
            hour=request.hour,
            is_holiday=request.is_holiday,
            temperature=request.temperature,
            feels_like=request.feels_like,
            wind_speed=request.wind_speed,
            bad_weather=request.bad_weather,
        )
    )

    return {
        "station": request.station,
        "date": request.date.isoformat(),
        "hour": request.hour,
        "predicted_total_demand": round(prediction, 2),
        "model_source": predictor.source,
        "features": features,
    }
