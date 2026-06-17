"""Data loading, range-filtering and leakage-safe feature preparation.

Key responsibilities:
  * load a split's Phase-1 feature table and **filter it to its intended
    (year, months)** — this hardens the pipeline against the known Phase-1
    date-range spillover in the arrival / 2024 files;
  * compute **leakage-safe** high-cardinality encodings for ``station_name``
    (frequency + smoothed target encoding) **fitted on TRAIN only**;
  * assemble the model matrix ``X`` (``config.MODEL_FEATURES``), the target
    ``y``, and a ``meta`` frame (station, lat/lon, demand tier) used by the
    fairness and explainability stages.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config
from . import io


# --------------------------------------------------------------------------- #
# Loading + range filtering
# --------------------------------------------------------------------------- #
def _schema_guard(df: pd.DataFrame, stem: str) -> None:
    missing = [c for c in config.EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{stem}: missing expected columns {missing}; got {list(df.columns)}")


def filter_to_range(df: pd.DataFrame, spec: config.SplitSpec) -> pd.DataFrame:
    ts = pd.to_datetime(df[config.TIME_COL])
    mask = ts.dt.year == spec.year
    if spec.months is not None:
        mask &= ts.dt.month.isin(spec.months)
    return df.loc[mask].reset_index(drop=True)


def load_split(
    target: str,
    split: str,
    *,
    local_dir: str | None = None,
    sample_stations: int | None = None,
    sample_frac: float | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Load + range-filter one split. Optional subsampling for fast local dev."""
    spec = config.split_specs(target)[split]
    df = io.read_feature_table(spec.file_stem, local_dir=local_dir)
    _schema_guard(df, spec.file_stem)

    before = len(df)
    df = filter_to_range(df, spec)
    after = len(df)
    if after < before:
        warnings.warn(
            f"{spec.file_stem}: range-filtered {before:,} -> {after:,} rows "
            f"(year={spec.year}, months={spec.months}) — dropped out-of-range rows."
        )

    if sample_stations:
        stations = (
            df[config.STATION_COL].drop_duplicates()
            .sample(min(sample_stations, df[config.STATION_COL].nunique()),
                    random_state=random_state)
        )
        df = df[df[config.STATION_COL].isin(stations)].reset_index(drop=True)
    if sample_frac and sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=random_state).reset_index(drop=True)

    # Compact dtypes to keep memory in check on the full table.
    df[config.TARGET_COL] = df[config.TARGET_COL].astype("float32")
    return df


# --------------------------------------------------------------------------- #
# Leakage-safe station encoding (fit on TRAIN only)
# --------------------------------------------------------------------------- #
@dataclass
class StationEncoder:
    """Frequency + smoothed target (mean-demand) encoding for ``station_name``.

    Advanced encoding for the high-cardinality station id, fit strictly on the
    training split so no validation/test signal leaks in. Unseen stations fall
    back to global statistics.
    """

    smoothing: float = 20.0
    target_map: dict | None = None
    freq_map: dict | None = None
    global_target: float = 0.0
    global_freq: float = 0.0

    def fit(self, df: pd.DataFrame) -> "StationEncoder":
        n = len(df)
        grp = df.groupby(config.STATION_COL)[config.TARGET_COL].agg(["mean", "count"])
        self.global_target = float(df[config.TARGET_COL].mean())
        self.global_freq = 1.0 / max(grp.shape[0], 1)
        smoothed = (
            (grp["mean"] * grp["count"] + self.global_target * self.smoothing)
            / (grp["count"] + self.smoothing)
        )
        self.target_map = smoothed.to_dict()
        self.freq_map = (grp["count"] / n).to_dict()
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        st = out[config.STATION_COL]
        out["station_target_enc"] = st.map(self.target_map).fillna(self.global_target).astype("float32")
        out["station_freq"] = st.map(self.freq_map).fillna(0.0).astype("float32")
        return out


# --------------------------------------------------------------------------- #
# Demand tiers (for fairness slicing) — derived from TRAIN station means
# --------------------------------------------------------------------------- #
def fit_demand_tiers(train_df: pd.DataFrame) -> dict:
    station_mean = train_df.groupby(config.STATION_COL)[config.TARGET_COL].mean()
    q1, q2 = station_mean.quantile([1 / 3, 2 / 3]).tolist()
    return {"q1": float(q1), "q2": float(q2), "station_mean": station_mean.to_dict(),
            "global_mean": float(station_mean.mean())}


def assign_tier(df: pd.DataFrame, tiers: dict) -> pd.Series:
    sm = df[config.STATION_COL].map(tiers["station_mean"]).fillna(tiers["global_mean"])
    return pd.cut(sm, bins=[-np.inf, tiers["q1"], tiers["q2"], np.inf],
                 labels=["low", "medium", "high"])


# --------------------------------------------------------------------------- #
# Assemble model matrices
# --------------------------------------------------------------------------- #
def prepare_xy(df: pd.DataFrame, encoder: StationEncoder, tiers: dict | None = None):
    """Return (X, y, meta). ``encoder`` must already be fitted on train."""
    enc = encoder.transform(df)
    X = enc[config.MODEL_FEATURES].astype("float32")
    y = enc[config.TARGET_COL].astype("float32")
    meta = pd.DataFrame({
        config.STATION_COL: enc[config.STATION_COL].values,
        "latitude": enc["latitude"].values,
        "longitude": enc["longitude"].values,
        config.TARGET_COL: y.values,
    })
    if tiers is not None:
        meta["demand_tier"] = assign_tier(enc, tiers).values
    return X, y, meta
