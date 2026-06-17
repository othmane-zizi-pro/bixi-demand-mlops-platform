"""Fairness / responsible-AI: prediction-error parity across station groups.

We check whether the model's error (RMSE / MAE / R2 / bias) is systematically
worse for some groups than others:
  * **demand tiers** (low / medium / high) — does the model under-serve
    quiet stations?
  * **geographic zones** (lat/lon grid) — are some areas of the city served
    less accurately?

A large disparity ratio (worst / best group RMSE) flags an equity concern and is
reported with mitigation notes for the write-up / Streamlit Monitoring page.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .models import metrics


def _group_metrics(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows = []
    for g, sub in df.groupby(group_col, observed=True):
        if len(sub) == 0:
            continue
        m = metrics(sub["y_true"], sub["y_pred"])
        m["bias"] = float((sub["y_pred"].clip(lower=0) - sub["y_true"]).mean())
        m["count"] = int(len(sub))
        m[group_col] = str(g)
        rows.append(m)
    cols = [group_col, "count", "rmse", "mae", "r2", "bias", "poisson_deviance"]
    return pd.DataFrame(rows)[cols].sort_values("rmse", ascending=False)


def add_geo_zone(meta: pd.DataFrame, round_deg: float = 0.02) -> pd.Series:
    lat = (meta["latitude"] / round_deg).round() * round_deg
    lon = (meta["longitude"] / round_deg).round() * round_deg
    return (lat.round(3).astype(str) + "," + lon.round(3).astype(str))


def fairness_report(meta: pd.DataFrame, y_true, y_pred, *, min_zone: int = 5000) -> dict:
    df = meta.copy()
    df["y_true"] = np.asarray(y_true, dtype="float64")
    df["y_pred"] = np.asarray(y_pred, dtype="float64")

    report: dict = {"overall": metrics(df["y_true"], df["y_pred"])}

    if "demand_tier" in df.columns:
        tier = _group_metrics(df, "demand_tier")
        report["by_demand_tier"] = tier.to_dict(orient="records")
        rmse = tier["rmse"].replace(0, np.nan)
        report["tier_rmse_disparity_ratio"] = float(rmse.max() / rmse.min())

    df["geo_zone"] = add_geo_zone(df)
    counts = df["geo_zone"].value_counts()
    keep = counts[counts >= min_zone].index
    zdf = df[df["geo_zone"].isin(keep)]
    if len(zdf):
        zone = _group_metrics(zdf, "geo_zone")
        report["worst_zones"] = zone.head(8).to_dict(orient="records")
        report["best_zones"] = zone.tail(8).to_dict(orient="records")
        rmse = zone["rmse"].replace(0, np.nan)
        report["zone_rmse_disparity_ratio"] = float(rmse.max() / rmse.min())
        report["n_zones_evaluated"] = int(len(zone))

    # Plain-language flags + mitigation notes
    flags = []
    if report.get("tier_rmse_disparity_ratio", 1) > 1.5:
        flags.append("RMSE differs >1.5x across demand tiers — high-demand stations "
                     "carry most of the absolute error; consider tier-weighted loss "
                     "or per-tier evaluation thresholds.")
    if report.get("zone_rmse_disparity_ratio", 1) > 2.0:
        flags.append("RMSE differs >2x across geographic zones — some areas are served "
                     "less accurately; consider zone-level features or monitoring.")
    report["flags"] = flags or ["No major error-parity disparities detected."]
    return report
