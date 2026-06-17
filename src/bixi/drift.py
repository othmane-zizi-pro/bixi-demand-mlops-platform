"""Drift analysis with Evidently (4 types), scoped to the dataset's constraints.

We compute all four drift types the course asks for, comparing the 2024 training
reference against the 2025 May/Oct data:

  1. Feature drift     — input feature distribution shift (Evidently DataDrift).
  2. Target drift      — shift in the demand distribution.
  3. Prediction drift  — shift in the model's output distribution.
  4. Concept drift     — change in the X->y relationship, evidenced by a drop in
                         regression quality (R2/RMSE) on new labelled data.

IMPORTANT data caveat (agreed with the team): BIXI exposes trip history only up
to ~Apr-2026 and we use **2024 as the baseline for all years**, so the historical
baseline features for 2025/2026 are computed from matching 2024 periods. This
limits how much "real" drift can exist in those engineered features and is why we
do **not** run a live weekly cron — drift here is an analysis under known data
constraints, with human review, not a production monitor.

HTML reports (the visual deliverable) come from Evidently; the pass/fail flags
are computed directly (scipy KS) so they don't depend on Evidently internals.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from . import config


CAVEAT = (
    "2024 is the baseline for all years; 2025/2026 historical-baseline features "
    "are derived from matching 2024 periods, which limits engineered-feature drift. "
    "No live weekly cron — analysis under data constraints, with human review."
)


# --------------------------------------------------------------------------- #
# Evidently HTML reports (new 0.7 API)
# --------------------------------------------------------------------------- #
def _evidently_drift_html(ref: pd.DataFrame, cur: pd.DataFrame,
                          num_cols: list[str], out_html: str) -> str | None:
    try:
        from evidently import Dataset, DataDefinition, Report
        from evidently.presets import DataDriftPreset

        defn = DataDefinition(numerical_columns=num_cols)
        ref_ds = Dataset.from_pandas(ref[num_cols], data_definition=defn)
        cur_ds = Dataset.from_pandas(cur[num_cols], data_definition=defn)
        snap = Report([DataDriftPreset()]).run(current_data=cur_ds, reference_data=ref_ds)
        os.makedirs(os.path.dirname(out_html) or ".", exist_ok=True)
        snap.save_html(out_html)
        return out_html
    except Exception as e:  # never let report rendering break the pipeline
        return f"ERROR: {type(e).__name__}: {e}"


def _evidently_regression_html(cur: pd.DataFrame, out_html: str) -> str | None:
    try:
        from evidently import DataDefinition, Dataset, Regression, Report
        from evidently.presets import RegressionPreset

        defn = DataDefinition(
            numerical_columns=["target", "prediction"],
            regression=[Regression(target="target", prediction="prediction")],
        )
        ds = Dataset.from_pandas(cur[["target", "prediction"]], data_definition=defn)
        snap = Report([RegressionPreset()]).run(current_data=ds)
        os.makedirs(os.path.dirname(out_html) or ".", exist_ok=True)
        snap.save_html(out_html)
        return out_html
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


# --------------------------------------------------------------------------- #
# Robust drift flags (scipy KS)
# --------------------------------------------------------------------------- #
def _ks(ref: np.ndarray, cur: np.ndarray, sample: int = 50000) -> dict:
    rng = np.random.default_rng(42)
    a = ref if len(ref) <= sample else rng.choice(ref, sample, replace=False)
    b = cur if len(cur) <= sample else rng.choice(cur, sample, replace=False)
    s = ks_2samp(a, b)
    return {"ks_stat": float(s.statistic), "p_value": float(s.pvalue),
            "drift": bool(s.pvalue < 0.05)}


# --------------------------------------------------------------------------- #
# Orchestrated drift analysis for one comparison period
# --------------------------------------------------------------------------- #
def analyze_period(ref_df: pd.DataFrame, cur_df: pd.DataFrame,
                   ref_pred: np.ndarray, cur_pred: np.ndarray,
                   period: str, out_dir: str, *, r2_alert: float = 0.55,
                   ref_r2: float | None = None) -> dict:
    """Return a summary dict for a single reference->current comparison and write
    the Evidently HTML reports into ``out_dir``."""
    from .models import metrics

    os.makedirs(out_dir, exist_ok=True)
    feats = config.RAW_FEATURE_COLS

    # Evidently renders on a bounded sample (drift on a 50k sample is robust and
    # keeps HTML rendering fast even over tens of millions of rows).
    cap = 50000
    ref_s = ref_df.sample(min(cap, len(ref_df)), random_state=1)
    cur_s = cur_df.sample(min(cap, len(cur_df)), random_state=1)

    summary: dict = {"period": period, "caveat": CAVEAT, "reports": {}}

    # 1. Feature drift
    summary["reports"]["feature_drift_html"] = _evidently_drift_html(
        ref_s, cur_s, feats, os.path.join(out_dir, f"feature_drift_{period}.html"))
    drifted = {c: _ks(ref_df[c].to_numpy(), cur_df[c].to_numpy()) for c in feats}
    summary["feature_drift"] = {
        "n_features": len(feats),
        "n_drifted": int(sum(v["drift"] for v in drifted.values())),
        "share_drifted": float(np.mean([v["drift"] for v in drifted.values()])),
        "per_feature": drifted,
    }

    # 2. Target drift
    summary["target_drift"] = _ks(ref_df[config.TARGET_COL].to_numpy(),
                                  cur_df[config.TARGET_COL].to_numpy())
    summary["reports"]["target_drift_html"] = _evidently_drift_html(
        ref_s, cur_s, [config.TARGET_COL],
        os.path.join(out_dir, f"target_drift_{period}.html"))

    # 3. Prediction drift
    summary["prediction_drift"] = _ks(np.asarray(ref_pred), np.asarray(cur_pred))
    rng = np.random.default_rng(1)
    rp = np.asarray(ref_pred); cp = np.asarray(cur_pred)
    pdf_ref = pd.DataFrame({"prediction": rp if len(rp) <= cap else rng.choice(rp, cap, replace=False)})
    pdf_cur = pd.DataFrame({"prediction": cp if len(cp) <= cap else rng.choice(cp, cap, replace=False)})
    summary["reports"]["prediction_drift_html"] = _evidently_drift_html(
        pdf_ref, pdf_cur, ["prediction"],
        os.path.join(out_dir, f"prediction_drift_{period}.html"))

    # 4. Concept drift — performance drop on new labelled data
    cur_perf = metrics(cur_df[config.TARGET_COL].to_numpy(), cur_pred)
    concept = {"current_r2": cur_perf["r2"], "current_rmse": cur_perf["rmse"],
               "r2_alert_threshold": r2_alert,
               "concept_drift_alert": bool(cur_perf["r2"] < r2_alert)}
    if ref_r2 is not None:
        concept["reference_r2"] = float(ref_r2)
        concept["r2_drop"] = float(ref_r2 - cur_perf["r2"])
    summary["concept_drift"] = concept
    reg_df = pd.DataFrame({"target": cur_df[config.TARGET_COL].to_numpy(),
                           "prediction": np.asarray(cur_pred)})
    reg_s = reg_df.sample(min(cap, len(reg_df)), random_state=1)
    summary["reports"]["concept_regression_html"] = _evidently_regression_html(
        reg_s, os.path.join(out_dir, f"concept_regression_{period}.html"))

    return summary
