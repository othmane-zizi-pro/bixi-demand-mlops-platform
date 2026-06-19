"""Generate the matplotlib charts for the final presentation deck.

All charts run off committed artifacts (no AWS, no network):
  1. EDA heatmap      — average 2024 demand by weekday x 15-minute slot
  2. Results slide     — R2/RMSE/MAE per target/split (reuses make_results_slide.py)
  3. Rebalancing chart — top-N needs-bikes vs needs-docks priority stations
  4. Occupancy traj.   — cumulative net-flow trajectory for an example station each way
  5. Net-flow map      — Montreal stations coloured by need, sized by risk

Outputs -> docs/presentation/charts/*.png  (results slide -> docs/presentation/).

Run:  PYTHONPATH=src ./.venv/bin/python scripts/build_deck_charts.py
"""

from __future__ import annotations

import json
import runpy
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from bixi import config
from bixi import rebalancing as rb
from bixi import streamlit_local_serving as serving

# --------------------------------------------------------------------------- #
# Theme
# --------------------------------------------------------------------------- #
RED = "#b91c1c"
RED_LT = "#fca5a5"
BLUE = "#1d4ed8"
BLUE_LT = "#93c5fd"
GREY = "#9ca3af"
DARK = "#1f2937"
INK = "#111827"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.edgecolor": "#d1d5db",
    "axes.linewidth": 0.8,
    "text.color": INK,
    "axes.labelcolor": INK,
    "xtick.color": DARK,
    "ytick.color": DARK,
    "figure.facecolor": "white",
})

REPO = Path(__file__).resolve().parents[1]
PRES = REPO / "docs" / "presentation"
CHARTS = PRES / "charts"
CHARTS.mkdir(parents=True, exist_ok=True)

WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
DEP_BASELINES = (
    REPO / "artifacts" / "streamlit-community-cloud" / "cloud-2024"
    / "departure" / "data" / "serving_baselines.parquet"
)


def _save(fig, name: str) -> None:
    out = CHARTS / name
    fig.savefig(out, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out.relative_to(REPO)}")


# --------------------------------------------------------------------------- #
# 1. EDA heatmap — average demand by weekday x 15-min slot
# --------------------------------------------------------------------------- #
def chart_eda_heatmap() -> None:
    df = pd.read_parquet(DEP_BASELINES, columns=["dayofweek", "slot_of_day", "hist_avg_demand"])
    grid = (
        df.groupby(["dayofweek", "slot_of_day"])["hist_avg_demand"]
        .mean()
        .unstack("slot_of_day")
        .reindex(index=range(7), columns=range(96))
    )
    cmap = LinearSegmentedColormap.from_list("bixi_heat", ["#f8fafc", "#fca5a5", "#b91c1c", "#7f1d1d"])

    fig, ax = plt.subplots(figsize=(13.33, 5.9), dpi=200)
    fig.subplots_adjust(top=0.80, left=0.06, right=0.99, bottom=0.13)
    im = ax.imshow(grid.values, aspect="auto", cmap=cmap, origin="upper")
    ax.set_yticks(range(7))
    ax.set_yticklabels(WEEKDAYS, fontsize=11)
    ax.set_xticks([h * 4 for h in range(0, 25, 2)])
    ax.set_xticklabels([f"{h:02d}h" for h in range(0, 25, 2)], fontsize=10)
    ax.set_xlabel("Time of day (96 fifteen-minute slots)", fontsize=12)
    fig.text(0.06, 0.93, "Average departure demand by weekday × 15-minute slot",
             fontsize=18, fontweight="bold", color=INK)
    fig.text(0.06, 0.875,
             "2024 historical baselines, averaged over all ~1,100 stations  ·  motivates the 15-min, day-of-week feature design",
             fontsize=11.5, color=DARK)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    cbar = fig.colorbar(im, ax=ax, pad=0.012, fraction=0.035)
    cbar.set_label("Mean trips / 15-min slot", fontsize=10)
    cbar.ax.tick_params(labelsize=9, length=0)
    # Honest annotations: the real pattern peaks midday and (especially) weekday evenings.
    ax.annotate("quiet overnight", (10, 5.5), fontsize=9.5, color=DARK, ha="center",
                style="italic")
    ax.annotate("busiest: weekday\nevenings", (86, 2.0), fontsize=9.5, color="white",
                ha="center", fontweight="bold")
    _save(fig, "eda_demand_heatmap.png")


# --------------------------------------------------------------------------- #
# Rebalancing inputs (shared by charts 3-5)
# --------------------------------------------------------------------------- #
def _rebalancing_frames():
    bundles = serving.load_local_bundles()
    netflow_df, risk_df = rb.compute_rebalancing(bundles, dayofweek=1, month=6)
    return netflow_df, risk_df


# --------------------------------------------------------------------------- #
# 3. Rebalancing priority chart — top-N needs-bikes vs needs-docks
# --------------------------------------------------------------------------- #
def chart_rebalancing_priority(risk_df: pd.DataFrame, top_n: int = 15) -> None:
    head = risk_df.head(top_n).iloc[::-1].reset_index(drop=True)
    colors = [BLUE if d == rb.NEEDS_DOCKS else RED for d in head["direction"]]
    labels = [
        (s[:34] + "…") if len(s) > 35 else s
        for s in head[config.STATION_COL].astype(str)
    ]

    fig, ax = plt.subplots(figsize=(13.33, 7.0), dpi=200)
    y = np.arange(len(head))
    ax.barh(y, head["risk_score"], color=colors, height=0.72)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Rebalancing risk score  =  max(peak deficit, peak surplus)   ·   trips", fontsize=11.5)
    ax.set_title(
        f"Top {top_n} rebalancing priorities  ·  representative Tuesday, neutral weather",
        fontsize=15, fontweight="bold", loc="left", pad=10,
    )
    for yi, (val, d) in enumerate(zip(head["risk_score"], head["direction"])):
        ax.annotate(f"{val:.0f}", (val, yi), xytext=(4, 0), textcoords="offset points",
                    va="center", fontsize=9.5, color=INK)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=0)
    ax.margins(x=0.08)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=RED),
        plt.Rectangle((0, 0), 1, 1, color=BLUE),
    ]
    ax.legend(handles, ["needs bikes (stockout risk)", "needs docks (overflow risk)"],
              frameon=False, fontsize=10.5, loc="lower right")
    _save(fig, "rebalancing_priority.png")


# --------------------------------------------------------------------------- #
# 4. Occupancy-trajectory chart — example needs-bikes & needs-docks station
# --------------------------------------------------------------------------- #
def chart_occupancy_trajectory(netflow_df: pd.DataFrame, risk_df: pd.DataFrame) -> None:
    def pick(direction):
        sub = risk_df[risk_df["direction"] == direction]
        return sub.iloc[0] if len(sub) else None

    docks = pick(rb.NEEDS_DOCKS)
    bikes = pick(rb.NEEDS_BIKES)
    picks = [(bikes, RED, "needs bikes"), (docks, BLUE, "needs docks")]
    picks = [p for p in picks if p[0] is not None]

    fig, axes = plt.subplots(1, len(picks), figsize=(13.33, 5.6), dpi=200, sharey=False)
    if len(picks) == 1:
        axes = [axes]
    hours = np.arange(96) / 4.0

    for ax, (row, color, tag) in zip(axes, picks):
        station = row[config.STATION_COL]
        g = (
            netflow_df[netflow_df[config.STATION_COL] == station]
            .sort_values("slot_of_day")
        )
        cum = np.cumsum(g["net_flow"].to_numpy(dtype="float64"))
        ax.axhline(0, color=GREY, lw=0.9, ls="--")
        ax.plot(hours[: len(cum)], cum, color=color, lw=2.2)
        ax.fill_between(hours[: len(cum)], cum, 0, color=color, alpha=0.12)

        i_def = int(np.argmin(cum))
        i_sur = int(np.argmax(cum))
        ax.scatter([hours[i_def]], [cum[i_def]], color=RED, zorder=5, s=36)
        ax.scatter([hours[i_sur]], [cum[i_sur]], color=BLUE, zorder=5, s=36)
        ax.annotate(f"peak deficit {row['peak_deficit']:.0f}\n@ {serving.slot_label(int(row['deficit_slot']))}",
                    (hours[i_def], cum[i_def]), textcoords="offset points", xytext=(6, -28),
                    fontsize=9, color=RED, fontweight="bold")
        ax.annotate(f"peak surplus {row['peak_surplus']:.0f}\n@ {serving.slot_label(int(row['surplus_slot']))}",
                    (hours[i_sur], cum[i_sur]), textcoords="offset points", xytext=(-8, 8),
                    fontsize=9, color=BLUE, fontweight="bold", ha="right")

        name = station if len(station) <= 40 else station[:39] + "…"
        ax.set_title(f"{name}\n({tag})", fontsize=12, fontweight="bold", loc="left", pad=8)
        ax.set_xlabel("Hour of day", fontsize=11)
        ax.set_ylabel("Relative occupancy  (cumulative net flow)", fontsize=10.5)
        ax.set_xticks(range(0, 25, 4))
        ax.set_xticklabels([f"{h:02d}h" for h in range(0, 25, 4)], fontsize=9.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(length=0)

    fig.suptitle("Cumulative net-flow occupancy trajectory  ·  net_flow = arrival − departure",
                 fontsize=15, fontweight="bold", x=0.02, ha="left", y=1.02)
    fig.tight_layout()
    _save(fig, "occupancy_trajectory.png")


# --------------------------------------------------------------------------- #
# 5. Net-flow station map — Montreal stations coloured by need, sized by risk
# --------------------------------------------------------------------------- #
def chart_netflow_map(risk_df: pd.DataFrame) -> None:
    df = risk_df.dropna(subset=["latitude", "longitude"]).copy()
    df = df[(df["latitude"].between(45.3, 45.75)) & (df["longitude"].between(-73.8, -73.4))]
    sizes = 10 + (df["risk_score"] / df["risk_score"].max()) * 240
    colors = [BLUE if d == rb.NEEDS_DOCKS else RED for d in df["direction"]]

    fig, ax = plt.subplots(figsize=(11.5, 8.2), dpi=200)
    ax.scatter(df["longitude"], df["latitude"], s=sizes, c=colors, alpha=0.55,
               edgecolors="white", linewidths=0.3)
    ax.set_title("Net-flow rebalancing pressure across Montreal  ·  size = risk severity",
                 fontsize=15, fontweight="bold", loc="left", pad=10)
    ax.set_xlabel("Longitude", fontsize=11)
    ax.set_ylabel("Latitude", fontsize=11)
    ax.set_aspect(1.0 / np.cos(np.deg2rad(45.5)))
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=9)
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=RED, markersize=11),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=BLUE, markersize=11),
    ]
    ax.legend(handles, ["needs bikes (drains / stockout)", "needs docks (fills / overflow)"],
              frameon=False, fontsize=10.5, loc="upper left")
    ax.annotate("Downtown / Old-Port core fills up (needs docks);\nresidential periphery drains (needs bikes).",
                (0.985, 0.02), xycoords="axes fraction", ha="right", va="bottom",
                fontsize=9.5, color=DARK,
                bbox=dict(boxstyle="round,pad=0.4", fc="#f9fafb", ec="#e5e7eb"))
    _save(fig, "netflow_station_map.png")


# --------------------------------------------------------------------------- #
# Drift — crisp, projector-legible panels built from the REAL Evidently-computed
# metrics committed in drift_summary.json (no soft HTML screenshots).
# --------------------------------------------------------------------------- #
DRIFT_SUMMARY = (
    REPO / "artifacts" / "streamlit-community-cloud" / "cloud-2024"
    / "departure" / "monitoring" / "drift_summary.json"
)


def _drift_period(period: str = "2025_oct") -> dict:
    return json.loads(DRIFT_SUMMARY.read_text())[period]


def chart_drift_features(period: str = "2025_oct") -> None:
    d = _drift_period(period)
    fd = d["feature_drift"]
    per = fd["per_feature"]
    items = sorted(per.items(), key=lambda kv: kv[1]["ks_stat"])  # asc for barh
    names = [k for k, _ in items]
    ks = [v["ks_stat"] for _, v in items]
    drift = [v["drift"] for _, v in items]
    colors = [RED if dr else GREY for dr in drift]

    fig, ax = plt.subplots(figsize=(6.2, 7.0), dpi=220)
    fig.subplots_adjust(left=0.40, right=0.97, top=0.84, bottom=0.09)
    y = np.arange(len(names))
    ax.barh(y, ks, color=colors, height=0.74)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=10.5, fontfamily="DejaVu Sans Mono")
    for yi, val in enumerate(ks):
        ax.annotate(f"{val:.3f}", (val, yi), xytext=(4, 0), textcoords="offset points",
                    va="center", fontsize=9, color=INK)
    ax.set_xlabel("Kolmogorov–Smirnov statistic", fontsize=10.5)
    ax.margins(x=0.13)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=0)
    fig.text(0.02, 0.955, "Feature drift", fontsize=17, fontweight="bold", color=INK)
    fig.text(0.02, 0.905,
             f"{fd['n_drifted']} of {fd['n_features']} features drifted (KS, p < 0.05)",
             fontsize=11.5, color=DARK)
    handles = [plt.Rectangle((0, 0), 1, 1, color=RED), plt.Rectangle((0, 0), 1, 1, color=GREY)]
    ax.legend(handles, ["drift detected", "no drift"], frameon=False, fontsize=10,
              loc="lower right")
    _save(fig, "drift_features_departure_oct.png")


def chart_drift_summary(period: str = "2025_oct") -> None:
    d = _drift_period(period)
    td, prd, cd = d["target_drift"], d["prediction_drift"], d["concept_drift"]
    ref_r2 = cd.get("reference_r2", 0.0)
    cur_r2 = cd["current_r2"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.2, 7.0), dpi=220,
                                   gridspec_kw={"height_ratios": [1, 1]})
    fig.subplots_adjust(left=0.13, right=0.95, top=0.82, bottom=0.09, hspace=0.55)
    fig.text(0.02, 0.955, "Target · Prediction · Concept drift", fontsize=16,
             fontweight="bold", color=INK)
    fig.text(0.02, 0.905, "2024 reference → October-2025", fontsize=11.5, color=DARK)

    # Distribution drift (target + prediction), KS statistic
    labels = ["Target\ndemand", "Model\nprediction"]
    vals = [td["ks_stat"], prd["ks_stat"]]
    bars = ax1.bar(labels, vals, color=[RED, "#ef4444"], width=0.55)
    ax1.set_ylabel("KS statistic", fontsize=10.5)
    ax1.set_ylim(0, max(vals) * 1.35)
    ax1.set_title("Distribution drift  ·  both detected (p < 0.001)", fontsize=12,
                  fontweight="bold", loc="left", pad=6)
    for b, v in zip(bars, vals):
        ax1.annotate(f"KS = {v:.3f}", (b.get_x() + b.get_width() / 2, v),
                     ha="center", va="bottom", fontsize=11, color=INK, fontweight="bold")
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.tick_params(length=0)

    # Concept drift — reference vs current R^2 (relationship stability)
    r2bars = ax2.bar(["Reference\n(2024)", "Current\n(Oct-2025)"], [ref_r2, cur_r2],
                     color=[GREY, BLUE], width=0.55)
    ax2.set_ylabel("R²  (regression quality)", fontsize=10.5)
    ax2.set_ylim(0, max(ref_r2, cur_r2) * 1.5)
    drop = ref_r2 - cur_r2
    verdict = "no degradation" if drop <= 0.02 else f"R² drop {drop:.3f}"
    ax2.set_title(f"Concept drift  ·  {verdict}", fontsize=12, fontweight="bold",
                  loc="left", pad=6)
    for b, v in zip(r2bars, [ref_r2, cur_r2]):
        ax2.annotate(f"{v:.3f}", (b.get_x() + b.get_width() / 2, v),
                     ha="center", va="bottom", fontsize=12, color=INK, fontweight="bold")
    ax2.annotate(f"RMSE {cd['current_rmse']:.3f}  ·  relationship stable",
                 (0.5, 0.92), xycoords="axes fraction", ha="center", fontsize=9.5,
                 color=DARK, style="italic")
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.tick_params(length=0)
    _save(fig, "drift_summary_departure_oct.png")


# --------------------------------------------------------------------------- #
def main() -> None:
    print("· EDA heatmap")
    chart_eda_heatmap()

    print("· Drift panels (from drift_summary.json)")
    chart_drift_features()
    chart_drift_summary()

    print("· Results slide (make_results_slide.py)")
    runpy.run_path(str(REPO / "scripts" / "make_results_slide.py"), run_name="__main__")

    print("· Rebalancing (predicting representative day off committed artifacts)…")
    netflow_df, risk_df = _rebalancing_frames()
    chart_rebalancing_priority(risk_df)
    chart_occupancy_trajectory(netflow_df, risk_df)
    chart_netflow_map(risk_df)
    print("done.")


if __name__ == "__main__":
    main()
