"""Explainability artifacts: SHAP (global + local) and LIME (local).

Produces PNGs/HTML written to a local dir (then uploaded to S3 by the pipeline)
and surfaced on the Streamlit Explainability page (Phase 4).
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _save(fig_path: str) -> str:
    plt.tight_layout()
    plt.savefig(fig_path, dpi=120, bbox_inches="tight")
    plt.close()
    return fig_path


def shap_artifacts(model, X: pd.DataFrame, out_dir: str, *, max_rows: int = 2000,
                   n_local: int = 3) -> list[str]:
    import shap

    os.makedirs(out_dir, exist_ok=True)
    Xs = X.sample(min(max_rows, len(X)), random_state=42) if len(X) > max_rows else X
    paths: list[str] = []

    try:
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(Xs)
    except Exception:
        # Non-tree fallback: model-agnostic explainer is far slower, so cap rows.
        Xs = Xs.sample(min(500, len(Xs)), random_state=42)
        explainer = shap.Explainer(model.predict, Xs)
        sv = explainer(Xs).values

    # Global: beeswarm + bar
    shap.summary_plot(sv, Xs, show=False)
    paths.append(_save(os.path.join(out_dir, "shap_summary_beeswarm.png")))
    shap.summary_plot(sv, Xs, plot_type="bar", show=False)
    paths.append(_save(os.path.join(out_dir, "shap_importance_bar.png")))

    # Local waterfalls for a few rows
    try:
        base = explainer.expected_value
        base = float(np.ravel(base)[0])
        for i in range(min(n_local, len(Xs))):
            expl = shap.Explanation(values=np.asarray(sv)[i], base_values=base,
                                    data=Xs.iloc[i].values,
                                    feature_names=list(Xs.columns))
            shap.plots.waterfall(expl, show=False)
            paths.append(_save(os.path.join(out_dir, f"shap_waterfall_{i}.png")))
    except Exception:
        pass

    # Persist mean |SHAP| importances as JSON-friendly table
    imp = pd.DataFrame({
        "feature": list(Xs.columns),
        "mean_abs_shap": np.abs(np.asarray(sv)).mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False)
    imp.to_csv(os.path.join(out_dir, "shap_importance.csv"), index=False)
    paths.append(os.path.join(out_dir, "shap_importance.csv"))
    return paths


def lime_artifacts(model, X_train: pd.DataFrame, X_explain: pd.DataFrame,
                   out_dir: str, *, n_local: int = 3) -> list[str]:
    from lime.lime_tabular import LimeTabularExplainer

    os.makedirs(out_dir, exist_ok=True)
    explainer = LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=list(X_train.columns),
        mode="regression",
        discretize_continuous=True,
        random_state=42,
    )
    paths: list[str] = []
    for i in range(min(n_local, len(X_explain))):
        exp = explainer.explain_instance(
            X_explain.iloc[i].values, model.predict, num_features=10
        )
        html_path = os.path.join(out_dir, f"lime_instance_{i}.html")
        exp.save_to_file(html_path)
        paths.append(html_path)
    return paths
