"""BIXI demand MLOps modeling package (Phase 2).

Predictive modeling, AutoML/HPO, MLflow tracking, explainability, fairness and
drift for 15-minute BIXI station demand (departures and arrivals).

The pipeline is staged and resumable (see ``bixi.pipeline``) and runs identically
on a laptop subsample and on AWS Batch over the full dataset.
"""

__version__ = "0.2.0"
