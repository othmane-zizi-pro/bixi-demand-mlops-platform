# Phase 2 — Predictive Modeling, MLflow, Explainability, Fairness & Drift

Owner: **Othmane Zizi** (`othmane-zizi-pro`). Branch: `phase-2-modeling-drift`.

This phase turns the Phase-1 15-minute feature tables into a production-grade,
**fully cloud-deployable and resumable** modeling pipeline: multi-model + AutoML +
Bayesian HPO, MLflow tracking & registry, SHAP/LIME explainability, fairness
analysis, and 4-type drift — trained in the cloud on AWS Batch, all provisioned
with AWS CDK.

## 1. Pipeline (`src/bixi/`, `python -m bixi.pipeline`)

Stages, each checkpointed to S3 with a `_SUCCESS` marker so any run resumes from
any step:

```
ingest -> data -> train -> explain -> fairness -> drift -> register
```

| Stage | What it does |
|---|---|
| `ingest` | (reproducibility) ensure raw BIXI trips + Open-Meteo weather are in S3; idempotent. Excluded from the default run since the good raw files already exist. |
| `data` | load each split, **filter to its intended (year, month)** (hardens against the Phase-1 date spillover), fit **leakage-safe** station encodings on TRAIN only, persist encoder + tiers. |
| `train` | naive baseline → LightGBM (L2/Poisson/Tweedie) + XGBoost + HistGB candidates → **FLAML AutoML** → **Optuna** HPO; select best by validation RMSE; evaluate on test; log everything to MLflow. |
| `explain` | SHAP global (beeswarm/bar) + local (waterfall) + LIME; artifacts → S3. |
| `fairness` | prediction-error parity across demand tiers and geographic zones; disparity flags + mitigation notes. |
| `drift` | Evidently feature/target/prediction/concept drift, 2024 ref vs May & Oct 2025; HTML reports → S3. |
| `register` | register the best model in the MLflow Model Registry and set the `production` alias. |

Run the whole thing, or resume / run one step:

```bash
# whole pipeline, both targets, in the cloud (AWS Batch)
./scripts/run_pipeline.sh --targets both --run-id 2024-prod

# resume from training; re-run only drift
./scripts/run_pipeline.sh --run-id 2024-prod --from train
./scripts/run_pipeline.sh --run-id 2024-prod --only drift --force

# fast local subsample (identical code path)
python -m bixi.pipeline --targets departure --run-id smoke \
  --local-dir ~/bixi_data --sample-stations 80 --n-trials 8 --flaml-budget 30
```

## 2. Modeling decisions

* **Two targets, one pipeline:** run per `departure` / `arrival` (operationally
  distinct — rebalancing cares about both).
* **Target = 15-minute `demand`** (zero-inflated counts) → we expose Poisson and
  Tweedie objectives alongside L2 and always clip predictions at 0. RMSE is the
  selection metric; we also report MAE, R² and Poisson deviance per split.
* **Splits (temporal, leakage-safe):** train = 2024, validation = May-2025,
  test = Oct-2025. 2025 baselines reference matching 2024 periods (Phase 1).
* **Advanced encoding:** high-cardinality `station_name` → frequency + smoothed
  target encoding, **fit on TRAIN only**; unseen stations fall back to global
  stats. `time_15min` is never a feature (ordering only).
* **Model selection:** a naive historical-average baseline, five candidate
  families, FLAML AutoML, and Optuna-tuned LightGBM all compete on validation
  RMSE; the winner is registered.

## 3. Responsible AI

* **Explainability:** SHAP (global + local) and LIME, saved as artifacts for the
  Streamlit Explainability page.
* **Fairness:** error parity (RMSE/MAE/R²/bias) across demand tiers (low/med/high)
  and geographic zones; disparity ratios are flagged with mitigation notes.

## 4. Drift — and an honest data caveat

All four drift types are computed (Evidently HTML + scipy KS flags). **But** BIXI
exposes trip history only to ~Apr-2026 and we use **2024 as the baseline for all
years** (2025/2026 historical-baseline features are derived from matching 2024
periods). That structurally limits how much engineered-feature drift can exist, so
drift here is an **analysis under known data constraints, with human review — not a
live monitor**. We deliberately do **not** ship a weekly cron (it cannot get fresh
labelled data). The concept-drift check flags an R² drop below threshold on new
labelled data.

## 5. Cloud architecture (all via AWS CDK, `infra/`)

| Stack | Resource |
|---|---|
| `BixiNetwork` | VPC, public subnets, **no NAT** (≈$0). |
| `BixiStorage` | S3 bucket for checkpoints / model artifacts / MLflow artifacts / reports. |
| `BixiMlflow` | MLflow tracking server on EC2 `t3.small` + S3 artifact store, Elastic IP, SG locked to the team CIDR. |
| `BixiBatch` | ECR training image (built by CDK) + AWS Batch managed EC2 compute + job definition. |

Deploy: `BIXI_ALLOW_CIDR=<your-ip>/32 ./scripts/deploy_infra.sh`.
Teardown: `./scripts/teardown.sh` — backs up all artifacts + an MLflow run snapshot
to `s3://insy684/bixi-mlops-backup/` (the CDK pipeline bucket auto-deletes on
destroy), then runs `cdk destroy --all`. Use `--backup-only` to keep the infra.

Source feature data is read from the existing `s3://insy684` bucket; everything
else is created and destroyed by CDK.

## 6. Known data issues flagged to Phase 1 (Rui)

Found while loading the feature tables (handled at load time; flagged for a clean
source fix — see `scripts/fix_misranged_features.py`):

* `2025_may_arrival_features` spans **May–Nov 2025** (not just May);
  `2025_oct_arrival_features` spans **Oct 2025–Jan 2026**. The departure files are
  correctly single-month. Month-filtered, arrivals match departures (totals within
  0.5%), so the data is correct — only the exported range is wrong.
* The 2024 files spill a few hours/days into 2025, and the 2024 grid appears to
  start at 05:00 (UTC-style) while the 2025 monthly files start 00:00 (local) —
  worth confirming the 2025 `hist_avg_demand` alignment to 2024 periods.
