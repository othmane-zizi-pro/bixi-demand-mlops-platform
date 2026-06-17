# BIXI Station Hourly Demand Prediction
**Live demo:** https://bixidashboard.streamlit.app/

---
![1](https://github.com/user-attachments/assets/be920a54-8920-4684-b21a-f190809852b1)

---
![2](https://github.com/user-attachments/assets/fd0ca475-885d-44b0-9c9d-f8534ac9ef9e)

---
## Project Overview
This project builds an **end-to-end machine learning pipeline** to predict **hourly bike-sharing demand** for BIXI stations in Montreal. Using historical **BIXI trip data** and **Montreal weather data**, the pipeline performs data cleaning and feature engineering on **temporal and weather features**, trains a **LightGBM regression model** with **Bayesian hyperparameter optimization**, and groups stations into demand tiers using **K-Means clustering**. The **Streamlit app** integrates a **16-day weather forecast** from the **Open-Meteo API** and visualizes station clusters with a **PyDeck heatmap** to support station-level operational planning.

---
## Phase 2 — Production MLOps modeling pipeline

The course-2 work upgrades the modeling into a **resumable, cloud-native pipeline**
(`src/bixi/`, `python -m bixi.pipeline`): 15-minute demand, **departures and
arrivals**, **all** stations, leakage-safe encoding, **multi-model + FLAML AutoML +
Optuna**, **MLflow** tracking & registry, **SHAP/LIME**, **fairness**, and
**4-type drift** — trained in the cloud on **AWS Batch**, with the MLflow server,
S3, VPC and Batch all provisioned by **AWS CDK** (`infra/`).

- Design & decisions: [`docs/phase2_modeling.md`](docs/phase2_modeling.md)
- Run in the cloud: `./scripts/run_pipeline.sh --targets both --run-id 2024-prod`
  (resume a step with `--from train` / `--only drift`)
- Deploy infra: `BIXI_ALLOW_CIDR=<your-ip>/32 ./scripts/deploy_infra.sh` ·
  teardown (backs up first): `./scripts/teardown.sh`

### Where every asset lives (S3 + MLflow)

Two buckets. **`insy684`** is persistent (created in Phase 1, *not* CDK-managed).
The **CDK pipeline bucket** holds the pipeline outputs + MLflow artifacts and is
**deleted on `cdk destroy`** — `scripts/teardown.sh` backs it up to `insy684` first.
Its name is in SSM `/bixi/pipeline-bucket` and the `BixiStorage` CDK output (currently
`bixistorage-pipelinebucketb967bd35-icnkid23rfsa`).

```
s3://insy684/                         # PERSISTENT (source data + backups)
├── bixi-data/{2024,2025,2026}/       # raw BIXI open-data trip CSVs
├── weather-data/                     # 15-min Montreal weather (Open-Meteo): 2024, 2025-may, 2025-oct
├── processed-data/                   # Phase-1 feature tables (originals)
├── processed-data-clean/             # date-filtered canonical copies (this PR; scripts/fix_misranged_features.py)
├── bixi-models/                      # course-1 hourly model (legacy)
├── bixi-mlops/runs/…                 # earlier LOCAL smoke-run checkpoints (run-id "smoke")
└── bixi-mlops-backup/                # created by scripts/teardown.sh (artifacts + mlflow_runs_snapshot.json)

s3://<CDK pipeline bucket>/           # EPHEMERAL (cloud run outputs; backed up on teardown)
├── bixi-mlops/runs/<run-id>/<target>/    # <target> = departure | arrival ; cloud run-id = "cloud-2024"
│   ├── data/      encoder.pkl, tiers.json, data_summary.json
│   ├── train/     best_model.pkl, metrics.json  (selected model + R²/RMSE/MAE per split)
│   ├── explain/   shap_summary_beeswarm.png, shap_importance_bar.png, shap_waterfall_*.png,
│   │              shap_importance.csv, lime_instance_*.html
│   ├── fairness/  fairness_report.json  (error parity across demand tiers + geo zones)
│   ├── drift/     {feature,target,prediction}_drift_{2025_may,2025_oct}.html,
│   │              concept_regression_{2025_may,2025_oct}.html, drift_summary.json
│   └── register/  registered_model.json
├── mlflow/<experiment_id>/           # MLflow run artifacts (logged models, etc.)
└── mlflow-bootstrap/                 # MLflow EC2 bootstrap logs (debug)
```

**MLflow** (CDK output `BixiMlflow.MlflowPublicUrl`, classic 2.x UI): experiments
`bixi-demand-departure` / `bixi-demand-arrival` (every candidate + FLAML + Optuna run);
each best model registered with the **`production`** alias.

---
## Repository Structure

```
├── data/
│   ├── .gitattributes
│   └── model_df.zip           # Feature-engineered dataset: output of the first notebook and input to the last two notebooks
├── notebooks/
│   ├── data_cleaning_eda_feature_engineering.ipynb  
│   ├── model_clustering.ipynb # Station clustering analysis
│   └── model_lightgbm.ipynb   # Prediction model training & evaluation
├── app.py                     # Streamlit dashboard app; requires the five files below as inputs
├── api/
│   └── main.py                # FastAPI backend for model serving
├── src/
│   ├── predictor.py           # Shared model loading, feature building, and prediction logic
│   └── s3_io.py               # S3 readers used by the FastAPI backend
├── Dockerfile                 # Container entrypoint for the FastAPI backend
├── .env.example               # Safe configuration template; does not contain secrets
├── model_lightgbm.txt         # Trained LightGBM model
├── meta_lightgbm.pkl          # Model metadata & feature lookups
├── station_clusters.csv       # Station cluster assignments
├── requirements.txt           # Python dependencies
└── runtime.txt                # Python runtime version
```

---
## Workflow

### 1. Data Cleaning, EDA & Feature Engineering
**Notebook:** `data_cleaning_eda_feature_engineering.ipynb`

- **Data Sources:** [BIXI Montreal Open Data](https://bixi.com/en/open-data/) and [Montreal Weather Open Data](https://montreal.weatherstats.ca/download.html)
- **Cleaning Steps:**
  - Convert timestamps to datetime and filter the data to 2024 and May/Oct 2025
  - Fill missing values using column-wise means and linear interpolation
  - Remove invalid trips (unfillable missing values; outlier durations)
  - Filter to the top 400 stations by 2024 trip volume
- **Exploratory Analysis:**
  - Demand patterns by hour, day of week, and month
  - Holiday vs. non-holiday demand comparison
  - Distribution and correlation analyses of weather variables
- **Features engineered:**

| Feature | Description |
|---------|-------------|
| `station_hour_demand_24` | Mean 2024 demand for station × hour |
| `station_dow_demand_24` | Mean 2024 demand for station × day-of-week |
| `station_month_demand_24` | Mean 2024 demand for station × month |
| `hour`, `dow`, `month` | Temporal indicators |
| `is_holiday` | Quebec/Montreal public holiday flag |
| `temperature`, `feels_like` | Hourly temperature metrics |
| `wind_speed` | Wind speed in km/h |
| `bad_weather` | Binary flag (humidity > 85% and visibility < 10km) |

**Target Variable:** `total_demand` (sum of departures and returns per station per hour)

---
### 2. Station Clustering
**Notebook:** `model_clustering.ipynb`

- **Algorithm:** K-Means (k=3)
- **Clustering Feature:** Mean hourly demand per station (2024)
- **Output Clusters:**
  - **Low demand:** 247 stations (~7.4 avg trips/hour)
  - **Medium demand:** 120 stations (~12.2 avg trips/hour)
  - **High demand:** 33 stations (~19.2 avg trips/hour)
- **Validation:** Silhouette score indicates moderate-to-strong cluster separation

---
### 3. Model Training & Evaluation
**Notebook:** `model_lightgbm.ipynb`

- **Algorithm:** LightGBM (Gradient Boosting Decision Trees)
- **Data Split (chronological):**
  - Training: 2024 data (83%) — build 2024-based baseline features for forecasting
  - Validation: May 2025 (9%)
  - Test: Oct 2025 (8%) — May and Oct show moderate demand and are more representative months for validation and testing
- **Hyperparameter Tuning:** Bayesian optimization via Optuna (40 trials)
- **Evaluation Metrics:**

| Dataset | R² | RMSE | MAE |
|---------|-----|------|-----|
| Train (2024) | 0.72 | 5.08 | 3.21 |
| Validation (May 2025) | 0.64 | 5.75 | 3.73 |
| Test (Oct 2025) | 0.63 | 5.85 | 3.82 |

- **Model Interpretation:** SHAP analysis reveals top predictors are `station_hour_demand_24`, `station_month_demand_24`, and `temperature`

---
### 4. Streamlit Application
**File:** `app.py`

The dashboard provides three views:

- **16-Day Demand Forecast**
   - Use weather forecast data from [Open-Meteo API](https://open-meteo.com/) (cached daily)
   - Single time-point prediction or full-day hourly forecast
   - Interactive line charts of predicted demand with a weather overlay

- **Custom Input Forecast**
   - Manual weather parameter entry
   - Predictions for any future datetime

- **Station Clusters Visualization**
   - Interactive PyDeck heatmap with station-level tooltips
   - Filter by cluster (low/medium/high)

---
## FastAPI Backend and S3 Endpoint Mode

The project also includes a FastAPI backend for production-style model serving. The intended cloud architecture is:

```
Local frontend / Streamlit / API client
    -> calls FastAPI endpoint on EC2
    -> FastAPI loads model artifacts from local files or S3
    -> backend returns prediction JSON
```

### Security note

Do not hard-code AWS credentials in Python files, notebooks, README files, or GitHub. On EC2, the backend should access S3 through an attached IAM Role. Local development can use the AWS CLI profile or environment credentials outside the repository.

### Local development

By default, the API loads the local model artifacts already present in the repository:

```bash
pip install -r requirements.txt
uvicorn api.main:app --reload
```

Open:

```
http://localhost:8000/docs
```

Health check:

```bash
curl http://localhost:8000/health
```

Example prediction request:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "station": "10e avenue / Masson",
    "date": "2026-01-01",
    "hour": 8,
    "is_holiday": 0,
    "temperature": 22.5,
    "feels_like": 23.0,
    "wind_speed": 12.0,
    "bad_weather": 0
  }'
```

### EC2 / S3 deployment mode

Copy `.env.example` to `.env` only on the server if needed. Do not commit `.env`.

For EC2 deployment, set:

```bash
export AWS_REGION=us-east-2
export S3_BUCKET=insy684
export MODEL_SOURCE=s3
export MODEL_KEY=bixi-models/model_lightgbm.txt
export META_KEY=bixi-models/meta_lightgbm.pkl
```

The exact `MODEL_KEY` and `META_KEY` values must match the S3 paths uploaded by the team.

Docker run example:

```bash
docker build -t bixi-demand-api .
docker run -p 8000:8000 \
  -e AWS_REGION=us-east-2 \
  -e S3_BUCKET=insy684 \
  -e MODEL_SOURCE=s3 \
  -e MODEL_KEY=bixi-models/model_lightgbm.txt \
  -e META_KEY=bixi-models/meta_lightgbm.pkl \
  bixi-demand-api
```

The EC2 security group must allow inbound traffic on the API port, such as `8000`, for demo access.

### Tests

Run:

```bash
pytest -q
```

### CI

GitHub Actions runs the test suite and checks that the Docker image can build on pull requests into `main`.

Workflow file:

```
.github/workflows/ci.yml
```

Team guide:

```
docs/github_actions_guide.md
```

AWS deployment notes are in:

```
docs/aws_deployment_checklist.md
```

Model/S3/EC2 operations guide:

```
docs/model_s3_ec2_operations_guide.md
```

---
## Limitations
- **Limited feature set:** Features are restricted to historical demand, time, and weather. The model struggles to capture extreme peaks.
- **Departures vs. arrivals not separated:** The model predicts only total demand, which limits operational usefulness.
- **Station capacity not considered:** All stations are compared by usage intensity without accounting for dock capacity, reducing practical relevance.
- **Sample selection bias:** The dataset includes only the top 400 stations by demand (out of >1100). Lower-volume stations may exhibit different patterns.
- **Historical pattern reinforcement:** Heavy reliance on historical demand may miss emerging trends and changes.

---
## Team
Rui Zhao, Laura Manzanos Zuriarrain, Ibukunoluwa Adeleye, Mariam Gueye, Calvin Chun Fung Yip
