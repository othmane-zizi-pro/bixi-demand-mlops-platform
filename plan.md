# BIXI Demand MLOps Platform — Master Build Plan (v2)

> **Course:** INSY 695 — Enterprise Data Science & ML in Production II (McGill, Desautels)
> **Repo:** https://github.com/ruizhaoca/bixi-demand-mlops-platform
> **Live demo (current, course-1 version):** https://bixidashboard.streamlit.app/
> **Milestones:** Final presentation **June 19, 2026** · Final submission (MyCourses) **June 21, 2026**
> **Goal:** turn the course-1 BIXI project into a production-grade MLOps platform, deployed to AWS via
> **infrastructure-as-code**, that scores **100% against the rubric** — without padding it with topics that
> don't add real value.

**v2 changes (per the team's review):** this version **follows the team's "Methodology Improvement & Labor
Division Plan"** for ownership (Sarah + Louis on data; Othmane on predictive modeling; Rui on clustering +
the app). We deliberately **do not chase every topic on the professor's reference list** — we drop the
low-value ones. Specifically vs. v1: **Causal Inference and Semi-supervised/Self-learning are removed**;
**Drift Monitoring + Drift Report move to Othmane** (drift is tied to the predictive model), while **Rui
covers the feature-drift analysis for clustering**.

---

## 0. Team & ownership

| Phase | Owner(s) | GitHub handle | Theme | Target |
|------:|----------|---------------|-------|--------|
| 1 | **Sarah Liu** + **Ruihe Zhang** (Louis) | _Sarah: confirm_ · `mudkipython` | Setup + Data: cleaning, EDA, feature engineering | Suggested Mon night · Hard Tue noon |
| 2 | **Othmane Zizi** | `othmane-zizi-pro` | Predictive modeling + tuning + MLflow + explainability + fairness + **drift** | Suggested Tue night · Hard Wed noon |
| 3 | **Rui Zhao** | `ruizhaoca` | Clustering + tuning + cluster feature-drift | Suggested Tue night · Hard Wed noon |
| 4 | **Rui Zhao** | `ruizhaoca` | Streamlit + serving + Docker + CI/CD + AWS IaC + docs/deck | Suggested Wed night · Hard Thu noon |

> **Open items for the deck (§9):** confirm **Sarah's GitHub handle** — the final presentation must list every
> member's GitHub id. Rui owns two phases (3 & 4) by his own request; Othmane offered to take on more, so
> **co-owning the AWS IaC/deployment with Rui is on the table — decide at the team sync.**

This matches the "Methodology Improvement & Labor Division Plan" (Part 1 Sarah+Louis, Part 2 Othmane, Part 3
Rui, Part 4 Rui). Louis's separate extended notes are superseded by this plan, per Louis.

---

## 1. What we are building (and what we are deliberately NOT building)

The base repo is the **course-1 BIXI project**: three notebooks (cleaning/EDA/FE, K-Means clustering, LightGBM
training) + a Streamlit app on Streamlit Community Cloud. It has **no MLOps**: no MLflow, Docker, CI/CD, tests,
model registry, drift monitoring, cloud IaC, or API. **That gap is the graded work — we build it.**

**Methodology improvements (graded value-adds):**
1. **15-minute granularity** instead of hourly (4× resolution, far more operationally useful).
2. **Departure vs. arrival demand predicted separately** via one shared pipeline run twice.
3. **All ~1,100+ stations** instead of only the top 400.
4. **Lagged, leakage-safe features** instead of naive historical means, with an explicit leakage audit.
5. **Multi-model + AutoML selection** instead of a single hand-picked LightGBM.
6. Full **MLflow tracking + registry**, **Evidently drift (4 types)**, **SHAP explainability**, **fairness
   analysis**, served via **FastAPI + Streamlit** on **AWS ECS Fargate behind an ALB**, all provisioned by
   **AWS CDK**.

**Deliberately OUT of scope** (low marginal value for this use case — per team decision):
- ❌ **Causal Inference** (DoWhy) — interesting but not central to a demand-forecasting product.
- ❌ **Semi-supervised / Self-learning** — we have labels; little to gain.
- ➖ **Entity embeddings**, **PCA/UMAP dimensionality reduction**, **PySpark feature job**, **batch/stream
  lineage** — kept only as *optional* enhancements; do them only if a phase finishes early.

The professor's topic list is a **menu**, not a checklist. We cover a strong, coherent subset end-to-end
rather than touching everything shallowly.

---

## 2. Target architecture

```
                          ┌─────────────────────────────────────────────┐
   GitHub (main)  ──push──►│ GitHub Actions CI/CD                          │
                          │  lint → pytest → build image → push ECR → deploy│
                          └───────────────┬─────────────────────────────────┘
                                          │ (CDK-provisioned infra)
            ┌─────────────────────────────▼───────────────────────────────┐
            │ AWS (ca-central-1), all via AWS CDK (Python)                  │
            │  access via IAM Identity Center (SSO) — no static root keys   │
            │                                                               │
            │  Amazon ECR ── image ──► ECS Fargate Service ──► ALB (HTTPS)  │
            │                              │  (FastAPI + Streamlit)         │
            │                              ├──► CloudWatch Logs/metrics      │
            │                              └──► reads model from S3 / MLflow │
            │                                                               │
            │  EC2 t3.small  ── MLflow Tracking Server ──► S3 (artifacts)    │
            │  S3 buckets: raw/feature data, mlflow-artifacts, drift-reports │
            └───────────────────────────────────────────────────────────────┘
```

**Region:** `ca-central-1` (Montreal). **AWS access:** the team account is on **AWS IAM Identity Center
(SSO)** — log in through the team's SSO portal (kept in our private channel, **never committed**). **IaC tool:**
**AWS CDK (Python)** under `infra/` — every resource is code; nothing is clicked in the console. Terraform is an
acceptable substitute, but pick one and don't mix.

---

## 3. Rubric coverage matrix (the 100% checklist)

Each row is owned by a phase and is "done" only when demonstrable (notebook output, MLflow run, screenshot,
test, or live endpoint).

| Course topic | Phase | How we satisfy it |
|---|:--:|---|
| Advanced imputation | 1 | KNN / IterativeImputer (MICE) for weather & demand gaps; missingness indicators |
| Feature engineering | 1 | 15-min slot/dow/month baselines, **lagged** features, weather joins, holiday flag |
| Data/Information leakage analysis | 1 | Temporal split; 2025 baselines reference matching 2024 periods; leakage audit notebook |
| Advanced encoding | 1 | Target/frequency encoding for high-cardinality `station_id` (entity embeddings = optional) |
| Clustering (unsupervised) | 3 | K-Means baseline **vs** GMM/Agglomerative/DBSCAN; auto-select by silhouette/DB/CH |
| Hyperparameter tuning + **AutoML** | 2 | Optuna Bayesian HPO logged to MLflow **+ FLAML** AutoML model search |
| Model performance | 2 | R²/RMSE/MAE per split (2024 train / May-25 val / Oct-25 test), tracked across versions |
| Explainability | 2 | SHAP global + local (force plots), LIME; surfaced on a Streamlit Explainability page |
| Fairness & Ethical AI | 2 | Prediction-error parity (RMSE/MAE) across demand tiers / boroughs; underserved-area check |
| MLflow / model tracking | 2,3 | Tracking server on EC2+S3; params/metrics/artifacts; Model Registry (Staging/Prod) |
| **Drift: feature/target/prediction/concept** | 2 | **Evidently AI, all 4 types, on the demand model**; weekly GitHub Actions cron → S3 reports |
| Cluster feature-drift | 3 | Input-distribution shift, cluster-assignment stability, centroid drift on the clustering model |
| Using GitHub toolset | all | feature branches, PRs, reviews, Project board, GitHub Secrets, Release tag |
| CI/CD | 4 | GitHub Actions: lint → test → build → push ECR → deploy ECS (zero-downtime rolling) |
| Docker containers | 4 | Dockerfile + `.dockerignore`; image versioned by commit SHA; local `docker compose` |
| ML model serving & deployment | 4 | FastAPI REST endpoint + Streamlit UI on ECS Fargate behind ALB (HTTPS) |
| Cloud-native application | 4 | Serverless Fargate, ALB, auto-scaling, 12-factor config, CloudWatch |
| Product & data management | 1,4 | Data dictionary, config-driven pipeline, README run-modes, GitHub Project board |
| (Optional) Security hygiene | 4 | Secrets in GitHub Secrets / SSM, least-privilege IAM, `.gitignore` hygiene, image scan |
| ~~Causal inference~~ / ~~Semi-supervised~~ | — | **Out of scope** by team decision (low marginal value) |

---

## 4. Authenticating your coding agent to GitHub (everyone, once)

Run your coding agent (Claude Code preferred; Codex works too) locally. Do this once per machine.

### Option A — GitHub CLI (recommended)
```bash
# Install gh (macOS: brew install gh), then:
gh auth login          # GitHub.com → HTTPS → "Login with a web browser" → paste the one-time code
gh auth setup-git      # let gh manage git credentials
gh auth status         # verify: Logged in as <your-handle>
```

### Option B — Personal Access Token
```bash
# Fine-grained PAT (Contents: R/W, Pull requests: R/W) scoped to the repo, then:
git config --global credential.helper store     # first push prompts once for the PAT
```

### Clone + set YOUR identity (so commits are yours, not the agent's)
```bash
gh repo clone ruizhaoca/bixi-demand-mlops-platform && cd bixi-demand-mlops-platform
git config user.name "<Your Name>"
git config user.email "<your-GitHub-no-reply-email>"   # keeps PII off commits, still attributes to you
```

> **All four members must be repo collaborators with _Write_** (Rui adds them in *Settings → Collaborators*).
> **Using Codex instead of Claude Code?** Copy the contents of `CLAUDE.md` into a local `CODEX.md` first so the
> agent gets the same project context.

---

## 5. Branching, PR & authorship conventions (everyone)

- **`main` is always stable** — the instructor must be able to clone and run it. Never push to `main` directly.
- **One feature branch per phase:** `phase-1-data`, `phase-2-modeling-drift`, `phase-3-clustering`,
  `phase-4-serving-deploy`. Sub-work uses `feat/<desc>` branches merged into the phase branch.
- **One PR per phase** against `main`, reviewed by ≥1 teammate, CI green, then squash-merge.
- **Authorship — commits & PRs must appear as the human, never the coding agent.** Every commit, branch, and
  PR is authored by the teammate driving the agent (their name + their own GitHub account). Set your
  `git config user.name`/`user.email` to *you*, and **do not** add agent attribution — no
  `Co-Authored-By: Claude`/agent trailer in commits and no "Generated with …" footer in PR bodies. The agent
  types; the human owns the contribution. Verify with `git log` / `gh pr view` before merging.
- **Project board:** a GitHub Project ("BIXI MLOps", Todo/In-Progress/Review/Done), one card per deliverable —
  itself graded under "GitHub toolset."
- **Branch protection on `main`:** require PR + passing CI before merge (Rui sets up in Phase 4, or earlier).
- **Secrets** (AWS, MLflow URI) live in **GitHub Secrets** + local `.env` (git-ignored) — never committed.

```bash
git checkout -b phase-N-<theme>
# ...work, commit...
git push -u origin phase-N-<theme>
gh pr create --base main --title "Phase N: <theme>" --body "Closes Phase N deliverables in plan.md."
```

---

## 6. The four phases

> Dependency order **1 → 2 → 3 → 4.** Phase 1 produces the feature tables everyone consumes. Phases 2 and 3
> can run **in parallel** off `main` once Phase 1 lands. Phase 4 integrates everything. Each phase ends with a
> handoff prompt (§7).

### Phase 1 — Sarah Liu & Ruihe Zhang (Louis) · Setup + Data Engineering · _in progress_
**Branch:** `phase-1-data` · **Suggested split:** Sarah → repo/AWS setup, cleaning & EDA; Louis → S3 data
connection & feature engineering (already underway). Ship as one coordinated Phase-1 PR.

**Build:**
1. **Production scaffold:** `src/bixi/` package (`config.py`, `data/ingest.py`, `data/clean.py`,
   `features/build.py`, `features/impute.py`), `notebooks/01_eda_feature_engineering.ipynb`, `tests/`,
   `pyproject.toml`, `Makefile`, `.pre-commit-config.yaml`, `.gitignore`, `.env.example`.
2. **AWS + GitHub setup** done via IAM Identity Center (SSO); **connect BIXI data to S3** (in progress).
3. **Ingest** 2024 + May/Oct-2025 BIXI trips + **15-minute** weather (if only hourly weather exists, repeat
   across the four 15-min slots and document it). Download from the open-data URLs in code, not by hand.
4. **Clean & reshape:** all stations; split into **departures** and **arrivals** sharing one pipeline.
5. **Advanced imputation:** KNN / IterativeImputer (MICE) + missingness indicators. *(rubric)*
6. **Feature engineering (15-min):** `station_slot_demand_24` (0–95), `station_dow_demand_24`,
   `station_month_demand_24`, `slot/dow/month`, `is_holiday`, weather features, **lagged demand**. *(rubric)*
7. **Leakage audit:** baselines from 2024 only; 2025 features reference matching 2024 periods; temporal split.
8. **Advanced encoding:** target/frequency encoding for `station_id` (entity embeddings optional). *(rubric)*
9. **Feature selection / multicollinearity** (VIF / importance); persist versioned feature tables to S3.
10. **Tests**; `make data` reproduces both feature tables end-to-end.

**Acceptance:** `make data` regenerates departures + arrivals tables from raw → S3; EDA + leakage audit
notebook renders; `pytest` green; `.env.example` documents every variable. **Handoff output:**
`features_departures.parquet`, `features_arrivals.parquet` + data dictionary (in S3).

---

### Phase 2 — Othmane Zizi · Predictive Modeling, Tuning, MLflow, Explainability, Fairness & Drift
**Branch:** `phase-2-modeling-drift`

**Build:**
1. **Training pipeline** `src/bixi/models/train.py` consuming the Phase-1 tables (run once per target:
   departures, arrivals).
2. **Multi-model + AutoML:** LightGBM baseline **plus** alternatives and **FLAML** AutoML search; pipeline
   **auto-selects** the best model for serving. *(rubric: model performance, AutoML)*
3. **Optuna** Bayesian HPO, every trial logged to MLflow. *(rubric: HPO)*
4. **MLflow** stand-up (defines conventions Rui reuses): tracking URI from env/Secrets, experiments per target,
   `log_params/metrics/model`, **register** best run → **Production** stage. *(rubric: MLflow/registry)*
5. **Evaluation:** R²/RMSE/MAE per split, logged + compared across versions.
6. **Explainability:** SHAP global + local (force plots) + LIME; save plots as artifacts for the app. *(rubric)*
7. **Fairness & ethical AI:** prediction-error parity across demand tiers / boroughs; flag underserved-area
   degradation + mitigation notes. *(rubric)*
8. **Drift monitoring + drift report (moved here):** **Evidently AI**, all **four** types on the demand model —
   - Feature drift: temperature, wind_speed, bad_weather, `station_slot_demand_24` (PSI>0.2 / KS p<0.05)
   - Target drift: demand per 15-min slot (Jensen-Shannon > 0.1)
   - Prediction drift: rolling 7-day model-output shift (> 1.5 trips/slot)
   - Concept drift: R² on new labelled data vs 0.63 baseline (< 0.55 ⇒ retrain alert)
   Upload the 2024 15-min reference to S3; save HTML reports to S3; add a **weekly GitHub Actions cron**
   `.github/workflows/drift_check.yml` (Mon 09:00 UTC + manual dispatch). *(rubric: all 4 drift types)*
9. `inference.py` that loads the Production model; **tests** for train/inference + drift thresholds.

> **Removed from v1:** causal inference and semi-supervised learning (team decision).

**Acceptance:** MLflow shows all Optuna trials + a registered Production model per target; SHAP/LIME + fairness
artifacts saved; all 4 drift types generate an Evidently report and the cron dispatch passes; `pytest` green.
**Handoff output:** registered models, the predict() contract serving calls, drift system + S3 report paths.

---

### Phase 3 — Rui Zhao · Clustering, Tuning & Cluster Feature-Drift
**Branch:** `phase-3-clustering`

**Build:**
1. **Clustering comparison** `src/bixi/cluster/train.py`: K-Means baseline **vs** GMM / Agglomerative / DBSCAN;
   **auto-select** best by silhouette / Davies-Bouldin / Calinski-Harabasz; log all to MLflow (reuse Phase-2
   conventions). *(rubric: clustering)*
2. **Operational clustering framework:** group stations by departure/arrival intensity across **morning rush /
   evening rush / other**, surfacing rebalancing risk (e.g. high-departure-low-arrival). Persist
   `station_clusters.csv` for the app.
3. **Cluster feature-drift analysis:** input-feature distribution changes, cluster-assignment stability, and
   centroid shifts over time (complements Othmane's model drift). *(rubric)*
4. *(Optional)* PCA / UMAP dimensionality reduction for cluster visualization — only if time allows.
5. **Tests** for cluster selection + drift checks.

**Acceptance:** clustering auto-selection logged to MLflow with `station_clusters.csv` produced; cluster
feature-drift report generated; `pytest` green. **Handoff output:** `station_clusters.csv` + cluster-drift
artifacts for the app's Monitoring page.

---

### Phase 4 — Rui Zhao · Serving, Containerization, CI/CD, AWS IaC, Docs & Presentation
**Branch:** `phase-4-serving-deploy`

**Build:**
1. **FastAPI service** `src/bixi/serve/api.py`: `/predict` (loads MLflow Production model), `/health`,
   `/clusters`; pydantic schemas. *(rubric: serving)*
2. **Streamlit app** `app.py` (refit from the base repo): **16-day forecast**, **custom input**, **clusters
   map** (PyDeck, dep/arr × time period), **Explainability** (Phase-2 SHAP), **Monitoring** (Evidently reports
   from S3). Calls the FastAPI backend.
3. **Docker:** `Dockerfile` (Python 3.12 + deps + model files) + `.dockerignore`; `docker-compose.yml` for
   local API+UI; image tagged by commit SHA. *(rubric: Docker)*
4. **CI/CD** `.github/workflows/deploy.yml`: push to `main` → lint (ruff/black) → `pytest` → build → push
   **ECR** → update **ECS** (rolling, zero-downtime); plus a PR-CI workflow (lint+test). *(rubric: CI/CD)*
5. **AWS IaC** under `infra/` with **AWS CDK (Python)**: `EcrStack`, `NetworkStack` (VPC/subnets/SGs),
   `FargateStack` (ECS cluster, task def, service, **ALB + ACM HTTPS**, CloudWatch, auto-scaling),
   `MlflowStack` (EC2 t3.small + **S3** artifacts, locked to team IPs), `StorageStack` (S3 for data/reports),
   least-privilege IAM. `cdk synth` must pass in CI; `cdk deploy` is Othmane's gated step (§8).
   *(co-ownership Othmane↔Rui — decide at sync)*
6. **README** with two run-modes: (a) full pipeline from scratch with your own cloud creds; (b) local Streamlit
   from prepared input files. Plus architecture diagram + per-phase summaries. *(product mgmt)*
7. **Branch protection + Project board** finalized; **Release tag** `v1.0-final`.
8. **LaTeX report → PDF via tectonic** covering **Section 5.9 Solution Presentation** (§9) + the
   **presentation deck**; the report aggregates each owner's phase write-up.

**Acceptance:** `docker compose up` runs API+UI locally; CI green on a PR; `cdk synth` succeeds; README covers
both run-modes; `report.pdf` builds via tectonic; deck lists team + GitHub ids + repo name. **This is the
integration phase — done only when the full stack works end-to-end.**

---

## 7. How each teammate starts their phase

The plan is built so you can just tell your agent your name and let it pick up the right phase from this file.

### ▶ Universal kickoff (paste this; change the name)
```
My name is <Sarah | Othmane | Rui | Louis/Ruihe>. We're building the BIXI Demand MLOps Platform
(repo: ruizhaoca/bixi-demand-mlops-platform). Read CLAUDE.md and plan.md in the repo root, find the phase I own
in §0/§6, and ship it end-to-end exactly as specified, meeting every acceptance criterion for that phase.
Authenticate to GitHub per §4 as me — commits and the PR must be authored by me, NOT the agent (no agent
attribution). Create the phase's feature branch, do the work, keep secrets out of git (.env is git-ignored),
then open ONE PR for my phase against main and stop. Only ask me if you hit a genuine blocker.
```

### Phase-specific reminders
- **Sarah & Louis (Phase 1):** branch `phase-1-data`; ingest 2024 + May/Oct-2025 trips + 15-min weather; all
  stations; departures/arrivals split; KNN/MICE imputation; 15-min lagged features; leakage audit;
  target/frequency encoding; persist feature tables to S3. (Louis: continue the S3 data connection you started.)
- **Othmane (Phase 2):** branch `phase-2-modeling-drift`; consume the Phase-1 tables; multi-model + FLAML +
  Optuna; MLflow tracking + Registry (Production); SHAP/LIME; fairness parity; **Evidently drift (4 types) +
  weekly cron**. No causal inference, no semi-supervised.
- **Rui (Phase 3):** branch `phase-3-clustering`; compare clustering models + auto-select; dep/arr × time-period
  operational clusters → `station_clusters.csv`; cluster feature-drift (assignment stability, centroid shift).
- **Rui (Phase 4):** branch `phase-4-serving-deploy`; FastAPI + Streamlit (5 pages) + Docker + CI/CD + AWS CDK
  IaC + README (two run-modes) + LaTeX/tectonic report + deck. `cdk synth` in CI; don't run `cdk deploy`
  (Othmane's step, §8).

---

## 8. Deployment runbook (Othmane runs this with AWS credentials)

AWS access is via **IAM Identity Center (SSO)**. When the phases are merged and `cdk synth` is green,
deployment is a single gated step. **When Othmane says "deploy," the agent does exactly this:**

```bash
# 0. Prereqs (once): Node ≥18, AWS CDK (npm i -g aws-cdk), infra/ Python deps, Docker running.

# 1. Log in via SSO (start URL is in the team's private channel — never committed):
aws configure sso            # enter SSO start URL + SSO region, then pick the account + role
aws sso login --profile bixi
export AWS_PROFILE=bixi
aws sts get-caller-identity   # confirm the right account

# 2. Provision infra (review the diff before applying):
cd infra && pip install -r requirements.txt
cdk bootstrap aws://<ACCOUNT_ID>/ca-central-1     # first time only
cdk diff
cdk deploy --all --require-approval any-change    # ECR, VPC, S3, EC2 MLflow, Fargate+ALB+ACM

# 3. App image: GitHub Actions deploy.yml builds+pushes on the next push to main, or do it manually.
#    CDK outputs print: ECR repo URI, ALB HTTPS URL, MLflow EC2 URL.

# 4. Smoke test:
curl https://<ALB_URL>/health     # FastAPI health
open  https://<ALB_URL>/          # Streamlit UI

# 5. GitHub Secrets for CI/CD + drift cron (CI can't use SSO interactively — use an OIDC role
#    or scoped IAM keys): AWS creds, MLFLOW_TRACKING_URI, ECR/ECS names (from CDK outputs).
```

**Teardown** (stop AWS charges after grading): `cdk destroy --all` (stop the EC2 MLflow box first if kept).

> Pinned: **AWS CDK (Python)**. If the team switches to Terraform, swap step 2 for
> `terraform init / plan / apply` under `infra/`; everything else is unchanged.

---

## 9. Report & presentation (Section 5.9 Solution Presentation)

The presentation (20%) and LaTeX report must cover the EnterpriseDataScience **Section 5.9 Solution
Presentation** map. Cover at least: **business problem & value** (BIXI rebalancing; why 15-min + dep/arr
split); **data** (sources, cleaning, imputation, FE, leakage); **methodology** (multi-model + AutoML, HPO,
clustering); **results** (metrics per split, SHAP insights, fairness findings, cluster map); **MLOps &
architecture** (MLflow, Docker, CI/CD, Fargate + ALB, IaC, 4-type drift); **responsible AI** (explainability,
fairness); **live demo** (deployed app + MLflow UI + an Evidently report); **roadmap / limitations**.

**Mandatory deck slide (submission requirement):** team name, **every member's name + GitHub id**
(`othmane-zizi-pro`, `mudkipython`, `ruizhaoca`, _Sarah — confirm_), and the repo name
`bixi-demand-mlops-platform`. **Only one teammate submits** on MyCourses with this info. Target length:
**7–8 minutes**, four presenters, with speaker notes (PowerPoint Cloud starting point).

**Report toolchain:** LaTeX → **tectonic** (`tectonic report/report.tex`) → `report.pdf`.

---

## 10. Timeline

| When | Milestone |
|---|---|
| Suggested Mon night / hard Tue noon | Phase 1 (Sarah + Louis) PR merged |
| Suggested Tue night / hard Wed noon | Phase 2 (Othmane) + Phase 3 (Rui) PRs merged |
| Suggested Wed night / hard Thu noon | Phase 4 (Rui) PR merged — full stack integrated |
| After integration | Othmane runs `cdk deploy`; rubric self-check; report (tectonic) + deck |
| **Thu June 19** | **Final presentation** (7–8 min, four presenters) |
| **Sat June 21** | **Final submission on MyCourses** (one teammate) |

Team sync to align (and a presentation mock run mid-week) as agreed in the group chat.

---

## 11. Local dev quickstart (everyone)

```bash
gh repo clone ruizhaoca/bixi-demand-mlops-platform && cd bixi-demand-mlops-platform
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # plus -e . once pyproject lands in Phase 1
cp .env.example .env                    # fill AWS / MLflow values; .env is git-ignored
pre-commit install                      # lint/format on commit
make data    # Phase 1+: regenerate feature tables   |  make train  # Phase 2+
make app     # run Streamlit locally                 |  pytest      # run tests
```

---

### Status checklist
- [ ] Phase 1 — Setup + Data (Sarah & Louis) · _in progress_
- [ ] Phase 2 — Modeling + Drift (Othmane) · PR merged
- [ ] Phase 3 — Clustering (Rui) · PR merged
- [ ] Phase 4 — Serving, CI/CD & AWS IaC (Rui) · PR merged
- [ ] `cdk deploy` run by Othmane · app live behind ALB
- [ ] Rubric self-check 100% · report (tectonic PDF) + deck ready for June 19
- [ ] One teammate submits on MyCourses · June 21
