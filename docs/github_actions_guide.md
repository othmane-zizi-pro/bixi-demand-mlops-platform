# GitHub Actions CI Guide

This project uses GitHub Actions to automatically check code quality before changes are merged into `main`.

## What The CI Workflow Does

The workflow is defined in:

```text
.github/workflows/ci.yml
```

It runs four checks:

1. Install Python dependencies from `requirements.txt`.
2. Run the automated tests with `pytest -q`.
3. Build the FastAPI Docker image with `docker build -t bixi-demand-api .`.
4. Start the Docker container and call `GET /health`.

If all three steps pass, the pull request gets a green check. If any step fails, GitHub shows a red X and the team should fix the branch before merging.

## When It Runs

The workflow runs automatically when:

- A pull request is opened or updated against `main`.
- Code is pushed to `main`.
- Code is pushed to the current feature branch used for this assignment.

It can also be run manually from the GitHub website:

```text
Repository -> Actions -> CI -> Run workflow
```

## How This Fits The Team Git Workflow

Recommended process:

1. Create or switch to a feature branch.
2. Make local changes.
3. Commit and push the feature branch.
4. Open a pull request into `main`.
5. Wait for the CI check to finish.
6. Ask teammates to review the pull request.
7. Merge only after the CI check is green and the review is complete.

This protects `main` from broken tests, missing dependencies, or Docker build errors.

## What To Do If CI Fails

Open the failed workflow run and check which step failed:

- `Install dependencies`: usually a missing or incompatible package in `requirements.txt`.
- `Run tests`: usually a Python code or API behavior issue.
- `Build Docker image`: usually a Dockerfile, dependency, or missing file issue.
- `Smoke test Docker container`: usually a runtime dependency, app import, model loading, or container startup issue.

Fix the issue locally, run the same command locally, commit the fix, and push again. GitHub Actions will rerun automatically.

## Local Commands That Match CI

Run tests:

```bash
pytest -q
```

Build Docker image:

```bash
docker build -t bixi-demand-api .
```

Optional local API smoke test:

```bash
docker run --rm -p 8000:8000 -e MODEL_SOURCE=local bixi-demand-api
curl http://localhost:8000/health
```

## Current Deployment Boundary

This CI workflow verifies that the code can be tested, containerized, and started successfully. It does not deploy to EC2 yet.

Deployment should be added later only after the team confirms:

- EC2 public IP or host name
- SSH username and key handling
- S3 bucket and object keys for model artifacts
- EC2 IAM Role with S3 read permission
- Security Group rule for the API port

Until those are confirmed, CI should stop at test plus Docker build.
