# AWS Deployment Checklist

Use this checklist before deploying the FastAPI backend on EC2.

## 1. Secrets and Access

- Do not put `AWS_ACCESS_KEY_ID` or `AWS_SECRET_ACCESS_KEY` in code, notebooks, README, or GitHub.
- Rotate/delete any access key that has been shared in chat or committed anywhere.
- Prefer an EC2 IAM Role for S3 access.

## 2. Information Needed From the Team Lead

Ask the team lead to confirm:

```text
AWS_REGION = ?
S3_BUCKET = ?
MODEL_KEY = ?
META_KEY = ?
EC2 public IP = ?
EC2 SSH username = ubuntu or ec2-user?
EC2 SSH key file = ?
API port = 8000 or 8080?
Security Group allows inbound API port = yes/no?
EC2 IAM Role can read the model artifacts = yes/no?
```

Current expected values:

```text
AWS_REGION=us-east-2
S3_BUCKET=insy684
MODEL_KEY=bixi-models/model_lightgbm.txt
META_KEY=bixi-models/meta_lightgbm.pkl
```

The `MODEL_KEY` and `META_KEY` values are placeholders until the exact S3 paths are confirmed.

## 3. EC2 Smoke Checks

After SSH into EC2, run:

```bash
aws sts get-caller-identity
aws s3 ls s3://insy684/
```

If either command fails, the EC2 IAM Role or AWS CLI setup is not ready.

## 4. Docker Deployment

From the repository folder on EC2:

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

Then open:

```text
http://<ec2-public-ip>:8000/docs
```

Health check:

```bash
curl http://<ec2-public-ip>:8000/health
```

## 5. Expected Architecture

```text
Local client / Streamlit
    -> HTTP request
FastAPI backend on EC2
    -> boto3
S3 model artifacts
    -> prediction
FastAPI returns JSON
```
