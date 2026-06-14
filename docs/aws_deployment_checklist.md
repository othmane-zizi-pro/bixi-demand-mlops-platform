# AWS Deployment Checklist

Use this checklist before deploying the FastAPI backend on EC2.

## 1. Secrets and Access

- Do not put `AWS_ACCESS_KEY_ID` or `AWS_SECRET_ACCESS_KEY` in code, notebooks, README, or GitHub.
- Rotate/delete any access key that has been shared in chat or committed anywhere.
- Prefer an EC2 IAM Role for S3 access.

## 2. Confirmed Deployment Values

Current confirmed values:

```text
AWS_REGION=us-east-2
S3_BUCKET=insy684
MODEL_KEY=bixi-models/model_lightgbm.txt
META_KEY=bixi-models/meta_lightgbm.pkl
EC2 public IP=18.118.143.165
EC2 SSH username=ubuntu
API port=8000
Security Group=bixi-fastapi-sg
EC2 IAM Role=bixi-ec2-s3-read-role
```

Public endpoints:

```text
http://18.118.143.165:8000/health
http://18.118.143.165:8000/docs
http://18.118.143.165:8000/predict
```

If the team creates a new EC2 instance later, confirm the same fields again before deployment.

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
sudo docker build -t bixi-demand-api .

sudo docker run -d \
  --name bixi-demand-api \
  -p 8000:8000 \
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
curl http://18.118.143.165:8000/health
```

Prediction check:

```bash
curl -X POST http://18.118.143.165:8000/predict \
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
