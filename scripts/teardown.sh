#!/usr/bin/env bash
# Tear down the BIXI cloud infrastructure and stop all AWS charges.
#
# WHAT GETS DESTROYED by `cdk destroy --all`:
#   - BixiMlflow : the MLflow EC2 (t3.medium) + Elastic IP  (the only always-on cost)
#   - BixiBatch  : Batch compute env / queue / job def + CloudWatch log group
#   - BixiStorage: the pipeline S3 bucket  *** and ALL its objects (auto_delete) ***
#                  -> model artifacts, MLflow artifacts, SHAP/fairness/drift reports
#   - BixiNetwork: the VPC
# The MLflow run history lives in SQLite ON the EC2 instance and is lost with it.
#
# Therefore this script FIRST backs everything up to the persistent (non-CDK)
# bucket s3://insy684/bixi-mlops-backup/ , then destroys.
#
#   ./scripts/teardown.sh                # back up, then destroy   (recommended)
#   ./scripts/teardown.sh --backup-only  # just back up, keep the infra running
#   ./scripts/teardown.sh --no-backup    # destroy without backing up
#
# NOT removed (shared / negligible cost): the CDK bootstrap stack (CDKToolkit) and
# the ECR image in the cdk-hnb659fds asset repo. Delete those manually if desired.
set -euo pipefail
REGION="${AWS_DEFAULT_REGION:-us-east-2}"
BACKUP=1; DESTROY=1
for a in "$@"; do
  case "$a" in
    --no-backup) BACKUP=0 ;;
    --backup-only) DESTROY=0 ;;
    *) echo "unknown arg: $a"; exit 2 ;;
  esac
done

BACKUP_DEST="s3://insy684/bixi-mlops-backup"
PIPELINE_BUCKET="$(aws ssm get-parameter --name /bixi/pipeline-bucket --region "$REGION" \
  --query Parameter.Value --output text 2>/dev/null || true)"
MLFLOW_URL="$(aws cloudformation describe-stacks --stack-name BixiMlflow --region "$REGION" \
  --query "Stacks[0].Outputs[?OutputKey=='MlflowPublicUrl'].OutputValue" --output text 2>/dev/null || true)"

if [ "$BACKUP" = 1 ]; then
  if [ -n "$PIPELINE_BUCKET" ] && [ "$PIPELINE_BUCKET" != "None" ]; then
    echo ">> Backing up s3://$PIPELINE_BUCKET  ->  $BACKUP_DEST/"
    aws s3 sync "s3://$PIPELINE_BUCKET" "$BACKUP_DEST/" --region "$REGION"
  else
    echo ">> No pipeline bucket found in SSM; skipping artifact backup."
  fi

  # Snapshot MLflow experiments/runs metadata (SQLite on EC2 is lost on destroy).
  if [ -n "$MLFLOW_URL" ] && [ "$MLFLOW_URL" != "None" ]; then
    echo ">> Snapshotting MLflow run metadata from $MLFLOW_URL"
    python3 - "$MLFLOW_URL" "$BACKUP_DEST" "$REGION" <<'PY' || echo "   (MLflow snapshot skipped — server unreachable)"
import sys, json, urllib.request, boto3
base, dest, region = sys.argv[1], sys.argv[2], sys.argv[3]
def post(path, body):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=20).read())
snap = {"experiments": [], "runs": {}, "registered_models": []}
exps = post("/api/2.0/mlflow/experiments/search", {"max_results": 1000}).get("experiments", [])
for e in exps:
    snap["experiments"].append(e)
    snap["runs"][e["experiment_id"]] = post(
        "/api/2.0/mlflow/runs/search",
        {"experiment_ids": [e["experiment_id"]], "max_results": 2000}).get("runs", [])
try:
    snap["registered_models"] = json.loads(urllib.request.urlopen(
        base + "/api/2.0/mlflow/registered-models/search", timeout=20).read()).get("registered_models", [])
except Exception:
    pass
b, k = dest.replace("s3://", "").split("/", 1)
boto3.client("s3", region_name=region).put_object(
    Bucket=b, Key=k.rstrip("/") + "/mlflow_runs_snapshot.json",
    Body=json.dumps(snap, indent=2, default=str).encode())
print(f"   wrote mlflow_runs_snapshot.json ({len(snap['experiments'])} experiments)")
PY
  fi
  echo ">> Backup complete at $BACKUP_DEST"
fi

if [ "$DESTROY" = 1 ]; then
  echo ">> Destroying all BIXI stacks (this is irreversible)..."
  cd "$(dirname "$0")/../infra"
  export CDK_DEFAULT_ACCOUNT="$(aws sts get-caller-identity --query Account --output text)"
  export CDK_DEFAULT_REGION="$REGION"
  export JSII_SILENCE_WARNING_UNTESTED_NODE_VERSION=1
  npx --yes aws-cdk@2 destroy --all --force
  echo ">> Done. All BIXI stacks destroyed; AWS charges stopped."
  echo "   Backup (if made): $BACKUP_DEST"
  echo "   Source feature data in s3://insy684/{processed-data,processed-data-clean,...} is untouched."
  echo "   Left in place (shared): CDKToolkit bootstrap stack + cdk asset ECR repo."
fi
