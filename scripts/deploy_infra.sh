#!/usr/bin/env bash
# Provision the BIXI cloud infrastructure with AWS CDK (uses your SSO creds).
#
#   export AWS_PROFILE=...            # or have temporary creds exported
#   export BIXI_ALLOW_CIDR=1.2.3.4/32 # who may reach MLflow :5000 / SSH
#   ./scripts/deploy_infra.sh
set -euo pipefail
cd "$(dirname "$0")/../infra"

export CDK_DEFAULT_ACCOUNT="$(aws sts get-caller-identity --query Account --output text)"
export CDK_DEFAULT_REGION="${AWS_DEFAULT_REGION:-us-east-2}"
ALLOW_CIDR="${BIXI_ALLOW_CIDR:-0.0.0.0/0}"

echo "Account=$CDK_DEFAULT_ACCOUNT Region=$CDK_DEFAULT_REGION allow_cidr=$ALLOW_CIDR"
pip install -q -r requirements.txt

npx --yes aws-cdk@2 bootstrap "aws://$CDK_DEFAULT_ACCOUNT/$CDK_DEFAULT_REGION"
npx --yes aws-cdk@2 deploy --all --require-approval never -c allow_cidr="$ALLOW_CIDR"
echo "Done. MLflow URL:"
aws cloudformation describe-stacks --stack-name BixiMlflow --region "$CDK_DEFAULT_REGION" \
  --query "Stacks[0].Outputs[?OutputKey=='MlflowPublicUrl'].OutputValue" --output text
