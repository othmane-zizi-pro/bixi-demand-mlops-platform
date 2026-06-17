"""S3 bucket for pipeline outputs: checkpoints, model artifacts, MLflow artifacts,
explainability/fairness/drift reports.

Source feature data stays in the existing ``insy684`` bucket (read-only here).
``auto_delete_objects`` + ``DESTROY`` keep teardown a single ``cdk destroy``.
"""

from aws_cdk import CfnOutput, RemovalPolicy, Stack
from aws_cdk import aws_s3 as s3
from aws_cdk import aws_ssm as ssm
from constructs import Construct


class StorageStack(Stack):
    def __init__(self, scope: Construct, cid: str, **kwargs) -> None:
        super().__init__(scope, cid, **kwargs)

        self.bucket = s3.Bucket(
            self,
            "PipelineBucket",
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_objects=True,
            encryption=s3.BucketEncryption.S3_MANAGED,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            enforce_ssl=True,
        )

        ssm.StringParameter(
            self,
            "PipelineBucketParam",
            parameter_name="/bixi/pipeline-bucket",
            string_value=self.bucket.bucket_name,
        )
        CfnOutput(self, "PipelineBucketName", value=self.bucket.bucket_name)
