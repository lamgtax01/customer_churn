import boto3
import json
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--region", default="us-east-1")
args = parser.parse_args()

s3 = boto3.client("s3", region_name=args.region)
emr = boto3.client("emr-serverless", region_name=args.region)

bucket = os.environ["S3_BUCKET"]
prefix = os.environ["S3_PREFIX"]
job_type = os.environ["JOB_TYPE"]
app_info_key = os.environ["APP_INFO_KEY"]
entry_point = os.environ["ENTRY_POINT"]
emr_role = os.environ["EMR_ROLE"]
output_path = os.environ["OUTPUT"]
train = os.environ["TRAIN"]
validation = os.environ["VALIDATION"]

# Load app_id
obj = s3.get_object(Bucket=bucket, Key=app_info_key)
app_id = json.loads(obj["Body"].read())["applicationId"]

# Submit EMR Job
response = emr.start_job_run(
    applicationId=app_id,
    executionRoleArn=emr_role,
    jobDriver={
        "sparkSubmit": {
            "entryPoint": entry_point,
            "entryPointArguments": [
                "--train", train,
                "--validation", validation,
                "--model_output", output_path
            ]
        }
    },
    configurationOverrides={
        "monitoringConfiguration": {
            "s3MonitoringConfiguration": {
                "logUri": f"s3://{bucket}/{prefix}/logs/"
                }
            }
        }
)

job_id = response["jobRunId"]
print(f"✅ Submitted EMR job ({job_type}) with ID: {job_id}")
print(f"✅ Log will be stored at s3://{bucket}/{prefix}/logs/{app_id}/{job_id}/")

# Save job ID to S3
s3.put_object(
    Bucket=bucket,
    Key=f"{prefix}/emr-tracking/{job_type}_job_info.json",
    Body=json.dumps({"jobRunId": job_id})
)
