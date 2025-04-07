import boto3
import json
import time
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

print("APP_INFO_KEY:", app_info_key)

# Load app_id
obj = s3.get_object(Bucket=bucket, Key=app_info_key)
print("obj:", obj)
app_id = json.loads(obj["Body"].read())["applicationId"]
print("app_id:", app_id)

# Load jobRunId
job_info = s3.get_object(Bucket=bucket, Key=f"{prefix}/emr-tracking/{job_type}_job_info.json")
print("job_info:", job_info)
job_id = json.loads(job_info["Body"].read())["jobRunId"]
print("job_id:", job_id)

# Poll until complete
print(f"⏳ Waiting for EMR job ({job_type}) ID {job_id}")
while True:
    state = emr.get_job_run(applicationId=app_id, jobRunId=job_id)["jobRun"]["state"]
    print(f"Status: {state}")
    if state in ["SUCCESS", "FAILED", "CANCELLED"]:
        break
    time.sleep(30)

job_run = emr.get_job_run(applicationId=app_id, jobRunId=job_id)["jobRun"]
state = job_run["state"]

# 🚀 Print logs
if "stateDetails" in job_run:
    print("State details:", job_run["stateDetails"])

if "jobRunId" in job_run:
    log_uri = job_run.get("configurationOverrides", {}).get("monitoringConfiguration", {}).get("s3MonitoringConfiguration", {}).get("logUri")
    print(f"💡 Log S3 URI (if set): {log_uri}")

print(f"🔗 Job run link (console): https://{args.region}.console.aws.amazon.com/emr-serverless/home?region={args.region}#/applications/{app_id}/job-runs/{job_id}")

if state != "SUCCESS":
    raise Exception(f"🚨 EMR job failed: {state}")
print("✅ EMR job completed successfully")
