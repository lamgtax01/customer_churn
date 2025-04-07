import boto3
import os
import json
import uuid

region = os.environ.get("AWS_REGION", "us-east-1")
app_name = os.environ.get("APP_NAME", f"sagemaker-pipeline-emr-{uuid.uuid4()}")
bucket = os.environ["S3_BUCKET"]
s3_key = os.environ["S3_KEY"]

emr = boto3.client("emr-serverless", region_name=region)
s3 = boto3.client("s3", region_name=region)

# Create EMR Serverless Application
response = emr.create_application(
    name=app_name,
    releaseLabel="emr-7.6.0",
    type="SPARK",
    initialCapacity={
        "Driver": {
            "workerCount": 3,
            "workerConfiguration": {
                "cpu": "4 vCPU",
                "memory": "20 GB",
                "disk": "40 GB"
            }
        },
        "Executor": {
            "workerCount": 3,
            "workerConfiguration": {
                "cpu": "8 vCPU",
                "memory": "20 GB",
                "disk": "40 GB"
            }
        }
    },
    maximumCapacity={
        "cpu": "100 vCPU",
        "memory": "300 GB",
        "disk": "500 GB"
    }
)

app_id = response["applicationId"]
print(f"✅ Created EMR Serverless App: {app_id}")

# Save to S3 for downstream steps
s3.put_object(
    Bucket=bucket,
    Key=s3_key,
    Body=json.dumps({"applicationId": app_id})
)

print(f"✅ Created EMR Serverless App: {app_id}")
