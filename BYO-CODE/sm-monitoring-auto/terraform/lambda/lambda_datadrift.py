import json
import boto3
import os

def lambda_handler(event, context):
    pipeline_name = os.getenv("DATA_DRIFT_PIPELINE")
    s3_event = event['Records'][0]['s3']
    
    print(f"Triggered by file: {s3_event['object']['key']}")
    
    sagemaker_client = boto3.client("sagemaker")
    response = sagemaker_client.start_pipeline_execution(PipelineName=pipeline_name)
    
    return {
        "statusCode": 200,
        "body": json.dumps("Data Drift Pipeline started")
    }