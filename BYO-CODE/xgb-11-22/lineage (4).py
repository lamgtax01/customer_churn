import subprocess

# Install SageMaker SDK and Boto3 dynamically
subprocess.check_call(["pip", "install", "sagemaker", "boto3"])
# subprocess.check_call(["pip", "install", "sagemaker", "boto3", "sagemaker-studio-lineage-tracking"])

from datetime import datetime
import boto3
import os
import argparse
from sagemaker.lineage.artifact import Artifact
from sagemaker.lineage.association import Association
from sagemaker.lineage.context import Context
from sagemaker.lineage.action import Action
from sagemaker.lineage.link import CreateLineageLink
import sagemaker

import uuid
import re

# Set the AWS config
# os.environ["AWS_REGION"] = args.region
# boto3.setup_default_session(region_name=args.region)
# sagemaker_session = sagemaker.session.Session()


def generate_unique_artifact_name(base_name):
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    return f"{base_name}-{timestamp}"


def generate_unique_artifact_name2(base_name):
    unique_id = uuid.uuid4().hex[:8]  # Short UUID for uniqueness
    return f"{base_name}-{unique_id}"


def delete_existing_artifact_by_source_uri(source_uri, sagemaker_session):
    # List artifacts matching the source URI
    artifacts = Artifact.list(source_uri=source_uri)
    for artifact_summary in artifacts:
        if artifact_summary.source.source_uri == source_uri:
            # # Load the full artifact object
            artifact = Artifact.load(artifact_summary.artifact_arn)

            # Remove all associations for the artifact
            print(f"Removing eventual associations for artifact: {artifact.artifact_arn} ...")
            associations = Association.list(destination_arn=artifact.artifact_arn)
            for association in associations:
                print(f"Deleted association from {association.source_arn} to {association.destination_arn} ...")
                association = Association(
                    source_arn=association.source_arn,
                    destination_arn=association.destination_arn,
                    sagemaker_session=sagemaker_session,
                    )
                association.delete()

            # Delete the artifact
            print(f"Deleting existing artifact: {artifact.artifact_arn} ...")
            # Load the full artifact object
            artifact = Artifact.load(artifact_summary.artifact_arn)
            artifact.delete()


def extract_model_package_arn_without_version(input_string):
    """
    Extracts the ARN of a SageMaker model package without the version suffix.

    Args:
        input_string (str): The input string containing the SageMaker model package ARN.

    Returns:
        str: The extracted model package ARN without the version suffix, or None if no match is found.
    """
    # Define the regex pattern for SageMaker model package ARNs without the version suffix
    pattern = r"(arn:aws:sagemaker:[\w-]+:\d{12}:model-package/[\w-]+)(?:/\d+)"

    # Search for the pattern in the input string
    match = re.search(pattern, input_string)

    # Return the matched ARN without the version suffix or None if no match is found
    return match.group(1) if match else None


def get_existing_artifact_by_source_uri(source_uri):
    artifacts = Artifact.list(source_uri=source_uri)
    for artifact in artifacts:
        if artifact.source.source_uri == source_uri:
            # Load the full artifact object
            artifact_object = Artifact.load(artifact.artifact_arn)
            return artifact_object
    return None


def attach_evaluation_report_to_lineage(evaluation_uri, model_artifact_url, sagemaker_session):
    """
    Attach evaluation.json to SageMaker Model Lineage.
    """
    # Check if the artifact exists, delete before creating a new one
    # delete_existing_artifact_by_source_uri(evaluation_uri, sagemaker_session)

    # Create the evaluation.json artifact
    artifact_name = generate_unique_artifact_name("EvaluationMetrics")
    print(f"Artifact name: {artifact_name}")
    print(f"Artifact name2: {generate_unique_artifact_name2('EvaluationMetrics')}")


    # Check for existing artifact
    existing_artifact = get_existing_artifact_by_source_uri(evaluation_uri)

    if existing_artifact:
        evaluation_artifact = existing_artifact
        print(f"Using existing artifact: {evaluation_artifact.artifact_arn}")
    else:
        # Create a new artifact if it doesn't exist
        evaluation_artifact = Artifact.create(
            artifact_name=generate_unique_artifact_name("EvaluationReport"),
            source_uri=evaluation_uri,
            artifact_type="EvaluationReport",
            properties={
                    "Source": "SageMaker Pipeline",
                    "EvaluationType": "Metrics",
                },
            )
        print(f"Created new artifact: {evaluation_artifact.artifact_arn}")

    print("model_artifact_url:")
    print(model_artifact_url)

    response = CreateLineageLink(
        name="ModelEvaluationMetrics",
        source_artifact=model_artifact_url,
        target_artifact=evaluation_artifact.artifact_arn,
        source_type="Model",
        target_type="Report"    
    )

    # # Initialize the SageMaker client
    # client = boto3.client('sagemaker')

    # # List actions for a specific source URI
    # response = client.list_actions(
    #     SourceUri=model_artifact_url
    #     )

    # for action in response.get('ActionSummaries', []):
    #     print(f"Action ARN: {action['ActionArn']}")

    # # Retrieve the context for the model
    # model_contexts = Context.list(source_uri=model_artifact_url)
    # print("model_contexts:")
    # print(model_contexts)
    # for context_summary in model_contexts:
    #     print(f"Model Context ARN: {context_summary.context_arn}")

    # # Retrieve the action for model creation or registration
    # actions = Action.list(source_uri=model_artifact_url)
    # print("actions:")
    # print(actions)
    # for action_summary in actions:
    #     print(f"Action ARN: {action_summary.action_arn}")

    # # Create an association between the artifact and the model package
    # association = Association.create(
    #     source_arn=evaluation_artifact.artifact_arn,
    #     destination_arn=context_summary.context_arn,
    #     association_type="ContributedTo",
    # )
    # print(f"Artifact Association created: {association.artifact_arn}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-uri", type=str, required=True, help="S3 URI of evaluation.json")
    parser.add_argument("--model-package-arn", type=str, required=True, help="Model Package ARN")
    parser.add_argument("--model-artifact-url", type=str, required=True, help="model artifact s3URL")
    parser.add_argument("--region", type=str, default="us-east-1", help="AWS region")
    args = parser.parse_args()

    # Set the AWS region
    os.environ["AWS_REGION"] = args.region
    boto3.setup_default_session(region_name=args.region)
    sagemaker_session = sagemaker.session.Session()

    attach_evaluation_report_to_lineage(args.evaluation_uri, args.model_artifact_url, sagemaker_session)

    print("Lineage successfully added !")


if __name__ == "__main__":
    main()
