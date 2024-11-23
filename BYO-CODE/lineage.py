import argparse
import boto3
from sagemaker.lineage.artifact import Artifact
from sagemaker.lineage.association import Association

def attach_evaluation_report_to_lineage(evaluation_uri, model_package_arn):
    """
    Attach evaluation.json to SageMaker Model Lineage.
    """
    # Create the evaluation.json artifact
    evaluation_artifact = Artifact.create(
        artifact_name="EvaluationReport",
        source_uri=evaluation_uri,
        artifact_type="EvaluationReport",
        properties={
            "Source": "SageMaker Pipeline",
            "EvaluationType": "Metrics",
        },
    )

    # Create an association between the artifact and the model package
    Association.create(
        source_arn=model_package_arn,
        destination_arn=evaluation_artifact.artifact_arn,
        association_type="ContributedTo",
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-uri", type=str, required=True, help="S3 URI of evaluation.json")
    parser.add_argument("--model-package-arn", type=str, required=True, help="Model Package ARN")
    args = parser.parse_args()

    attach_evaluation_report_to_lineage(args.evaluation_uri, args.model_package_arn)



from sagemaker.processing import ScriptProcessor
from sagemaker.workflow.steps import ProcessingStep

# ScriptProcessor for attaching evaluation.json to lineage
lineage_processor = ScriptProcessor(
    image_uri="your-script-processor-image-uri",  # Replace with an appropriate container image
    command=["python3"],
    instance_type="ml.m5.large",
    instance_count=1,
    role=sagemaker.get_execution_role(),
)

# Custom step to attach evaluation.json to the lineage graph
attach_lineage_step = ProcessingStep(
    name="AttachEvaluationLineage",
    processor=lineage_processor,
    inputs=[
        ProcessingInput(
            source=evaluation_step.properties.ProcessingOutputConfig.Outputs["evaluation_output"].S3Output.S3Uri,
            destination="/opt/ml/processing/evaluation",
        ),
        ProcessingInput(
            source=register_model_step.properties.ModelPackageArn,
            destination="/opt/ml/processing/model_package",
        ),
    ],
    code="attach_lineage.py",
    arguments=[
        "--evaluation-uri",
        evaluation_step.properties.ProcessingOutputConfig.Outputs["evaluation_output"].S3Output.S3Uri,
        "--model-package-arn",
        register_model_step.properties.ModelPackageArn,
    ],
)

# Add all steps to the pipeline
pipeline = Pipeline(
    name="YourPipelineName",
    steps=[
        processing_step,
        training_step,
        evaluation_step,
        register_model_step,
        attach_lineage_step,
    ],
)