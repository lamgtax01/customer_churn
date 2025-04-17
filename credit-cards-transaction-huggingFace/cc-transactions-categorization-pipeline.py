# sagemaker_pipeline.py

import sagemaker
from sagemaker.workflow.steps import ProcessingStep, TrainingStep, CacheConfig
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.parameters import ParameterString, ParameterFloat
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.huggingface import HuggingFace
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.workflow.functions import JsonGet
from sagemaker.processing import ScriptProcessor
from sagemaker.workflow.properties import PropertyFile

# AWS setup
sagemaker_session = sagemaker.session.Session()
role = sagemaker.get_execution_role()

# Parameters
input_data = ParameterString(name="InputData", default_value="s3://my-bucket/transactions/")
model_approval_status = ParameterString(name="ModelApprovalStatus", default_value="PendingManualApproval")
accuracy_threshold = ParameterFloat(name="AccuracyThreshold", default_value=0.85)

# SKLearn Preprocessing
sklearn_processor = SKLearnProcessor(
    framework_version="1.2-1",
    role=role,
    instance_type="ml.m5.large",
    instance_count=1,
    base_job_name="preprocessing"
)

processing_step = ProcessingStep(
    name="PreprocessData",
    processor=sklearn_processor,
    inputs=[
        ProcessingInput(source=input_data, destination="/opt/ml/processing/input")
    ],
    outputs=[
        ProcessingOutput(output_name="train", source="/opt/ml/processing/train"),
        ProcessingOutput(output_name="validation", source="/opt/ml/processing/validation"),
        ProcessingOutput(output_name="test", source="/opt/ml/processing/test")
    ],
    code="preprocessing.py"
)

# Hugging Face Training
huggingface_estimator = HuggingFace(
    entry_point="train.py",
    source_dir="./",
    base_job_name="hf-train",
    instance_type="ml.p3.2xlarge",
    instance_count=1,
    role=role,
    transformers_version="4.26",
    pytorch_version="1.13",
    py_version="py39",
    hyperparameters={
        "epochs": 3,
        "per_device_train_batch_size": 16,
        "per_device_eval_batch_size": 32,
        "model_name": "bert-base-uncased"
    }
)

training_step = TrainingStep(
    name="TrainModel",
    estimator=huggingface_estimator,
    inputs={
        "train": processing_step.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
        "validation": processing_step.properties.ProcessingOutputConfig.Outputs["validation"].S3Output.S3Uri
    },
    cache_config=CacheConfig(enable_caching=True)
)

# Evaluation step
eval_processor = ScriptProcessor(
    image_uri=sagemaker.image_uris.retrieve("sklearn", sagemaker_session.boto_region_name),
    command=["python3"],
    instance_type="ml.m5.large",
    instance_count=1,
    role=role
)

evaluation_report = PropertyFile(
    name="EvaluationReport",
    output_name="metrics",
    path="evaluation/metrics.json"
)

evaluation_step = ProcessingStep(
    name="EvaluateModel",
    processor=eval_processor,
    inputs=[
        ProcessingInput(source=training_step.properties.ModelArtifacts.S3ModelArtifacts, destination="/opt/ml/processing/model"),
        ProcessingInput(source=processing_step.properties.ProcessingOutputConfig.Outputs["test"].S3Output.S3Uri, destination="/opt/ml/processing/test")
    ],
    outputs=[
        ProcessingOutput(output_name="metrics", source="/opt/ml/processing/evaluation"),
    ],
    code="evaluate.py",
    property_files=[evaluation_report]
)

# Condition and register
cond_step = ConditionStep(
    name="CheckAccuracy",
    conditions=[
        ConditionGreaterThanOrEqualTo(
            left=JsonGet(
                step_name=evaluation_step.name,
                property_file=evaluation_report,
                json_path="metrics.accuracy"
            ),
            right=accuracy_threshold
        )
    ],
    if_steps=[
        RegisterModel(
            name="RegisterModel",
            estimator=huggingface_estimator,
            model_data=training_step.properties.ModelArtifacts.S3ModelArtifacts,
            content_types=["application/json"],
            response_types=["application/json"],
            inference_instances=["ml.m5.large"],
            transform_instances=["ml.m5.large"],
            model_package_group_name="HuggingFaceTransactions",
            approval_status=model_approval_status
        )
    ],
    else_steps=[]
)

# Final pipeline
pipeline = Pipeline(
    name="CreditCardTransactionCategorization",
    parameters=[input_data, model_approval_status, accuracy_threshold],
    steps=[processing_step, training_step, evaluation_step, cond_step],
    sagemaker_session=sagemaker_session
)
