from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
import argparse
import json
import os
import tarfile
import boto3

# 🎯 Parse CLI
parser = argparse.ArgumentParser()
parser.add_argument("--test", type=str, required=True)
parser.add_argument("--model_path", type=str, required=True)  # s3://.../model.tar.gz
parser.add_argument("--eval_output", type=str, required=True)
parser.add_argument("--label_column", type=str, required=True)
args = parser.parse_args()

label_col = args.label_column
indexed_label_col = "indexedLabel"

# 🚀 Spark session
spark = SparkSession.builder.appName("SparkEvaluation").getOrCreate()

# 📁 Download model.tar.gz to /tmp
local_tar_path = "/tmp/model.tar.gz"
s3 = boto3.client("s3")

bucket = args.model_path.replace("s3://", "").split("/")[0]
key = "/".join(args.model_path.replace("s3://", "").split("/")[1:])

s3.download_file(bucket, key, local_tar_path)
print(f"✅ Downloaded model.tar.gz from s3://{bucket}/{key}")

# 📦 Extract model.tar.gz to /tmp/model
extract_path = "/tmp/model"
os.makedirs(extract_path, exist_ok=True)

with tarfile.open(local_tar_path, "r:gz") as tar:
    tar.extractall(path=extract_path)
print("📦 Extracted model.tar.gz to /tmp/model")

# 🔄 Load Spark MLlib model from extracted path
model = PipelineModel.load(os.path.join(extract_path, "model"))

# 🧾 Load test data
test_df = spark.read.parquet(args.test)

# 🔮 Predict
predictions = model.transform(test_df)

# 📊 Evaluators
binary_eval = BinaryClassificationEvaluator(
    labelCol=indexed_label_col, rawPredictionCol="rawPrediction", metricName="areaUnderROC"
)
multi_eval = MulticlassClassificationEvaluator(
    labelCol=indexed_label_col, predictionCol="prediction"
)

# ✅ Compute metrics
metrics = {
    "accuracy": round(multi_eval.evaluate(predictions, {multi_eval.metricName: "accuracy"}), 4),
    "precision": round(multi_eval.evaluate(predictions, {multi_eval.metricName: "weightedPrecision"}), 4),
    "recall": round(multi_eval.evaluate(predictions, {multi_eval.metricName: "weightedRecall"}), 4),
    "f1": round(multi_eval.evaluate(predictions, {multi_eval.metricName: "f1"}), 4),
    "auc": round(binary_eval.evaluate(predictions), 4)
}

# 🧪 AWS SageMaker JSON format
evaluation_json = {
    "binary_classification_metrics": {
        k: {"value": v, "standard_deviation": "NaN"} for k, v in metrics.items()
    }
}

# 💾 Save locally
output_dir = "/tmp/evaluation"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "evaluation.json")
with open(output_path, "w") as f:
    json.dump(evaluation_json, f, indent=4)

# ☁️ Upload evaluation.json using boto3
bucket = args.eval_output.replace("s3://", "").split("/")[0]
prefix = "/".join(args.eval_output.replace("s3://", "").split("/")[1:])
key = f"{prefix}/evaluation.json" if prefix else "evaluation.json"

s3.upload_file(output_path, bucket, key)
print(f"✅ evaluation.json uploaded to s3://{bucket}/{key}")

