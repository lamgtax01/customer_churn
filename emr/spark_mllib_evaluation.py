from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
import argparse
import json
import os

# 🎯 CLI args
parser = argparse.ArgumentParser()
parser.add_argument("--test", type=str, required=True)
parser.add_argument("--model_path", type=str, required=True)
parser.add_argument("--eval_output", type=str, required=True)
parser.add_argument("--label_column", type=str, required=True)
args = parser.parse_args()

# 🧠 Column config
label_col = args.label_column
indexed_label_col = "indexedLabel"

# 🚀 Spark session
spark = SparkSession.builder.appName("SparkEvaluation").getOrCreate()

# 📂 Load model & test set
model = PipelineModel.load(args.model_path)
test_df = spark.read.parquet(args.test)

# 🔮 Predictions
predictions = model.transform(test_df)

# 📊 Evaluators
binary_eval = BinaryClassificationEvaluator(
    labelCol=indexed_label_col,
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC"
)

multi_eval = MulticlassClassificationEvaluator(
    labelCol=indexed_label_col,
    predictionCol="prediction"
)

# ✅ Calculate metrics
metrics = {
    "accuracy": round(multi_eval.evaluate(predictions, {multi_eval.metricName: "accuracy"}), 4),
    "precision": round(multi_eval.evaluate(predictions, {multi_eval.metricName: "weightedPrecision"}), 4),
    "recall": round(multi_eval.evaluate(predictions, {multi_eval.metricName: "weightedRecall"}), 4),
    "f1": round(multi_eval.evaluate(predictions, {multi_eval.metricName: "f1"}), 4),
    "auc": round(binary_eval.evaluate(predictions), 4)
}

# 🧪 Build AWS-compatible JSON
aws_format_metrics = {
    "binary_classification_metrics": {
        metric_name: {
            "value": metric_value,
            "standard_deviation": "NaN"
        }
        for metric_name, metric_value in metrics.items()
    }
}

# 💾 Write evaluation.json locally
output_dir = "/tmp/evaluation"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "evaluation.json")

with open(output_path, "w") as f:
    json.dump(aws_format_metrics, f, indent=4)

# ☁️ Upload to S3 using Hadoop FS
hadoop_conf = spark._jsc.hadoopConfiguration()
fs = spark._jvm.org.apache.hadoop.fs.FileSystem.get(hadoop_conf)

src_path = spark._jvm.org.apache.hadoop.fs.Path("file://" + output_path)
dst_path = spark._jvm.org.apache.hadoop.fs.Path(args.eval_output + "/evaluation.json")

fs.copyFromLocalFile(False, True, src_path, dst_path)

print(f"✅ evaluation.json uploaded to {args.eval_output}/evaluation.json (AWS-compatible format)")
n.json")
