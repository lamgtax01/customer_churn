from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
import argparse
import os
import tarfile

# 🎯 Parse args
parser = argparse.ArgumentParser()
parser.add_argument("--train", type=str, required=True)
parser.add_argument("--validation", type=str, required=True)
parser.add_argument("--model_output", type=str, required=True)  # Must be s3://
args = parser.parse_args()

spark = SparkSession.builder.appName("ChurnPredictionSparkMLlib").getOrCreate()

# 🧾 Load data
train_df = spark.read.parquet(args.train)
val_df = spark.read.parquet(args.validation)

# 🧠 Feature setup
label_col = "Churn"
feature_cols = [col for col in train_df.columns if col != label_col]

indexer = StringIndexer(inputCol=label_col, outputCol="label")
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
lr = LogisticRegression(featuresCol="features", labelCol="label")
pipeline = Pipeline(stages=[indexer, assembler, lr])

# 🔁 Train model
model = pipeline.fit(train_df)

# ✅ Evaluate
predictions = model.transform(val_df)
predictions.select("label", "prediction", "probability").show(5)

# 💾 Save model locally in Spark format
local_model_dir = "/tmp/model"
model.write().overwrite().save(local_model_dir)

# 📦 Package into model.tar.gz for SageMaker
tar_path = "/tmp/model.tar.gz"
with tarfile.open(tar_path, "w:gz") as tar:
    tar.add(local_model_dir, arcname="model")

# 💡 Save .tar.gz to S3 using Spark (no boto3)
# Spark 3.3+ supports binary output — so we use workaround via Hadoop FS
if args.model_output.startswith("s3://"):
    hadoop = spark._jsc.hadoopConfiguration()
    fs = spark._jvm.org.apache.hadoop.fs.FileSystem.get(hadoop)
    s3_path = spark._jvm.org.apache.hadoop.fs.Path(args.model_output + "/model.tar.gz")
    local_path = spark._jvm.org.apache.hadoop.fs.Path("file://" + tar_path)
    fs.copyFromLocalFile(False, True, local_path, s3_path)

print(f"✅ Model packaged and uploaded to: {args.model_output}/model.tar.gz")
