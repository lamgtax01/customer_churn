from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
import argparse
import boto3
# import os
# import shutil
import tarfile

# 🎯 Parse args
parser = argparse.ArgumentParser()
parser.add_argument("--train", type=str, required=True)
parser.add_argument("--validation", type=str, required=True)
parser.add_argument("--model_output", type=str, required=True)
args = parser.parse_args()

# 🚀 Start Spark session
spark = SparkSession.builder.appName("ChurnPredictionSparkMLlib").getOrCreate()

# 🧾 Load data
train_df = spark.read.parquet(args.train)
val_df = spark.read.parquet(args.validation)

# 🧠 Prepare label and features
label_col = "Churn"
feature_cols = [col for col in train_df.columns if col != label_col]

indexer = StringIndexer(inputCol=label_col, outputCol="label")
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

# 🎯 Classifier
lr = LogisticRegression(featuresCol="features", labelCol="label")

# 🧪 Build pipeline
pipeline = Pipeline(stages=[indexer, assembler, lr])

# 🔁 Train
model = pipeline.fit(train_df)

# ✅ Evaluate
predictions = model.transform(val_df)
predictions.select("label", "prediction", "probability").show(5)

# 💾 Save model locally
local_model_dir = "/tmp/spark-mllib-model"
model.write().overwrite().save(local_model_dir)

# 📦 Package as model.tar.gz
tar_path = "/tmp/model.tar.gz"
with tarfile.open(tar_path, "w:gz") as tar:
    tar.add(local_model_dir, arcname="model")

# ☁️ Upload to S3
s3 = boto3.client("s3")
bucket, *key_parts = args.model_output.replace("s3://", "").split("/")
key_prefix = "/".join(key_parts)
s3.upload_file(tar_path, bucket, f"{key_prefix}/model.tar.gz")

print(f"✅ Spark MLlib model uploaded to: s3://{bucket}/{key_prefix}/model.tar.gz")
