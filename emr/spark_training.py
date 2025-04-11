import subprocess
import sys

# 🔧 Inline install required packages
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "xgboost", "scikit-learn", "pandas", "boto3"])

import xgboost as xgb
import pandas as pd
from sklearn.metrics import accuracy_score
import argparse
import os
import boto3
import tarfile

from pyspark.sql import SparkSession

# 📥 Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--train", type=str, required=True)
parser.add_argument("--validation", type=str, required=True)
parser.add_argument("--model_output", type=str, required=True)
args = parser.parse_args()

spark = SparkSession.builder.appName("ChurnXGBoostTraining").getOrCreate()

# 📄 Load data from Parquet via Spark → Pandas
train_df = spark.read.parquet(args.train).toPandas()
val_df = spark.read.parquet(args.validation).toPandas()

# 🧪 Prepare features and labels
label_col = "Churn"
X_train = train_df.drop(columns=[label_col])
y_train = train_df[label_col]

X_val = val_df.drop(columns=[label_col])
y_val = val_df[label_col]

# 📊 Train XGBoost
dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)

params = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "max_depth": 4,
    "eta": 0.1
}

evals = [(dtrain, 'train'), (dval, 'eval')]
model = xgb.train(params, dtrain, num_boost_round=100, evals=evals, early_stopping_rounds=10)

# 🎯 Evaluate
preds = model.predict(dval)
pred_labels = (preds > 0.5).astype(int)
acc = accuracy_score(y_val, pred_labels)
print(f"✅ Validation Accuracy: {acc:.4f}")

# 💾 Save model to /tmp
model_path = "/tmp/xgboost-model.bst"
model.save_model(model_path)

# 📦 Package into model.tar.gz for SageMaker
tar_path = "/tmp/model.tar.gz"
with tarfile.open(tar_path, "w:gz") as tar:
    tar.add(model_path, arcname="xgboost-model.bst")

# ☁️ Upload to S3
bucket, *key_parts = args.model_output.replace("s3://", "").split("/")
key_prefix = "/".join(key_parts)
s3 = boto3.client("s3")
s3.upload_file(tar_path, bucket, f"{key_prefix}/model.tar.gz")

print(f"✅ Model artifact uploaded to: s3://{bucket}/{key_prefix}/model.tar.gz")
