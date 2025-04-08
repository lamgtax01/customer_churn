from pyspark.sql import SparkSession
import argparse

# Args
parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, required=True, help="Input Parquet file (full s3 path)")
parser.add_argument("--output", type=str, required=True, help="S3 output directory base path")
args = parser.parse_args()

spark = SparkSession.builder.appName("CustomerChurnPreprocessing").getOrCreate()

# Read data
df = spark.read.parquet(args.input)

# Split dataset: 70% train, 15% val, 15% test
train_df, val_df, test_df = df.randomSplit([0.7, 0.15, 0.15], seed=42)

# Write splits to S3
train_df.write.mode("overwrite").parquet(f"{args.output}/train")
val_df.write.mode("overwrite").parquet(f"{args.output}/validation")
test_df.write.mode("overwrite").parquet(f"{args.output}/test")

print("✅ Data split and saved to S3")
