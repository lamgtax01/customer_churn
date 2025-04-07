from pyspark.sql import SparkSession
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str)
parser.add_argument("--output", type=str)
args = parser.parse_args()

print("Training start ...")

spark = SparkSession.builder.appName("Training").getOrCreate()
df = spark.read.parquet(args.input)
result = df.groupBy("Churn").count()
result.write.mode("overwrite").parquet(args.output)
