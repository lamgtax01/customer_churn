from pyspark.sql import SparkSession
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str)
parser.add_argument("--output", type=str)
args = parser.parse_args()

print("Preprocessing start ...")

spark = SparkSession.builder.appName("Preprocessing").getOrCreate()
df = spark.read.option("header", "true").csv(args.input)
df = df.dropna()
df.write.mode("overwrite").parquet(args.output)
