# import argparse
# import pathlib
# import boto3

import logging
import os
import pandas as pd

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, rand
from pyspark.ml.feature import StringIndexer
import os


def main():
    # Define the output paths
    input_data_path = "/opt/ml/processing/input"
    output_train_path = "/opt/ml/processing/train"
    output_validation_path = "/opt/ml/processing/validation"
    output_test_path = "/opt/ml/processing/test"

    # Initialize Spark session
    spark = SparkSession.builder.appName("PySpark Preprocessing").getOrCreate()

    # 1. Read data
    df = spark.read.csv(input_data_path, header=True, inferSchema=True)

    # 2. Drop the "Phone" feature column
    df = df.drop("Phone")

    # Print DataFrame schema and row count for reference (PySpark equivalent to df.shape in pandas)
    print(f"DataFrame schema: {df.schema}")
    print(f"Row count: {df.count()}, Column count: {len(df.columns)}")

    # 3. Log feature engineering start
    print("-- Feature engineering.")

    # 4. Change the data type of "Area Code" to string
    df = df.withColumn("Area Code", col("Area Code").cast("string"))

    # 5. Drop several other columns
    columns_to_drop = ["Day Charge", "Eve Charge", "Night Charge", "Intl Charge"]
    df = df.drop(*columns_to_drop)

    # 6. Convert categorical variables into dummy/indicator variables
    # For PySpark, use StringIndexer and one-hot encoding-like transformations
    categorical_columns = [col_name for col_name, dtype in df.dtypes if dtype == "string"]
    indexed_columns = []
    for col_name in categorical_columns:
        indexer = StringIndexer(inputCol=col_name, outputCol=f"{col_name}_indexed")
        df = indexer.fit(df).transform(df)
        indexed_columns.append(f"{col_name}_indexed")
    df = df.drop(*categorical_columns)  # Drop original categorical columns

    # 7. Create one binary classification target column
    df = df.withColumn(
        "Churn_binary",
        when(col("Churn?_True.") == 1, 1).otherwise(0)
    ).drop("Churn?_False.", "Churn?_True.")

    # 8. Split the data into train, validation, and test sets
    # Shuffle and split
    df = df.orderBy(rand())
    train_data, validation_data, test_data = df.randomSplit([0.7, 0.2, 0.1], seed=1729)

    # 9. Save the splits to CSV
    # Ensure output directories exist
    os.makedirs(output_train_path, exist_ok=True)
    os.makedirs(output_validation_path, exist_ok=True)
    os.makedirs(output_test_path, exist_ok=True)

    train_data.write.csv(output_train_path, header=False, mode="overwrite")
    validation_data.write.csv(output_validation_path, header=False, mode="overwrite")
    test_data.write.csv(output_test_path, header=False, mode="overwrite")


def main2():
    logger.info("Starting preprocessing...")

    # Define the output paths
    input_data_path = "/opt/ml/processing/input"
    output_train_path = "/opt/ml/processing/train"
    output_validation_path = "/opt/ml/processing/validation"
    output_test_path = "/opt/ml/processing/test"

    # Ensure output directories exist
    os.makedirs(output_train_path, exist_ok=True)
    os.makedirs(output_validation_path, exist_ok=True)
    os.makedirs(output_test_path, exist_ok=True)

    logger.info("-- Reading downloaded data.")

    # Define the input paths
    input_data_path = "/opt/ml/processing/input"

    # Read in csv
    data_file = f"{input_data_path}/data.csv"
    df = pd.read_csv(data_file)
    df = df.head(1000)

    # drop the "Phone" feature column
    df = df.drop(["Phone"], axis=1)
    print(df.shape)

    logger.info("-- Feature engineering.")

    # Change the data type of "Area Code"
    df["Area Code"] = df["Area Code"].astype(object)

    # Drop several other columns
    df = df.drop(["Day Charge", "Eve Charge", "Night Charge", "Intl Charge"], axis=1)

    # Convert categorical variables into dummy/indicator variables.
    model_data = pd.get_dummies(df)

    # Create one binary classification target column
    model_data = pd.concat(
        [
            model_data["Churn?_True."],
            model_data.drop(["Churn?_False.", "Churn?_True."], axis=1),
        ],
        axis=1,
    )

    # Split the data
    train_data, validation_data, test_data = np.split(
        model_data.sample(frac=1, random_state=1729),
        [int(0.7 * len(model_data)), int(0.9 * len(model_data))],
    )

    pd.DataFrame(train_data).to_csv(
        f"{output_train_path}/train.csv", header=False, index=False
    )
    pd.DataFrame(validation_data).to_csv(
        f"{output_validation_path}/validation.csv", header=False, index=False
    )
    pd.DataFrame(test_data).to_csv(
        f"{output_test_path}/test.csv", header=False, index=False
    )

    logger.info("-- Preprocessing successful !!!")


if __name__ == "__main__":
    main()
