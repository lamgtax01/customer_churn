import argparse
import logging
import pathlib

import os
import boto3
import numpy as np
import pandas as pd

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


def main():
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
