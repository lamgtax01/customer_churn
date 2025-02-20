#!/usr/bin/env python3

import os
import tensorflow as tf
import pandas as pd
import json
import tarfile
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


def main():

    model_path = "/opt/ml/processing/model/model.tar.gz"
    with tarfile.open(model_path) as tar:
        tar.extractall(path=".")

    print("Loading TensorFlow Model ...")
    model = tf.keras.models.load_model("./")

    print("Loading Test DataSet ...")
    test_df = pd.read_csv("/opt/ml/processing/test/test.csv")
    X_test = test_df.drop("Churn", axis=1)
    y_test = test_df["Churn"]

    # Evaluate
    # loss, accuracy = model.evaluate(test_df.drop("Churn", axis=1), test_df["Churn"])
    loss, accuracy = model.evaluate(X_test, y_test)

    # # Save results
    # with open("/opt/ml/processing/evaluation/evaluation.json", "w") as f:
    #     json.dump({"accuracy": accuracy}, f)

    # The metrics reported can change based on the model used, but it must be a specific name per (https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor-model-quality-metrics.html)
    report_dict = {
        "binary_classification_metrics": {
            "accuracy": {
                "value": accuracy,
                "standard_deviation": "NaN"
            }
        }
    }

    logger.info("Classification report:\n{}".format(report_dict))

    evaluation_output_path = os.path.join(
        "/opt/ml/processing/evaluation", "evaluation.json"
    )
    logger.info("Saving classification report to {}".format(evaluation_output_path))

    with open(evaluation_output_path, "w") as f:
        f.write(json.dumps(report_dict))

    print("Model evaluation evaluation.json saved on s3")


if __name__ == "__main__":
    main()