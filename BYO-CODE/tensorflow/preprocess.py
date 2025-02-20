#!/usr/bin/env python3

import pandas as pd
import numpy as np


def main():
    # Load dataset
    df = pd.read_csv("/opt/ml/processing/input/rawdata1.csv")

    # Split dataset
    train, validate, test = np.split(df.sample(frac=1, random_state=42), [int(0.7*len(df)), int(0.85*len(df))])

    # Save CSV files
    train.to_csv("/opt/ml/processing/train/train.csv", index=False)
    validate.to_csv("/opt/ml/processing/validation/validation.csv", index=False)
    test.to_csv("/opt/ml/processing/test/test.csv", index=False)


if __name__ == "__main__":
    main()