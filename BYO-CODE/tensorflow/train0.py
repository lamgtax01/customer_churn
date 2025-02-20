#!/usr/bin/env python3

import argparse
import os
import tensorflow as tf
import pandas as pd


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--validation', type=str, default=os.environ.get('SM_CHANNEL_VALIDATION'))
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--model_dir', type=str, default=os.environ['SM_MODEL_DIR'])

    parser.add_argument('--epochs', type=int, default=10)

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    print("Training path:")
    print(args.train)

    print("Validation path:")
    print(args.validation)

    # Load Data
    # train_df = pd.read_csv("/opt/ml/input/data/train/train.csv")
    # val_df = pd.read_csv("/opt/ml/input/data/validation/validation.csv")

    train_df = pd.read_csv(os.path.join(args.train, "train.csv"))
    val_df = pd.read_csv(os.path.join(args.train, "validation.csv"))

    X_train = train_df.drop("Churn", axis=1)
    y_train = train_df["Churn"]
    X_val = val_df.drop("Churn", axis=1)
    y_val = val_df["Churn"]

    # Model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    # model.fit(train_df.drop("Churn", axis=1), train_df["Churn"], validation_data=(val_df.drop("Churn", axis=1), val_df["Churn"]), epochs=10)
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=args.epochs)

    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    model.save(args.model_dir)
    model.save("/opt/ml/model")
    print("Model saved at:")
    print(args.model_dir)


if __name__ == "__main__":
    main()