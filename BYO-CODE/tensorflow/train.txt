#!/usr/bin/env python3

import pandas as pd
import numpy as np
import os
import argparse

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler


def parse_args():
    # ✅ Parse hyperparameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=0.001)

    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--validation', type=str, default=os.environ.get('SM_CHANNEL_VALIDATION'))
    # parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--model_dir', type=str, default=os.environ['SM_MODEL_DIR'])

    args = parser.parse_args()

    return args


def main():

    args = parse_args()

    # ✅ Load Preprocessed Data
    # train_df = pd.read_csv("/opt/ml/input/data/train/train.csv")
    # val_df = pd.read_csv("/opt/ml/input/data/validation/validation.csv")
    train_df = pd.read_csv(os.path.join(args.train, "train.csv"))
    val_df = pd.read_csv(os.path.join(args.train, "validation.csv"))

    # ✅ Separate Features and Target
    X_train, y_train = train_df.drop(columns=["Churn"]), train_df["Churn"]
    X_val, y_val = val_df.drop(columns=["Churn"]), val_df["Churn"]

    # ✅ Normalize Data
    X_train = X_train / np.max(X_train)
    X_val = X_val / np.max(X_val)

    # ✅ Reshape Data for Conv2D (Assuming Tabular Data → Expand to 2D)
    X_train = np.expand_dims(X_train, axis=-1)  # Reshape for Conv2D
    X_val = np.expand_dims(X_val, axis=-1)

    # ✅ Learning Rate Scheduler: Adjusts learning rate dynamically to improve convergence
    def lr_scheduler(epoch, lr):
        return lr * 0.95 if epoch > 3 else lr  # Decrease LR after 3 epochs

    lr_callback = LearningRateScheduler(lr_scheduler)

    # ✅ Early Stopping: Stops training if the model stops improving (prevents overfitting)
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=3, restore_best_weights=True
    )

    # ✅ Model Definition (Conv2D + Dense)
    model = Sequential([
        Conv2D(16, (1, 3), activation="relu", input_shape=(X_train.shape[1], 1, 1)),  # Convolution Layer: Extracts features from tabular data, making training more robust
        MaxPooling2D(pool_size=(2, 2)),  # Pooling: Extracts features from tabular data, making training more robust
        Flatten(),  # Flatten to 1D: Converts the output of Conv2D into a fully connected dense network
        Dense(128, activation="relu"),  # Fully Connected Layer:
        Dropout(0.3),  # Dropout Regularization: Prevents overfitting by randomly deactivating neurons
        Dense(64, activation="relu"),  # Another Dense Layer : Adds more capacity for learning complex patterns
        Dense(1, activation="sigmoid"),  # Output Layer (Binary Classification)
    ])

    # ✅ Compile Model
    model.compile(
        optimizer=Adam(learning_rate=args.learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    # ✅ Train Model with Callbacks
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[early_stopping, lr_callback]  # Apply callbacks
    )

    # ✅ Save Model for SageMaker
    model.save("/opt/ml/model")
    print("🎉 Model training completed and saved!")


if __name__ == "__main__":
    main()