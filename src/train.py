"""
Stage 3 — train.py
Builds a Sequential ANN:
    Flatten → Dense(ReLU) → Dropout → Dense(10, Softmax)
Compiles with Adam and sparse_categorical_crossentropy.
Trains on data/processed/, saves model to models/model.h5
and training history to models/history.csv.
All hyperparameters come from params.yaml.
"""

import os
import csv
import yaml
import numpy as np
import tensorflow as tf

PROCESSED_DIR = os.path.join("data", "processed")
MODELS_DIR = "models"
PARAMS_FILE = "params.yaml"


def main():
    with open(PARAMS_FILE, "r") as f:
        params = yaml.safe_load(f)["train"]

    dense_units = params["dense_units"]
    dropout_rate = params["dropout_rate"]
    learning_rate = params["learning_rate"]
    epochs = params["epochs"]
    batch_size = params["batch_size"]

    os.makedirs(MODELS_DIR, exist_ok=True)

    # Load processed data
    x_train = np.load(os.path.join(PROCESSED_DIR, "x_train.npy"))
    y_train = np.load(os.path.join(PROCESSED_DIR, "y_train.npy"))
    x_val = np.load(os.path.join(PROCESSED_DIR, "x_val.npy"))
    y_val = np.load(os.path.join(PROCESSED_DIR, "y_val.npy"))

    # Build ANN: Flatten → Dense(ReLU) → Dropout → Dense(10, Softmax)
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(dense_units, activation="relu"),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(10, activation="softmax"),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()

    history = model.fit(
        x_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_val, y_val),
        verbose=1,
    )

    # Save model
    model_path = os.path.join(MODELS_DIR, "model.h5")
    model.save(model_path)
    print(f"Model saved to {model_path}")

    # Save training history as CSV
    history_path = os.path.join(MODELS_DIR, "history.csv")
    with open(history_path, "w", newline="") as f:
        writer = csv.writer(f)
        keys = list(history.history.keys())
        writer.writerow(["epoch"] + keys)
        for i, values in enumerate(zip(*[history.history[k] for k in keys])):
            writer.writerow([i + 1] + list(values))

    print(f"Training history saved to {history_path}")


if __name__ == "__main__":
    main()
