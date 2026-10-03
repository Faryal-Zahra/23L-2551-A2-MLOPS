"""
Stage 2 — preprocess.py
Loads raw arrays, normalizes pixel values to [0, 1], splits a validation set,
and saves the resulting train/val/test arrays to data/processed/.
Hyperparameters are read from params.yaml.
"""

import os
import numpy as np
import yaml
from sklearn.model_selection import train_test_split

RAW_DIR = os.path.join("data", "raw")
PROCESSED_DIR = os.path.join("data", "processed")
PARAMS_FILE = "params.yaml"


def main():
    with open(PARAMS_FILE, "r") as f:
        params = yaml.safe_load(f)["preprocess"]

    test_size = params["test_size"]
    seed = params["seed"]

    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # Load raw data
    x_train = np.load(os.path.join(RAW_DIR, "x_train.npy"))
    y_train = np.load(os.path.join(RAW_DIR, "y_train.npy"))
    x_test = np.load(os.path.join(RAW_DIR, "x_test.npy"))
    y_test = np.load(os.path.join(RAW_DIR, "y_test.npy"))

    # Normalize pixel values to [0, 1]
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Split validation set from training data
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=test_size, random_state=seed
    )

    # Save processed arrays
    np.save(os.path.join(PROCESSED_DIR, "x_train.npy"), x_train)
    np.save(os.path.join(PROCESSED_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(PROCESSED_DIR, "x_val.npy"), x_val)
    np.save(os.path.join(PROCESSED_DIR, "y_val.npy"), y_val)
    np.save(os.path.join(PROCESSED_DIR, "x_test.npy"), x_test)
    np.save(os.path.join(PROCESSED_DIR, "y_test.npy"), y_test)

    print(f"Saved processed data to {PROCESSED_DIR}/")
    print(f"  x_train: {x_train.shape}, x_val: {x_val.shape}, x_test: {x_test.shape}")
    print(f"  Normalization: [0, 1],  val split: {test_size},  seed: {seed}")


if __name__ == "__main__":
    main()
