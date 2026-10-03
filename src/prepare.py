"""
Stage 1 — prepare.py
Downloads Fashion-MNIST via Keras and saves raw arrays to data/raw/.
No hyperparameters needed at this stage.
"""

import os
import numpy as np
import tensorflow as tf

OUTPUT_DIR = os.path.join("data", "raw")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Downloading Fashion-MNIST...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

    np.save(os.path.join(OUTPUT_DIR, "x_train.npy"), x_train)
    np.save(os.path.join(OUTPUT_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(OUTPUT_DIR, "x_test.npy"), x_test)
    np.save(os.path.join(OUTPUT_DIR, "y_test.npy"), y_test)

    print(f"Saved raw data to {OUTPUT_DIR}/")
    print(f"  x_train: {x_train.shape}, y_train: {y_train.shape}")
    print(f"  x_test:  {x_test.shape},  y_test:  {y_test.shape}")


if __name__ == "__main__":
    main()
