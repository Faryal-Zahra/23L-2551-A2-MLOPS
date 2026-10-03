"""
Stage 4 — evaluate.py
Loads the trained model and processed test set.
Computes test loss and accuracy, generates a confusion matrix image,
and writes all metrics to metrics.json at the project root.
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend (no display required)
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

PROCESSED_DIR = os.path.join("data", "processed")
MODEL_PATH = os.path.join("models", "model.h5")
REPORTS_DIR = "reports"
METRICS_FILE = "metrics.json"

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


def main():
    os.makedirs(REPORTS_DIR, exist_ok=True)

    # Load model and test data
    model = tf.keras.models.load_model(MODEL_PATH)
    x_test = np.load(os.path.join(PROCESSED_DIR, "x_test.npy"))
    y_test = np.load(os.path.join(PROCESSED_DIR, "y_test.npy"))

    # Evaluate
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test Loss:     {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # Predictions for confusion matrix
    y_pred = np.argmax(model.predict(x_test, verbose=0), axis=1)

    # Generate and save confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    disp.plot(ax=ax, xticks_rotation=45, colorbar=True)
    ax.set_title("Fashion-MNIST Confusion Matrix")
    plt.tight_layout()
    cm_path = os.path.join(REPORTS_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to {cm_path}")

    # Write metrics.json
    metrics = {
        "test_loss": round(float(test_loss), 4),
        "test_accuracy": round(float(test_accuracy), 4),
    }
    with open(METRICS_FILE, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics written to {METRICS_FILE}: {metrics}")


if __name__ == "__main__":
    main()
