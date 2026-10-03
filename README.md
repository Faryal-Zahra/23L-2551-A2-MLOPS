# Fashion-ANN Pipeline

End-to-End ML Versioning with Git, DVC & Google Drive

## Overview

A fully-connected Artificial Neural Network (ANN) that classifies Fashion-MNIST images into 10 clothing categories. The entire pipeline is reproducible via a single `dvc repro` call, with all intermediate artifacts versioned through DVC and stored on Google Drive.

## Dataset

**Fashion-MNIST** — 70,000 grayscale 28×28 images across 10 clothing categories:
- T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot
- 60,000 training images, 10,000 test images
- Loaded via `tf.keras.datasets.fashion_mnist.load_data()` — no manual download required.

## Project Structure

```
fashion-ann-pipeline/
├── src/
│   ├── prepare.py       # Downloads & saves raw data to data/raw/
│   ├── preprocess.py    # Normalizes, splits, saves to data/processed/
│   ├── train.py         # Builds & trains ANN, saves models/model.h5
│   └── evaluate.py      # Loads model, writes metrics.json
├── data/
│   ├── raw/             # DVC-tracked raw .npy files
│   └── processed/       # DVC-tracked normalized arrays
├── models/              # DVC-tracked model.h5 + history.csv
├── reports/             # Confusion matrix images
├── params.yaml          # Central hyperparameters (single source of truth)
├── dvc.yaml             # Pipeline stage definitions
├── dvc.lock             # Auto-generated after dvc repro
└── metrics.json         # Test loss & accuracy (DVC metric)
```

## Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # macOS/Linux

# Install dependencies
pip install tensorflow dvc "dvc[gdrive]" pyyaml scikit-learn matplotlib

# Run the full pipeline
dvc repro
```

## DVC Remote (Google Drive)

```bash
# Configure remote (replace FOLDER_ID with your Drive folder ID)
dvc remote add -d gdrive_storage gdrive://<FOLDER_ID>

# Push artifacts
dvc push

# Pull artifacts
dvc pull
```

## Target

- Test accuracy: **≥ 85%**
- Pipeline: Fully reproducible via `dvc repro`
