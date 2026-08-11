# Preprocessing

Shared preprocessing steps used to prepare the CIFAKE dataset for CNN input.

## Contents

- `cnn_preprocessing.py` — converts labels to one-hot encoded vectors and reshapes flattened image arrays back into `(32, 32, 3)` image tensors for both the training and test sets.

## Requirements

- Kaggle dataset downloaded locally (see root [README](../README.md#dataset))
- Assumes `X_train`, `X_test`, `y_train`, and `y_test` have already been loaded (see `data_visualization/` for the initial load/sort step)
