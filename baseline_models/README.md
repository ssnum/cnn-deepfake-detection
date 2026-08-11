# Baseline Models

Classical machine learning models trained before moving to CNNs, used to establish a performance floor and build familiarity with model evaluation. Models were trained on both per-image brightness values and raw RGB pixel data.

## Contents

- `brightness_training.py`: KNN, Random Forest (including a bootstrapping variant), Logistic Regression, and SVM models trained on per-image brightness values.
- `full_image_training.py`: KNN, Random Forest, and Logistic Regression models trained on raw RGB pixel data.
- `model_metrics.py`: evaluation utilities for accuracy, precision, recall, F1 score, confusion matrix, ROC curve, and AUC.

## Requirements

- Kaggle dataset downloaded locally (see root [README](../README.md#dataset))
- Depends on preprocessing/data loading performed in `data_visualization/`
- scikit-learn, NumPy, Matplotlib
