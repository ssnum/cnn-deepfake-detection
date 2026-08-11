# Age Classifier Experiment

An experiment adapting a CNN architecture originally designed for age classification — from Aydogdu, M.F. & Demirci, M.F., *Age Classification Using an Optimized CNN Architecture* — and applying it to deepfake detection instead.

Twelve specialized model variants were built and evaluated to see whether an architecture tuned for a different image classification task would transfer to detecting deepfakes.

## Contents

- `conv_model_2layer.py`, `conv_model_3layer.py`, `conv_model_4layer.py` — variants of the adapted architecture with 2, 3, and 4 convolutional layers respectively.
- `final_model.py` — the selected final variant of the adapted age-classifier architecture, retrained on the deepfake dataset.
- `final_model_results.py` — generates and saves results across all 12 model variants: best- and worst-performing confusion matrices and all 12 validation curves.

## Requirements

- Kaggle dataset downloaded locally (see root [README](../../README.md#dataset))
- TensorFlow, Keras, Matplotlib
