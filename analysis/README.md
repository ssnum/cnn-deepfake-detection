# Analysis

Model-agnostic tools for the generalizability analysis described in the [published paper](https://nhsjs.com/2025/applications-of-existing-convolutional-neural-networks-to-deepfake-detection/): saliency maps and cross-dataset evaluation.

## Contents

- `saliency_maps.py`:
  - `compute_saliency_map`: vanilla-gradient saliency for a single model and image, the gradient of the predicted class score with respect to input pixels.
  - `plot_saliency_comparison`: plots an input image alongside each model's saliency map, for comparing how differently each architecture attends to the same input, and for spotting overfitting (saliency scattered across background vs. concentrated on face/artifact regions).
  - `generalization_gap`: evaluates a trained model's accuracy and cross-entropy loss on the primary dataset vs. a second, held-out dataset, and reports the accuracy drop between the two.

## Requirements

- A trained model object (any Keras model from `cnn_models/`) and loaded test data
- TensorFlow, Matplotlib, NumPy
