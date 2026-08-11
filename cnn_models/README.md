# CNN Models

CNN architectures evaluated for deepfake image classification, plus the custom model developed for this research. See the [published paper](https://nhsjs.com/2025/applications-of-existing-convolutional-neural-networks-to-deepfake-detection/) for full methodology and results.

## Contents

- `vgg16_architecture.py` — VGG16 built from scratch (layer by layer) for binary classification.
- `vgg19_architecture.py` — VGG19 built from scratch, extending VGG16 with additional convolutional layers.
- `vgg_transfer_learning.py` — VGG applied via transfer learning (pretrained weights) rather than trained from scratch.
- `densenet121_transfer_learning.py` — DenseNet121 via transfer learning, using densely connected layers for improved gradient flow.
- `resnet50_transfer_learning.py` — ResNet50 via transfer learning; the first 7 residual blocks are frozen as general feature extractors, with block 8 onward (final convolutional layers, global average pooling, and the output layer) fine-tuned on the deepfake dataset.
- `custom_model_v1.py`, `custom_model_v2.py`, `custom_model_v3.py` — iterative versions of the custom CNN, progressively adding batch normalization, data augmentation, and learning-rate/early-stopping callbacks. This line of experimentation produced the paper's best-performing model (97% validation accuracy).
- `age_classifier_experiment/` — a separate experiment adapting a CNN architecture originally designed for age classification to deepfake detection (see its own README).

## Requirements

- Kaggle dataset downloaded locally (see root [README](../README.md#dataset))
- Depends on the preprocessing steps in `preprocessing/`
- TensorFlow, Keras, Matplotlib
