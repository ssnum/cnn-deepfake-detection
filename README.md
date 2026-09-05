# CNNs for Deepfake Detection

Research code for an independent study applying convolutional neural networks to deepfake image detection. The project spans exploratory data analysis, classical machine learning baselines, transfer learning with established CNN architectures, and a custom-built model.

This repository consolidates the codebase behind the peer-reviewed paper:

> **Sheoran, S.** (2025). *Applications of Existing Convolutional Neural Networks to Deepfake Detection.* National High School Journal of Science (NHSJS).
> [Read the paper →](https://nhsjs.com/2025/applications-of-existing-convolutional-neural-networks-to-deepfake-detection/)

## Overview

Deepfakes present growing risks to cybersecurity, misinformation, and personal privacy. This project investigates how well existing, well-established CNN architectures, originally designed for general image classification, transfer to the task of distinguishing real images from AI-generated ones, and compares them against a custom-built CNN.

The research proceeded in stages, each reflected in a directory of this repository:

1. **Data exploration**: analyzing color intensity, brightness, and gradient magnitude patterns between real and AI-generated images.
2. **Classical ML baselines**: KNN, Random Forest, Logistic Regression, and SVM models trained on brightness values and raw pixel data, establishing a performance floor before moving to deep learning.
3. **CNN architectures**: VGG16 and VGG19 (built from scratch and via transfer learning), DenseNet121, and ResNet50, each fine-tuned for binary real-vs-fake classification.
4. **Custom CNN**: an original architecture iterated across several versions, adding data augmentation, batch normalization, and learning-rate scheduling.
5. **Transfer experiment**: adapting a CNN architecture originally designed for age classification to the deepfake detection task.

## Results

Validation accuracy by model, on the primary (CIFAKE) dataset:

| Model | Validation Accuracy |
|---|---|
| VGG16 (from scratch) | 50.25% |
| VGG19 (from scratch) | 50.25% |
| VGG (transfer learning) | 80.97% |
| DenseNet121 (transfer learning) | 83.78% |
| ResNet50 (transfer learning) | 93.55% |
| Age-classifier, best variant (4 conv, [512,512,2]) | 92.37% |
| **Custom CNN** | **96.96%** |

- The **custom CNN** was the best-performing model at **~97%** validation accuracy, ahead of every pretrained architecture and the best modified age-classification variant.
- VGG16 and VGG19 trained from scratch (no pretrained weights) essentially failed to learn, landing at chance level for binary classification; the transfer-learning versions of the same architectures performed far better, underscoring how much of the signal came from ImageNet pretraining rather than the architecture itself.
- When evaluated on a second, held-out dataset (Manjil Karki's Deepfake and Real Images, from Kaggle) to test generalization, the custom CNN's accuracy dropped to **81.1%**, and saliency maps showed the model's focus shifting and becoming less concentrated, evidence that some of its performance was dataset-specific rather than general.
- Full methodology, per-epoch metrics, and saliency map visualizations are in the [published paper](https://nhsjs.com/2025/applications-of-existing-convolutional-neural-networks-to-deepfake-detection/).

## Repository Structure

```
CNNs-for-Deepfake-Detection/
├── data_visualization/          # EDA: color, brightness, and gradient analysis of real vs. fake images
├── preprocessing/                # Dataset loading and preprocessing for CNN input
├── baseline_models/              # Classical ML baselines (KNN, Random Forest, Logistic Regression, SVM)
└── cnn_models/                   # CNN architectures and the custom model
    └── age_classifier_experiment/   # Transfer experiment adapting an age-classification CNN
```

Each directory has its own README describing its contents in more detail.

## Dataset

Models were trained and evaluated on the [CIFAKE: Real and AI-Generated Synthetic Images](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images) dataset from Kaggle: 120,000 images total, 60,000 real (from CIFAR-10) and 60,000 AI-generated (Stable Diffusion 1.4, generated to match CIFAR-10), split 80:20 into train/test, with the test set further split in half to create a validation set.

Generalizability was additionally tested on a second, held-out dataset: [Deepfake and Real Images](https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images) by Manjil Karki, also from Kaggle.

## Tech Stack

- **Deep learning:** TensorFlow, Keras
- **Classical ML:** scikit-learn
- **Data handling and visualization:** NumPy, Matplotlib

## Setup

```bash
git clone https://github.com/ssnum/CNNs-for-Deepfake-Detection.git
cd CNNs-for-Deepfake-Detection
pip install -r requirements.txt
```

Download the [CIFAKE dataset](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images) from Kaggle before running any preprocessing or training scripts. The scripts assume it has already been downloaded locally.

## Citation

If you reference this work, please cite the published paper:

```
Sheoran, S. (2025). Applications of Existing Convolutional Neural Networks to Deepfake
Detection. National High School Journal of Science.
https://nhsjs.com/2025/applications-of-existing-convolutional-neural-networks-to-deepfake-detection/
```

## Credits

- CNN architecture references: [Keras Documentation](https://keras.io/)
- Dataset: [CIFAKE on Kaggle](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images)
- Age classification architecture adapted from Aydogdu, M.F. and Demirci, M.F., *Age Classification Using an Optimized CNN Architecture*

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Author

**Saanvi Sheoran**
[GitHub](https://github.com/ssnum)
