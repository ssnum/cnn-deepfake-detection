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
3. **CNN architectures**: VGG16 and VGG19 (built from scratch and via transfer learning), DenseNet121, and ResNet50 (partial fine-tuning: the first 7 residual blocks frozen, block 8 onward unfrozen), each evaluated for binary real-vs-fake classification.
4. **Custom CNN**: an original architecture iterated across several versions, adding data augmentation, batch normalization, and learning-rate scheduling.
5. **Transfer experiment**: adapting a CNN architecture originally designed for age classification to the deepfake detection task.
6. **Generalizability analysis**: saliency maps and cross-entropy loss compared across models and across a second, held-out dataset, to see how much of each model's accuracy is dataset-specific rather than general.

## Results

- The **custom CNN** achieved the highest validation accuracy at **97%** on the primary dataset, ahead of the fine-tuned **ResNet50** (94%) and the best modified age-classification variant (92.36%).
- When evaluated on a second, held-out dataset to test generalization, the custom CNN's accuracy dropped to **81.1%**, highlighting the generalization gap common to deepfake detectors trained on a single data distribution.
- Saliency maps showed clear differences in what each architecture attended to, and evidence of overfitting, underscoring the need for larger, more generalizable deepfake datasets.
- Full architecture-by-architecture comparisons and complete methodology are detailed in the [published paper](https://nhsjs.com/2025/applications-of-existing-convolutional-neural-networks-to-deepfake-detection/).

## Repository Structure

```
CNNs-for-Deepfake-Detection/
├── data_visualization/          # EDA: color, brightness, and gradient analysis of real vs. fake images
├── preprocessing/                # Dataset loading and preprocessing for CNN input
├── baseline_models/              # Classical ML baselines (KNN, Random Forest, Logistic Regression, SVM)
├── cnn_models/                   # CNN architectures and the custom model
│   └── age_classifier_experiment/   # Transfer experiment adapting an age-classification CNN
└── analysis/                     # Saliency maps and cross-dataset generalizability analysis
```

Each directory has its own README describing its contents in more detail.

## Dataset

All models were trained and evaluated on the [CIFAKE: Real and AI-Generated Synthetic Images](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images) dataset from Kaggle (approximately 20,000 real images and 60,000 AI-generated images).

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
