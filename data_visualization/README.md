# Data Visualization

Exploratory analysis of the [CIFAKE](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images) dataset (~20,000 real images, ~60,000 AI-generated images), used to identify measurable differences between real and AI-generated images before model design began.

## Contents

- `data_visualizations.py`: downloads and sorts the dataset, then computes and plots, for real vs. AI-generated images:
  - Average red, green, and blue color intensity per image
  - Average brightness per image
  - Gradient magnitude per image

## Requirements

- Kaggle dataset downloaded locally (see root [README](../README.md#dataset))
- Libraries imported at the top of the script (NumPy, Matplotlib, OpenCV/PIL as referenced)
