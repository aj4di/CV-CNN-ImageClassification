# CV-CNN-ImageClassification

A collection of computer vision projects using Convolutional Neural Networks (CNNs) built with TensorFlow and Keras, covering image classification and object detection tasks.

## Projects

### [plant-seedling-classification](plant-seedling-classification)
Plant Seedling classifier built from a Kaggle dataset of 4,750 RGB images (128x128x3) across multiple plant species.
- Builds and compares several CNN architectures
- Uses transfer learning with Xception
- Achieves 95%+ precision

### [mnist-digit-classification](mnist-digit-classification)
CNN-based classifier for the MNIST handwritten digit / Fashion-MNIST dataset (60,000 grayscale images).
- Builds a CNN using Keras' Sequential API
- Covers data loading, preprocessing, and model evaluation

### [mnist-object-detection](mnist-object-detection)
Object detection on MNIST digits, extending classification with bounding-box regression.
- Combines classification and regression in a single CNN
- Demonstrates a simple object detection pipeline on a familiar dataset

## Project Structure

Each project follows a standard ML project layout:

```
<project-name>/
├── data/              # Raw/processed datasets (not tracked in git)
├── notebooks/         # Original exploratory Jupyter notebook
├── src/               # Python script version of the analysis/pipeline
└── requirements.txt   # Python dependencies
```

The `src/*.py` files are converted from the original notebooks (in `# %%` cell-marker format, compatible with VS Code / Jupyter interactive mode).

## Tech Stack
- Python
- TensorFlow / Keras
- NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn
