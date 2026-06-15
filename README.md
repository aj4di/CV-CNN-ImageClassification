# CV-CNN-ImageClassification

A collection of computer vision projects using Convolutional Neural Networks (CNNs) built with TensorFlow and Keras, covering image classification and object detection tasks.

## Projects

### [plant-seedling-classification](plant-seedling-classification/Project_seeds.ipynb)
Plant Seedling classifier built from a Kaggle dataset of 4,750 RGB images (128x128x3) across multiple plant species.
- Builds and compares several CNN architectures
- Uses transfer learning with Xception
- Achieves 95%+ precision

### [mnist-digit-classification](mnist-digit-classification/mnistproject.ipynb)
CNN-based classifier for the MNIST handwritten digit / Fashion-MNIST dataset (60,000 grayscale images).
- Builds a CNN using Keras' Sequential API
- Covers data loading, preprocessing, and model evaluation

### [mnist-object-detection](mnist-object-detection/mnist-object-detection.ipynb)
Object detection on MNIST digits, extending classification with bounding-box regression.
- Combines classification and regression in a single CNN
- Demonstrates a simple object detection pipeline on a familiar dataset

## Tech Stack
- Python
- TensorFlow / Keras
- NumPy, Pandas, Matplotlib, Seaborn
