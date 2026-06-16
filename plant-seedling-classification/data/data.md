# Dataset: Plant Seedlings Classification

## Source
- **Kaggle / Aarhus University** — [Plant Seedlings Dataset](https://www.kaggle.com/c/plant-seedlings-classification)
- Collaboration between Aarhus University Signal Processing Group and University of Southern Denmark
- Stored as pre-processed numpy files on Google Drive: `images2.npy` + `Labels2.csv`

## Overview
| Property | Value |
|---|---|
| Total images | 4,750 |
| Image size (original) | 128 × 128 × 3 (RGB) |
| Image size (training) | 64 × 64 × 3 (resized) |
| Classes | 12 plant species |
| Format | Numpy array (.npy) + CSV labels |

## Classes (12 species)
| # | Species |
|---|---|
| 0 | Black-grass |
| 1 | Charlock |
| 2 | Cleavers |
| 3 | Common Chickweed |
| 4 | Common Wheat |
| 5 | Fat Hen |
| 6 | Loose Silky-bent |
| 7 | Maize |
| 8 | Scentless Mayweed |
| 9 | Shepherds Purse |
| 10 | Small-flowered Cranesbill |
| 11 | Sugar beet |

> Note: Dataset is imbalanced — some species have 400+ images, others fewer.

## Preprocessing Applied
- BGR → RGB conversion (images were created with OpenCV)
- Resized from 128×128 → 64×64 using `cv2.resize` (INTER_LINEAR) for Models 1–3
- Gaussian Blur (3×3) applied for Models 4–5 (Xception on original 128×128)
- Labels one-hot encoded with `sklearn.preprocessing.LabelBinarizer`
- Pixel values normalized: divided by 255.0 → [0, 1]
- Train/test split: 90% / 10%, stratified by class

## Models Trained

### Model 1: Simple CNN (`build_model1`)
- Conv2D(128) → MaxPool → Conv2D(64) → MaxPool → Conv2D(32) → MaxPool → Flatten → Dense(12, relu) → Dropout(0.3) → Dense(12, softmax)
- Optimizer: Adam | Loss: categorical_crossentropy | Epochs: 30 | Batch: 32

### Model 2: CNN + Data Augmentation + BatchNorm (`build_model2`) — Best base model
- Conv2D(64) → MaxPool → Conv2D(32) → MaxPool → BatchNorm → Flatten → Dense(16, relu) → Dropout(0.3) → Dense(12, softmax)
- Augmentation: `ImageDataGenerator(rotation_range=20)`
- Optimizer: Adam | Epochs: 25 | Batch: 64
- **Test accuracy: ~74%**

### Model 3: VGG16 Transfer Learning (`build_vgg16`)
- Frozen VGG16 (ImageNet, 64×64 input) → Flatten → Dense(32) → Dropout(0.2) → Dense(16) → Dense(12, softmax)
- Optimizer: Adam | Epochs: 25 | Batch: 64
- Did not outperform Model 2 (ImageNet weights not well-suited for plant images)

### Model 4: Xception Transfer Learning (`build_xception`) — Best model
- Frozen Xception (ImageNet, 128×128 input) → GlobalAveragePooling2D → Dropout(0.5) → Dense(1024, relu) → Dropout(0.5) → Dense(12, softmax)
- Optimizer: Adam (lr=0.0005) | Epochs: 50 | Batch: 16
- **Test accuracy: >90%**

### Model 5: Xception + Heavy Augmentation
- Same Xception model, fine-tuned with heavy augmentation (rotation 180°, zoom, flips, shifts)
- Epochs: 15 | Batch: 8
- **Test accuracy: >95%**
