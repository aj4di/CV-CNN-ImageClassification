# Dataset: MNIST Object Detection

## Source
- Custom dataset built on MNIST handwritten digits
- Provided as a zip file via Google Drive

## Overview
| Property | Value |
|---|---|
| Training images | 9,000 |
| Test images | 1,000 |
| Image size | 300 × 300 pixels |
| Color | RGB (3 channels) |
| Classes | 10 (digits 0–9) |
| Task | Object detection (classification + bounding box regression) |

## File Structure
```
mnist_detection/
├── train/
│   ├── images/    0.png … 9000.png
│   └── labels/    0.txt … 9000.txt
└── test/
    ├── images/    0.png … 1000.png
    └── labels/    0.txt … 1000.txt
```

Each `.txt` label file contains one row with columns: `label, xmin, xmax, ymin, ymax`.

## Preprocessing Applied
- Images loaded with OpenCV (`cv2.imread`), resized to 300×300
- Bounding box coordinates scaled to [0, 1] by dividing by IMG_SIZE (300)
- Labels stored in a pandas DataFrame, cast to float
- Bounding boxes converted to `numpy.float32` arrays

## Model: VGG16 Dual-Head (`get_model`)

### Architecture
- **Base**: VGG16 (frozen, ImageNet weights, include_top=False, input 300×300×3)
- **Shared**: Flatten
- **Classifier head**: Dropout(0.3) → Dense(10, softmax) — predicts digit class
- **Regression head**: Dense(128) → Dense(64) → Dense(32) → Dense(16) → Dense(4, sigmoid) — predicts (xmin, ymin, xmax, ymax)

### Training
- Optimizer: SGD (lr=0.01, momentum=0.94)
- Loss: sparse_categorical_crossentropy (labels) + MSE (bounding boxes)
- Epochs: 10 | Batch size: 64 | Validation split: 10%

### Results
- Bounding box MAE ≈ 0.023 | MSE ≈ 0.001
- Label accuracy ≈ 67% (limited by low-quality ground-truth annotations)
