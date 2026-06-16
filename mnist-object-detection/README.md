# MNIST Object Detection — VGG16 Dual-Head Model

Object detection on MNIST handwritten digits: simultaneous digit classification and bounding-box regression using a dual-head VGG16 network.

## Dataset

- Custom MNIST-based object detection dataset (from Google Drive zip)
- 9,000 train / 1,000 test images, 300 × 300 px, RGB
- 10 classes (digits 0–9)
- Labels: `label, xmin, xmax, ymin, ymax` per `.txt` file

See [data/data.md](data/data.md) for full schema and preprocessing details.

## Model: VGG16 Dual-Head (`get_model`)

**Architecture**:
```
Input (300×300×3)
  └── VGG16 (frozen, ImageNet weights, include_top=False)
      └── Flatten
          ├── Classification head: Dropout(0.3) → Dense(10, softmax)  ← digit label
          └── Regression head:    Dense(128) → Dense(64) → Dense(32) → Dense(16) → Dense(4, sigmoid)  ← bbox
```

**Training**:
- Optimizer: SGD (lr=0.01, momentum=0.94)
- Loss: `sparse_categorical_crossentropy` (labels) + `mse` (bounding boxes)
- Epochs: 10 | Batch: 64 | Validation: 10%

**Results**:
- Bounding box MAE ≈ 0.023 | MSE ≈ 0.001
- Label accuracy ≈ 67% (limited by annotation quality)

## Project Structure

```
mnist-object-detection/
├── data/
│   └── data.md          # Dataset schema, file structure, preprocessing, model results
├── models/
├── notebooks/
│   └── mnist-object-detection.ipynb
├── src/
│   ├── config.py        # Imports and constants (IMG_SIZE=300)
│   ├── data_loader.py   # Load images with OpenCV, parse label .txt files
│   ├── preprocessing.py # Resize to 300×300, normalize bbox coords to [0,1]
│   ├── model.py         # get_model() — VGG16 dual-head (classification + regression)
│   ├── train.py         # Compile and fit with two loss functions
│   ├── evaluate.py      # MAE/MSE for bbox, accuracy for labels
│   └── visualize.py     # Draw predicted bounding boxes on sample images
└── requirements.txt
```

## Key Techniques

- **Multi-task learning**: single backbone, two output heads (classification + regression)
- **Transfer Learning**: VGG16 frozen base (ImageNet weights)
- Bounding box coordinates normalized to [0, 1] via sigmoid output
- Labels loaded from per-image `.txt` files using pandas
