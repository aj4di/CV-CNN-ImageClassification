# MNIST Digit Classification — Fashion-MNIST CNN

Multi-model comparison for 10-class image classification on the Fashion-MNIST dataset, built with TensorFlow/Keras.

## Dataset

- **Source**: `tensorflow.keras.datasets.fashion_mnist` (Zalando Research)
- 70,000 grayscale images (28 × 28 px) — 60,000 train / 10,000 test
- 10 classes: T-shirt, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot

See [data/data.md](data/data.md) for full schema and preprocessing details.

## Models

| Model | Architecture | Val Accuracy |
|---|---|---|
| `build_cnn_model` | Conv2D(64) → MaxPool → Conv2D(32) → MaxPool → BatchNorm → Conv2D(16) → Dense | — |
| `build_vgg16_models` | Frozen VGG16 (ImageNet, 224×224) → Dense(32) → Dense(16) → Dense(10) | — |
| `build_ann_model` | Flatten → Dense(100) → Dense(10) — SGD | ~97% |
| **`build_final_cnn`** | Conv2D(64) → MaxPool → Flatten → Dense(100) → Dense(10) — Adam | **~98%** |

**Best model**: `build_final_cnn` — simpler CNN with Adam (lr=0.01), categorical_crossentropy, 5 epochs.

## Project Structure

```
mnist-digit-classification/
├── data/
│   └── data.md          # Dataset schema, class labels, preprocessing, model results
├── models/
├── notebooks/
│   └── mnistproject.ipynb
├── src/
│   ├── config.py        # Imports and constants
│   ├── data_loader.py   # Load Fashion-MNIST, reshape, normalize
│   ├── preprocessing.py # One-hot encoding, train/test split
│   ├── model.py         # build_cnn_model, build_vgg16_models, build_ann_model, build_final_cnn
│   ├── train.py         # Fit all models
│   ├── evaluate.py      # Accuracy, classification report, confusion matrix
│   └── visualize.py     # Sample images, training curves, confusion matrix heatmap
└── requirements.txt
```

## Key Techniques

- **Sequential CNN**: Conv2D + MaxPooling2D + BatchNormalization + Dropout
- **Transfer Learning**: VGG16 (ImageNet weights, frozen base, custom Dense head)
- **Baseline ANN**: Flatten → Dense, trained with SGD + momentum
- Pixel normalization to [0, 1]; labels one-hot encoded
