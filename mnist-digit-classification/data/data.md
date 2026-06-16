# Dataset: Fashion-MNIST

## Source
- **Built into Keras/TensorFlow** — `tensorflow.keras.datasets.fashion_mnist`
- Original dataset: [Fashion-MNIST (Zalando Research)](https://github.com/zalandoresearch/fashion-mnist)

## Overview
| Property | Value |
|---|---|
| Training images | 60,000 |
| Test images | 10,000 |
| Image size | 28 × 28 pixels |
| Color | Grayscale (1 channel) |
| Classes | 10 |

## Classes
| Label | Category |
|---|---|
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot |

## Preprocessing Applied
- Reshaped from `(N, 28, 28)` → `(N, 28, 28, 1)` (add channel dimension)
- Normalized pixel values: divided by 255.0 → range [0, 1]
- Labels one-hot encoded using `tf.keras.utils.to_categorical`

## Models Trained

### 1. CNN with BatchNorm (`build_cnn_model`)
- Conv2D(64) → MaxPool → Conv2D(32) → MaxPool → BatchNorm → Conv2D(16) → Flatten → Dense(32) → Dropout(0.25) → Dense(16) → Dense(10, softmax)
- Optimizer: Adam | Loss: binary_crossentropy | Epochs: 5

### 2. VGG16 Transfer Learning (`build_vgg16_models`)
- Frozen VGG16 (ImageNet weights, 224×224 input) → Flatten → Dense(32) → Dense(16) → Dense(10, softmax)
- Two variants tested (new_model, new_model2)
- Optimizer: Adam | Loss: binary_crossentropy

### 3. ANN (`build_ann_model`)
- Flatten → Dense(100, relu) → Dense(10, softmax)
- Optimizer: SGD (lr=0.01, momentum=0.9) | Loss: categorical_crossentropy | Epochs: 15 | Batch: 64
- Validation accuracy: ~97%

### 4. Final CNN — Best Model (`build_final_cnn`)
- Conv2D(64) → MaxPool → Flatten → Dense(100, relu) → Dense(10, softmax)
- Optimizer: Adam (lr=0.01) | Loss: categorical_crossentropy | Epochs: 5 | Batch: 64
- Validation accuracy: ~98%
