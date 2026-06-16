# Plant Seedling Classification — CNN & Transfer Learning

12-class plant species classifier comparing custom CNNs against VGG16 and Xception transfer learning, using a Kaggle dataset of RGB seedling images.

## Dataset

- **Source**: [Kaggle Plant Seedlings Classification](https://www.kaggle.com/c/plant-seedlings-classification) (Aarhus University)
- 4,750 RGB images (128 × 128 px) across 12 plant species
- Stored as numpy array (`images2.npy`) + `Labels2.csv`
- Imbalanced classes: some species have 400+ images, others fewer

See [data/data.md](data/data.md) for full species list, preprocessing, and model results.

## Models

| Model | Architecture | Test Accuracy |
|---|---|---|
| `build_model1` | Simple 3-block CNN (128→64→32 filters) | — |
| `build_model2` | CNN + BatchNorm + Data Augmentation | ~74% |
| `build_vgg16` | Frozen VGG16 (ImageNet, 64×64) + Dense head | < 74% |
| `build_xception` | Frozen Xception (ImageNet, 128×128) + GlobalAvgPool | >90% |
| **`build_xception` + heavy aug** | Xception + rotation/zoom/flip augmentation | **>95%** |

**Best model**: Xception with heavy augmentation — >95% test accuracy.

VGG16 underperformed because ImageNet weights are poorly suited to plant images; Xception's more expressive architecture and heavier augmentation overcame this.

## Project Structure

```
plant-seedling-classification/
├── data/
│   └── data.md          # Dataset schema, class labels, preprocessing, model results
├── models/
├── notebooks/
│   └── Project_seeds.ipynb
├── src/
│   ├── config.py        # Imports, constants (IMG_SIZE=64/128)
│   ├── data_loader.py   # Load images2.npy, Labels2.csv from Google Drive
│   ├── preprocessing.py # BGR→RGB, resize, Gaussian blur, normalize, LabelBinarizer, train/test split
│   ├── model.py         # build_model1, build_model2, build_vgg16, build_xception
│   ├── train.py         # Fit all models; ImageDataGenerator for augmentation
│   ├── evaluate.py      # Accuracy, classification report per species
│   └── visualize.py     # Sample images, training curves, confusion matrix
└── requirements.txt
```

## Key Techniques

- **Transfer Learning**: VGG16 and Xception (frozen ImageNet weights, custom classification head)
- **Data Augmentation**: `ImageDataGenerator` — rotation, zoom, horizontal/vertical flips, width/height shifts
- **GlobalAveragePooling2D** in Xception head instead of Flatten (reduces parameters, improves generalization)
- Stratified 90/10 train/test split; pixel values normalized to [0, 1]
- Labels one-hot encoded with `sklearn.preprocessing.LabelBinarizer`
