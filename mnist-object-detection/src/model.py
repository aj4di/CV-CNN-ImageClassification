# %% [markdown]
# # Model Definition - VGG16 Dual-Head Object Detection
#
# Architecture:
#   Input (300×300×3)
#     └── VGG16 (frozen ImageNet weights, include_top=False)
#         └── Flatten
#             ├── Classifier head: Dropout(0.3) → Dense(10, softmax, name='label')
#             └── Regression head: Dense(128) → Dense(64) → Dense(32) → Dense(16) → Dense(4, sigmoid, name='bbox')
#
# Loss: sparse_categorical_crossentropy (labels) + mse (bounding boxes)
# Optimizer: SGD (lr=0.01, momentum=0.94)

# %%
from config import *


def get_model():
    vgg = VGG16(weights="imagenet", include_top=False, input_tensor=Input(shape=(300, 300, 3)))
    vgg.trainable = False

    x = Flatten()(vgg.output)

    # Classification branch
    classifier_head = Dropout(0.3)(x)
    classifier_head = Dense(10, activation='softmax', name='label')(classifier_head)

    # Bounding box regression branch
    reg_head = Dense(128, activation='relu')(x)
    reg_head = Dense(64, activation='relu')(reg_head)
    reg_head = Dense(32, activation='relu')(reg_head)
    reg_head = Dense(16, activation='relu')(reg_head)
    reg_head = Dense(4, activation='sigmoid', name='bbox')(reg_head)

    return Model(inputs=vgg.input, outputs=[classifier_head, reg_head])
