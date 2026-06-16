# %% [markdown]
# # Data Loading - Fashion-MNIST Dataset
#
# 60,000 grayscale 28x28 images across 10 clothing categories.
# Loaded directly from tensorflow.keras.datasets (no download needed).

# %%
from config import *
from tensorflow.keras.datasets import fashion_mnist

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)
