# %% [markdown]
# # Data Preprocessing

# %%
from data_loader import *

# %%
# Resize images from 128×128 to 64×64 (reduces compute cost)
images_decreased = [
    cv2.resize(images[i], (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    for i in range(len(images))
]

# %%
# Apply Gaussian Blur (used for Xception models with original size)
images_gb = [cv2.GaussianBlur(images[i], ksize=(3, 3), sigmaX=0) for i in range(len(images))]

# %%
# Train / test split — stratified to handle class imbalance
X_train, X_test, y_train, y_test = train_test_split(
    np.array(images_decreased), labels, test_size=0.1, random_state=42, stratify=labels
)

print("X_train:", X_train.shape, "X_test:", X_test.shape)

# %%
# Encode labels to one-hot vectors using LabelBinarizer
enc = LabelBinarizer()
y_train_encoded = enc.fit_transform(y_train)
y_test_encoded = enc.transform(y_test)

# %%
# Normalize pixel values to [0, 1]
X_train_normalized = X_train.astype('float32') / 255.0
X_test_normalized = X_test.astype('float32') / 255.0

# %%
# ImageDataGenerator for data augmentation (used by Model 2+)
train_datagen = ImageDataGenerator(rotation_range=20, fill_mode='nearest')
