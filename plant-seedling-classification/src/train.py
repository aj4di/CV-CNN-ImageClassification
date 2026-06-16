# %% [markdown]
# # Model Training

# %%
from preprocessing import *
from model import build_model1, build_model2, build_vgg16, build_xception

# ── Model 1: Simple CNN ──────────────────────────────────────────────────────
# %%
model1 = build_model1()
model1.summary()
history_1 = model1.fit(
    X_train_normalized, y_train_encoded,
    epochs=30,
    validation_split=0.1,
    batch_size=32,
    verbose=2,
)

# ── Model 2: CNN + Augmentation + BatchNorm ───────────────────────────────────
# %%
model2 = build_model2()
model2.summary()
history_2 = model2.fit(
    train_datagen.flow(X_train_normalized, y_train_encoded, batch_size=64, seed=42, shuffle=False),
    epochs=25,
    steps_per_epoch=X_train_normalized.shape[0] // 64,
    validation_data=(X_test_normalized, y_test_encoded),
    verbose=1,
)

# ── Model 3: VGG16 Transfer Learning ─────────────────────────────────────────
# %%
model_vgg = build_vgg16()
model_vgg.summary()
history_vgg16 = model_vgg.fit(
    train_datagen.flow(X_train_normalized, y_train_encoded, batch_size=64, seed=42, shuffle=False),
    epochs=25,
    steps_per_epoch=X_train_normalized.shape[0] // 64,
    validation_data=(X_test_normalized, y_test_encoded),
    verbose=1,
)

# ── Model 4: Xception Transfer Learning ──────────────────────────────────────
# Uses original-size images (128×128) with Gaussian Blur
# %%
X_train2, X_test2, y_train2, y_test2 = train_test_split(
    np.array(images_gb), labels, test_size=0.1, random_state=42, stratify=labels
)
y_train_encoded2 = enc.fit_transform(y_train2)
y_test_encoded2 = enc.transform(y_test2)
X_train_normalized2 = X_train2.astype('float32') / 255.0
X_test_normalized2 = X_test2.astype('float32') / 255.0

xception_model = build_xception(img_size=128)
historyx = xception_model.fit(
    train_datagen.flow(X_train_normalized2, y_train_encoded2, batch_size=16, seed=42, shuffle=False),
    epochs=50,
    steps_per_epoch=X_train_normalized2.shape[0] // 16,
    validation_data=(X_test_normalized2, y_test_encoded2),
    verbose=1,
)

# ── Model 5: Xception + Heavy Augmentation ────────────────────────────────────
# %%
datagen_heavy = ImageDataGenerator(
    rotation_range=180,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=True,
)
datagen_heavy.fit(X_train2)
historyx2 = xception_model.fit_generator(
    datagen_heavy.flow(X_train2, y_train_encoded2, batch_size=8),
    epochs=15,
    validation_data=(X_test2, y_test_encoded2),
)
