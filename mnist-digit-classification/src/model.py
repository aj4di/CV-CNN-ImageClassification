# %% [markdown]
# # Model Definitions
#
# Four models for Fashion-MNIST classification:
# - build_cnn_model(): CNN with BatchNorm (binary_crossentropy, adam)
# - build_vgg16_models(): VGG16 transfer learning (frozen, two variants)
# - build_ann_model(): Simple ANN with SGD
# - build_final_cnn(): Final CNN (Adam lr=0.01, categorical_crossentropy)

# %%
from config import *
from tensorflow.keras import backend
from keras.applications.vgg16 import VGG16


# %%
def build_cnn_model():
    backend.clear_session()
    np.random.seed(1)
    random.seed(1)
    tf.random.set_seed(1)

    m = Sequential()
    m.add(Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1), padding='same'))
    m.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    m.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    m.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    m.add(BatchNormalization())
    m.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
    m.add(Flatten())
    m.add(Dense(32, activation='relu'))
    m.add(Dropout(0.25))
    m.add(Dense(16, activation='relu'))
    m.add(Dense(10, activation='softmax'))
    return m


# %%
def build_vgg16_models():
    """VGG16 transfer learning (frozen weights). Expects 224x224 input."""
    backend.clear_session()
    vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    for layer in vgg_model.layers:
        layer.trainable = False

    new_model = Sequential([
        vgg_model,
        Flatten(),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(10, activation='softmax'),
    ])
    new_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    new_model2 = Sequential([
        vgg_model,
        Flatten(),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(16, activation='relu'),
        Dense(10, activation='softmax'),
    ])
    new_model2.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return new_model, new_model2


# %%
def build_ann_model():
    m = Sequential()
    m.add(Flatten(input_shape=(28, 28)))
    m.add(Dense(100, activation='relu'))
    m.add(Dense(10, activation='softmax'))
    opt = SGD(learning_rate=0.01, momentum=0.9)
    m.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return m


# %%
def build_final_cnn():
    backend.clear_session()
    np.random.seed(42)
    random.seed(42)
    tf.random.set_seed(42)

    m = Sequential()
    m.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)))
    m.add(MaxPooling2D((2, 2), padding='same'))
    m.add(Flatten())
    m.add(Dense(100, activation='relu'))
    m.add(Dense(10, activation='softmax'))
    opt = tf.keras.optimizers.Adam(learning_rate=0.01)
    m.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return m
