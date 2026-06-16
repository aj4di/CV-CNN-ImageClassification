# %% [markdown]
# # Model Definitions
#
# Five models for 12-class plant seedling classification:
# - build_model1():   Simple CNN (3× Conv+MaxPool → Dense)
# - build_model2():   CNN + DataAugmentation + BatchNorm
# - build_vgg16():    VGG16 transfer learning (frozen, 64×64 input)
# - build_xception(): Xception transfer learning (128×128 input, GlobalAvgPool)
#
# All use Adam optimizer and categorical_crossentropy loss.

# %%
from config import *


def build_model1():
    """Simple 3-block CNN — baseline."""
    backend.clear_session()
    np.random.seed(42)
    random.seed(42)
    tf.random.set_seed(42)

    m = Sequential()
    m.add(Conv2D(128, (3, 3), activation='relu', padding='same', input_shape=(64, 64, 3)))
    m.add(MaxPooling2D((2, 2), padding='same'))
    m.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    m.add(MaxPooling2D((2, 2), padding='same'))
    m.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    m.add(MaxPooling2D((2, 2), padding='same'))
    m.add(Flatten())
    m.add(Dense(12, activation='relu'))
    m.add(Dropout(0.3))
    m.add(Dense(12, activation='softmax'))
    m.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return m


def build_model2():
    """CNN with BatchNormalization (used with ImageDataGenerator augmentation)."""
    backend.clear_session()
    np.random.seed(42)
    random.seed(42)
    tf.random.set_seed(42)

    m = Sequential()
    m.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(64, 64, 3)))
    m.add(MaxPooling2D((2, 2), padding='same'))
    m.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    m.add(MaxPooling2D((2, 2), padding='same'))
    m.add(BatchNormalization())
    m.add(Flatten())
    m.add(Dense(16, activation='relu'))
    m.add(Dropout(0.3))
    m.add(Dense(12, activation='softmax'))
    m.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return m


def build_vgg16():
    """VGG16 transfer learning — frozen base, custom head (64×64 input)."""
    from keras.applications.vgg16 import VGG16

    backend.clear_session()
    np.random.seed(42)
    random.seed(42)
    tf.random.set_seed(42)

    vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(64, 64, 3))
    for layer in vgg_model.layers:
        layer.trainable = False

    m = Sequential([
        vgg_model,
        Flatten(),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(12, activation='softmax'),
    ])
    m.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return m


def build_xception(img_size=128):
    """Xception transfer learning — GlobalAvgPool head (img_size×img_size input)."""
    backend.clear_session()
    np.random.seed(42)
    random.seed(42)
    tf.random.set_seed(42)

    base_model = tf.keras.applications.Xception(
        weights='imagenet', input_shape=(img_size, img_size, 3), include_top=False
    )
    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    predictions = tf.keras.layers.Dense(12, activation='softmax')(x)
    m = Model(inputs=base_model.input, outputs=predictions)
    m.compile(
        loss='categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(0.0005),
        metrics=['accuracy'],
    )
    return m
