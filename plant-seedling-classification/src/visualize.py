# %% [markdown]
# # Visualization

# %%
from evaluate import *


def plot_confusion_matrix(cm, title="Confusion Matrix"):
    f, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, linewidths=0.4, fmt='d', square=True, ax=ax)
    plt.title(title)
    plt.show()


def plot_training_curve(history, title="Model Accuracy"):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title(title)
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()


# %%
# Sample images grid
def plot_images(images, labels):
    keys = dict(labels['Label'])
    rows, cols = 3, 4
    fig = plt.figure(figsize=(10, 8))
    for i in range(cols):
        for j in range(rows):
            idx = np.random.randint(0, len(labels))
            ax = fig.add_subplot(rows, cols, i * rows + j + 1)
            ax.imshow(images[idx])
            ax.set_title(keys[idx])
    plt.tight_layout()
    plt.show()

plot_images(images, labels)

# %%
# Class distribution
sns.countplot(labels['Label'])
plt.xticks(rotation='vertical')
plt.title('Class Distribution')
plt.show()

# %%
# Training curves
plot_training_curve(history_1, "Model 1 — Simple CNN Accuracy")
plot_training_curve(history_2, "Model 2 — CNN + Aug + BatchNorm Accuracy")
plot_training_curve(history_vgg16, "Model 3 — VGG16 Transfer Learning Accuracy")
plot_training_curve(historyx, "Model 4 — Xception Transfer Learning Accuracy")

# %%
# Confusion matrices
plot_confusion_matrix(cm1, "Model 1 Confusion Matrix")
plot_confusion_matrix(cm2, "Model 2 Confusion Matrix")
plot_confusion_matrix(cm_vgg, "Model 3 (VGG16) Confusion Matrix")
plot_confusion_matrix(cm_xcep, "Model 4 (Xception) Confusion Matrix")

# %%
# Show individual predictions
for idx in [2, 33, 59, 36]:
    plt.figure(figsize=(2, 2))
    plt.imshow(X_test[idx])
    plt.show()
    pred = enc.inverse_transform(model2.predict(X_test_normalized[idx].reshape(1, 64, 64, 3)))
    true = enc.inverse_transform(y_test_encoded)[idx]
    print(f'Predicted: {pred} | True: {true}')
