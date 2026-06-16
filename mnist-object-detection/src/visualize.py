# %% [markdown]
# # Visualization

# %%
from preprocessing import *

# %%
# Sample images with labels
f, axarr = plt.subplots(2, 2, figsize=(10, 10))
for ax, idx in zip(axarr.flatten(), [100, 200, 300, 400]):
    ax.imshow(X_train[idx])
    ax.set_title("Label: " + str(int(y_train['Label'].iloc[idx])), fontsize=15)
plt.tight_layout()
plt.show()

# %%
def plot_boxes(index):
    """Draw ground-truth bounding box on a training image."""
    (startX, endX, startY, endY) = y_train_boxes_scaled[index]
    image = X_train[index].copy()
    h, w, _ = image.shape
    startX, startY = int(startX * w), int(startY * h)
    endX, endY = int(endX * w), int(endY * h)
    label = int(y_train['Label'].iloc[index])
    thick = int((h + w) // 900)
    cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), thick)
    cv2.putText(image, str(label), (startX, startY - 12), 0, 1 * h, (0, 255, 0))
    plt.title("Label: " + str(label), fontsize=15)
    plt.imshow(image)
    plt.show()

plot_boxes(10)
plot_boxes(55)
plot_boxes(250)

# %%
def plot_predictions(index):
    """Draw predicted bounding box on a test image."""
    label_pred, bbox_pred = model.predict(X_test[index].reshape(1, 300, 300, 3))
    (startX, endX, startY, endY) = bbox_pred[0]
    label = np.argmax(label_pred)
    image = X_test[index].copy()
    h, w = image.shape[:2]
    startX, startY = int(startX * w), int(startY * h)
    endX, endY = int(endX * w), int(endY * h)
    cv2.rectangle(image, (startX, startY), (endX, endY), (255, 0, 0), 2)
    plt.title("Predicted label: " + str(label), fontsize=15)
    plt.imshow(image)
    plt.show()

plot_predictions(55)
plot_predictions(66)
plot_predictions(99)
