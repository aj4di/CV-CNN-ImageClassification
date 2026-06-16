# %% [markdown]
# # Visualization

# %%
from evaluate import *

# %%
# Sample images grid
num_classes = 10
categories = np.unique(y_train)
rows, cols = 3, 5
fig = plt.figure(figsize=(10, 8))
for i in range(cols):
    for j in range(rows):
        idx = np.random.randint(0, len(y_train))
        ax = fig.add_subplot(rows, cols, i * rows + j + 1)
        ax.imshow(X_train[idx, :], cmap=plt.get_cmap('gray'))
        ax.set_title(categories[y_train[idx]])
plt.tight_layout()
plt.show()

# %%
# Class distribution
sns.countplot(y_train)
plt.title('Class Distribution')
plt.show()

# %%
# ANN training curve
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('ANN Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# %%
# ANN confusion matrix
f, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(cm_ann, annot=True, linewidths=0.4, fmt='d', square=True, ax=ax)
plt.title('ANN Confusion Matrix')
plt.show()

# %%
# Final CNN training curve
plt.plot(history_1.history['accuracy'])
plt.plot(history_1.history['val_accuracy'])
plt.title('CNN Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# %%
# Final CNN confusion matrix
f, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(cm_cnn, annot=True, linewidths=0.4, fmt='d', square=True, ax=ax)
plt.title('CNN Confusion Matrix')
plt.show()
