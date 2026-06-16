# %% [markdown]
# # Data Preprocessing

# %%
from data_loader import *

# %%
# Scale bounding boxes to [0, 1] relative to image size
y_train_boxes_scaled = [tuple(v / IMG_SIZE for v in boxes) for boxes in y_train_boxes]
y_test_boxes_scaled = [tuple(v / IMG_SIZE for v in boxes) for boxes in y_test_boxes]

# %%
# Convert label lists to DataFrames for easier indexing
y_train = pd.DataFrame(y_train, columns=["Label"], dtype=object).astype('float')
y_test = pd.DataFrame(y_test, columns=["Label"], dtype=object).astype('float')

# Convert scaled boxes to numpy arrays
y_train_boxes_scaled = np.array(y_train_boxes_scaled, dtype='float32')
y_test_boxes_scaled = np.array(y_test_boxes_scaled, dtype='float32')

# %%
# Free memory from unscaled box lists
del y_train_boxes
del y_test_boxes

print("Label distribution:")
print(y_train.Label.value_counts())
