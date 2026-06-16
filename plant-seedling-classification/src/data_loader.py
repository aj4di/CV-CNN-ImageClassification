# %% [markdown]
# # Data Loading - Plant Seedlings Dataset
#
# Dataset: 4,750 RGB images (128×128×3) from Kaggle / Aarhus University.
# 12 plant species. Stored as images.npy + Labels2.csv on Google Drive.

# %%
from config import *
from google.colab import drive
drive.mount('/content/drive')

# %%
# Load image array and labels
images = np.load('/content/drive/MyDrive/images2.npy')
labels = pd.read_csv('/content/drive/MyDrive/Labels2.csv')

print("Images shape:", images.shape)   # (4750, 128, 128, 3)
print("Labels shape:", labels.shape)

# %%
# Convert BGR → RGB (images were created with OpenCV which reads BGR)
for i in range(len(images)):
    images[i] = cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB)

categories = np.unique(labels)
print("Classes:", len(categories), categories)
