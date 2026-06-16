# %% [markdown]
# # Data Loading - MNIST Object Detection Dataset
#
# Dataset: 9,000 train + 1,000 test images (300×300 RGB).
# Each image has a matching .txt label file with: label, xmin, xmax, ymin, ymax.
# Mount Google Drive and unzip before running.

# %%
from config import *
from google.colab import drive
drive.mount('/content/drive')

# %%
# Unzip dataset (run once)
# !unzip -q "/content/drive/MyDrive/Advanced CNN Recordings/Datasets/MNIST_Object_Detection.zip"

# %%
def create_training_data():
    training_data = []
    folder_dir = "/content/mnist_detection/train/images"
    dirpath = "/content/mnist_detection/train/labels"

    list_dir = [int(f.split(".")[0]) for f in os.listdir(folder_dir)]
    list_dir.sort()
    label_dir = [int(f.split(".")[0]) for f in os.listdir(dirpath)]
    label_dir.sort()

    for images, filename, i in zip(list_dir, label_dir, range(10000)):
        data = pd.read_csv(os.path.join(dirpath, str(filename) + ".txt"), sep=',', header=0)
        xmin = data['xmin'].values[0]
        xmax = data['xmax'].values[0]
        ymin = data['ymin'].values[0]
        ymax = data['ymax'].values[0]
        img_array = cv2.imread(os.path.join(folder_dir, str(images) + ".png"))
        new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
        training_data.append([new_array, data['label'].values[0], (xmin, ymin, xmax, ymax)])
    return training_data


def create_testing_data():
    testing_data = []
    folder_dir = "/content/mnist_detection/test/images"
    dirpath = "/content/mnist_detection/test/labels"

    list_dir = [int(f.split(".")[0]) for f in os.listdir(folder_dir)]
    list_dir.sort()
    label_dir = [int(f.split(".")[0]) for f in os.listdir(dirpath)]
    label_dir.sort()

    for images, filename, i in zip(list_dir, label_dir, range(10000)):
        data = pd.read_csv(os.path.join(dirpath, str(filename) + ".txt"), sep=',', header=0)
        xmin = data['xmin'].values[0]
        xmax = data['xmax'].values[0]
        ymin = data['ymin'].values[0]
        ymax = data['ymax'].values[0]
        img_array = cv2.imread(os.path.join(folder_dir, str(images) + ".png"))
        new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
        testing_data.append([new_array, data['label'].values[0], (xmin, ymin, xmax, ymax)])
    return testing_data


# %%
training_data = create_training_data()
testing_data = create_testing_data()

# %%
X_train, y_train, y_train_boxes = [], [], []
X_test, y_test, y_test_boxes = [], [], []

for features, labels, boxes in training_data:
    X_train.append(features)
    y_train.append(labels)
    y_train_boxes.append(boxes)

for features, labels, boxes in testing_data:
    X_test.append(features)
    y_test.append(labels)
    y_test_boxes.append(boxes)

X_train = np.array(X_train).reshape(-1, 300, 300, 3)
X_test = np.array(X_test).reshape(-1, 300, 300, 3)

print("X_train:", X_train.shape, "X_test:", X_test.shape)
