# %% [markdown]
# # **MNIST Handwritten Digit Classification using a CNN**

# %% [markdown]
# ## **Introduction**

# %% [markdown]
# ## **Introduction to the MNIST Dataset**

# %% [markdown]
# The **MNIST** dataset is an acronym that stands for the **Modified National Institute of Standards and Technology** dataset.
# 
# *   **This dataset consists of 60,000 grayscale images**, which are small 28x28 pixel images. 
# <br> **These are images of handwritten digits from 0 to 9.**
# *   **The task is to correctly classify the image of a handwritten digit into the right number**, that is - one of the 10 numbers from 0 to 9.
# 
# 
# 
# 

# %% [markdown]
# ## **Importing the Libraries**

# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

import tensorflow as tf

# Keras Sequential Model
from tensorflow.keras.models import Sequential

# Importing all the different layers and optimizers
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Activation, LeakyReLU
from tensorflow.keras.optimizers import Adam,SGD

# The below code can be used to ignore the warnings that may occur due to deprecations
import warnings
warnings.filterwarnings("ignore")

# %%
import random
np.random.seed(1) #for numpy
random.seed(1) 
tf.random.set_seed(1) #for tensorflow




# %% [markdown]
# ## **Loading the Dataset**

# %% [markdown]
# *   The MNIST dataset is already present in TensorFlow and Keras, in the form of an **N-dimensional Numpy array**, so we can directly import the dataset from the package and use it.
# *   The dataset can be imported as shown below: <br>
# `from tensorflow.keras.datasets import mnist`<br>
# `mnist.load_data()`
# *   **mnist.load_data()** returns both the train and test data. The train data consists of 60,000 images in the form of Numpy arrays, while the test data consists of 10,000 images as Numpy arrays.
# 
# 
# 
# 

# %%
# Loading the data
from tensorflow.keras.datasets import fashion_mnist

#tf.keras.datasets.fashion_mnist.load_data()
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# %%
X_train.shape

# %% [markdown]
# Here the data is stored in a **3-dimensional Numpy array**. 
# - The first dimension **60000** denotes **the number of images in the training data**, and each image is stacked on top of the other, making it a 3-dimensional Numpy array.
# - The second and third dimensions **28, 28** denote the number of pixels along the height and width of the 2-dimensional image.

# %% [markdown]
# Below is a 2-dimensional Numpy representation (the pixelmap) of the first image in the training data. Each image is represented by 28x28 square pixel values.

# %%
X_train[0]

# %%
y_train[0]

# %% [markdown]
# The **target labels** are numerical digits between 0 to 9. 

# %%
X_test.shape

# %% [markdown]
# The test dataset, on the other hand, has **10,000 images**. Each image, as before, is a 28x28 square image.

# %% [markdown]
# ## **Data Visualization**

# %% [markdown]
# ### **Converting the Numpy arrays to images and visualizing a few random images**

# %%
num_classes=10                                                                  # Number of Classes
categories=np.unique(y_train)                                                   # Obtaing the unique classes from y_train
rows = 3                                                                        # Defining number of rows=3
cols = 5                                                                        # Defining number of columns=4
fig = plt.figure(figsize=(10, 8))                                               # Defining the figure size to 10x8
for i in range(cols):
    for j in range(rows):
        random_index = np.random.randint(0, len(y_train))                       # Generating random indices from the data and plotting the images
        ax = fig.add_subplot(rows, cols, i * rows + j + 1)                      # Adding subplots with 3 rows and 4 columns
        ax.imshow(X_train[random_index, :], cmap=plt.get_cmap('gray'))          # Plotting the image using cmap=gray
        ax.set_title(categories[y_train[random_index]])
plt.show()

# %%
# Plot distribution of each category 
count_plot = sns.countplot(y_train)

# %% [markdown]
# We observe that **the dataset appears to be quite balanced**, with each category having approximately the same number of images. 
# 
# So **accuracy should be a good evaluation metric** for the model performance in this case study.

# %% [markdown]
# ## **Data Preparation**

# %% [markdown]
# In the data preparation stage, **we generally reshape the dataset to have a single channel** and **we also normalize the feature inputs.**
# 
# Normalization is highly recommended as it has the following benefits when training a neural network model:
# 
# 1. **Normalization makes the training faster and reduces the chances of getting stuck at a local optima.**
# 3. **Weight decay and estimation can be done more conveniently** with normalized inputs.
# 4. In deep neural networks, **normalization helps to avoid the Vanishing/Exploding gradient problem** The Vanishing/Exploding gradient problem occurs when very small or very large error gradients accumulate, and that results in either extremely small or very large updates to neural network model weights during the training process. This makes a model unstable and unable to learn from the training data.

# %% [markdown]
# As we already know, the images are in grayscale, so let us **reshape the arrays to just have a single channel**.

# %%
# Reshape dataset to have a single channel
X_train = X_train.reshape((X_train.shape[0], 28, 28, 1))
X_test = X_test.reshape((X_test.shape[0], 28, 28, 1))

# %% [markdown]
# Since the **image pixel values range from 0-255**, our method of normalization here will be **scaling** - we shall **divide all the pixel values by 255 to standardize the images to have values between 0-1.**

# %%
# Normalizing the image pixels
X_train_normalized = X_train.astype('float32')/255.0
X_test_normalized = X_test.astype('float32')/255.0

# %% [markdown]
# Since this is a **10-class classification problem**, **the output layer should have 10 neurons** which will provide us with the probabilities of the input image belonging to each of those 10 classes. Therefore, we also need to create a **one-hot encoded representation for the target classes.**

# %%
# Creating one-hot encoded representation of target labels
# We can do this by using this utility function - https://www.tensorflow.org/api_docs/python/tf/keras/utils/to_categorical
# to_categorical() function is also explained in the Neural Networks Module

y_train_encoded = tf.keras.utils.to_categorical(y_train)
y_test_encoded = tf.keras.utils.to_categorical(y_test)

# %%
y_train_encoded.shape

# %%
y_test_encoded.shape

# %% [markdown]
# ## **Model Building - Artificial Neural Network (ANN)**

# %%
# Fixing the seed for random number generators
import random
np.random.seed(1)
random.seed(1)
tf.random.set_seed(1)

# %%
cnn_model = Sequential()
cnn_model.add(Conv2D(64, (3,3), activation='relu', input_shape=(28, 28, 1), padding = 'same'))
cnn_model.add(MaxPooling2D(pool_size=(2,2),padding = 'same'))
#cnn_model.add(BatchNormalization())
cnn_model.add(Conv2D(32, (3,3), activation='relu', padding = 'same'))
cnn_model.add(MaxPooling2D(pool_size=(2,2),padding = 'same'))
cnn_model.add(BatchNormalization())
#cnn_model.add(Conv2D(32, (3,3), activation='relu', padding = 'same'))
#cnn_model.add(MaxPooling2D(2,2))
cnn_model.add(Conv2D(16, (3,3), activation='relu', padding = 'same'))
cnn_model.add(Flatten())
cnn_model.add(Dense(32, activation='relu'))
cnn_model.add(Dropout(0.25))
cnn_model.add(Dense(16, activation='relu'))
#cnn_model.add(Dropout(0.25))
#cnn_model.add(Dense(32, activation='relu'))
cnn_model.add(Dense(10, activation='softmax'))

# %%
cnn_model.compile(loss="binary_crossentropy", optimizer="adam", metrics = ['accuracy'])
cnn_model.summary()

# %%
model_history = cnn_model.fit(X_train_normalized,y_train_encoded, 
                              validation_split=0.2, epochs=5, verbose=2)
                        

# %%
from tensorflow.keras import backend
backend.clear_session()

# %%
# Summary of the whole model
from keras.applications.vgg16 import VGG16
# Summary of the whole model
model = VGG16(weights='imagenet',include_top=False,input_shape=(224,224,3))
model.summary()


# %%
vgg_model = VGG16(weights='imagenet', include_top = False, input_shape = (224,224,3))
vgg_model.summary()

# %%
# Making all the layers of the VGG model non-trainable. i.e. freezing them
for layer in vgg_model.layers:
    layer.trainable = False

# %%
for layer in vgg_model.layers:
    print(layer.name, layer.trainable)

# %%
new_model = Sequential()

# Adding the convolutional part of the VGG16 model from above
new_model.add(vgg_model)

# Flattening the output of the VGG16 model because it is from a convolutional layer
new_model.add(Flatten())

# Adding a dense output layer
new_model.add(Dense(32, activation='relu'))
new_model.add(Dense(16, activation='relu'))
new_model.add(Dense(10, activation='softmax'))

# %%
new_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
new_model.summary()

# %%

new_model2 = Sequential()

# Adding the convolutional part of the VGG16 model from above
new_model2.add(vgg_model)

# Flattening the output of the VGG16 model because it is from a convolutional layer
new_model2.add(Flatten())

# Adding a dense output layer
new_model2.add(Dense(32, activation='relu'))
new_model2.add(Dense(16, activation='relu'))
new_model2.add(Dense(16,activation='relu'))
new_model2.add(Dense(10, activation='softmax'))

# %%
new_model2.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
new_model2.summary()

# %% [markdown]
# Let's now build a **sequential ANN model.**

# %%
# Intializing a sequential model
ann_model = Sequential()

# Flatten the input to add dense convolutional layers on top of it
ann_model.add(Flatten(input_shape=(28, 28)))

# Adding a sequential layer with 100 neurons
ann_model.add(Dense(100, activation='relu'))

# Adding the output layer with 10 neurons and activation functions as softmax since this is a multi-class classification problem  
ann_model.add(Dense(10, activation='softmax'))

# Using SGD Optimizer
opt = SGD(learning_rate=0.01, momentum=0.9)

# Compile model
ann_model.compile(optimizer=opt,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Generating the summary of the model
ann_model.summary()

# %% [markdown]
# <b> Fitting the model on the train data:

# %%
history = ann_model.fit(
            X_train_normalized, y_train_encoded,
            epochs=15,
            validation_split=0.1,
            shuffle=True,
            batch_size=64,
            verbose=2
)

# %% [markdown]
# ### **Model Evaluation**

# %%
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# %% [markdown]
# **Observations:** 
# 
# *   We can see from the above plot that the model has perfomed well on the train and validation data, with a **validation accuracy of 97%.**
# 

# %% [markdown]
# ### **Evaluating the model on the test data**

# %%
accuracy = ann_model.evaluate(X_test_normalized, y_test_encoded, verbose=2)

# %% [markdown]
# ### **Generating the Predictions using the test data**

# %%
# Here we would get the output as probablities for each category
y_pred=ann_model.predict(X_test_normalized)

# %% [markdown]
# ### **Plotting the Confusion Matrix**

# %%
# Obtaining the categorical values from y_test_encoded and y_pred
y_pred_arg=np.argmax(y_pred,axis=1)
y_test_arg=np.argmax(y_test_encoded,axis=1)

# Plotting the Confusion Matrix using confusion matrix() function which is also predefined tensorflow module
confusion_matrix = tf.math.confusion_matrix(y_test_arg,y_pred_arg)
f, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(
    confusion_matrix,
    annot=True,
    linewidths=.4,
    fmt="d",
    square=True,
    ax=ax
)
plt.show()

# %% [markdown]
# **We observe that most of the classes are predicted correctly**, and the misclassification occurs mostly with the digit 4 which is sometimes confused with digit 9, and digit 5 which is sometimes confused with digit 3.

# %% [markdown]
# ## **Model Building - Convolutional Neural Network (CNN)**

# %% [markdown]
# Let's create a CNN model sequentially, where we will be adding the layers one after another.

# %%
# Clearing backend
from tensorflow.keras import backend
backend.clear_session()

# %%
# Fixing the seed for random number generators
import random
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

# %%
# Intializing a sequential model
model = Sequential()
#The input shape to the conv2D will be (28,28,1), number of filters = 64, kernel size = (3,3), ,  activation function 'relu', and padding='same'
# Adding first conv layer with 64 filters and kernel size 3x3 , padding 'same' provides the output size same as the input size
# Input_shape denotes input image dimension of MNIST images
model.add(Conv2D(64, (3, 3), activation='relu', padding="same", input_shape=(28, 28, 1)))

# Adding max pooling to reduce the size of output of first conv layer
model.add(MaxPooling2D((2, 2), padding = 'same'))

#model.add(Conv2D(32, (3, 3), activation='relu', padding="same"))
#model.add(MaxPooling2D((2, 2), padding = 'same'))
#model.add(Conv2D(32, (3, 3), activation='relu', padding="same"))
#model.add(MaxPooling2D((2, 2), padding = 'same'))

# flattening the output of the conv layer after max pooling to make it ready for creating dense connections
model.add(Flatten())

# Adding a fully connected dense layer with 100 neurons    
model.add(Dense(100, activation='relu'))

# Adding the output layer with 10 neurons and activation functions as softmax since this is a multi-class classification problem  
model.add(Dense(10, activation='softmax'))

# Using SGD Optimizer
opt = tf.keras.optimizers.Adam(learning_rate=0.01)

# Compile model
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

# Generating the summary of the model
model.summary()

# %% [markdown]
# ### <b> Fitting the model on the train data

# %%
history_1 = model.fit(
            X_train_normalized, y_train_encoded,
            epochs=5,
            validation_split=0.1,
            shuffle=True,
            batch_size=64,
            verbose=2
)

# %% [markdown]
# ### **Model Evaluation**

# %%
plt.plot(history_1.history['accuracy'])
plt.plot(history_1.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# %% [markdown]
# **Observations:** 
# 
# 
# *   We can see from the above plot that **the model has perfomed well on train and validation data with a validation accuracy of 98%.**
# 
# 
# 
# 
# 

# %% [markdown]
# ### **Evaluating the model on test data**

# %%
accuracy = model.evaluate(X_test_normalized, y_test_encoded, verbose=2)

# %% [markdown]
# ### **Generating the predictions using test data**

# %%
# Here we would get the output as probablities for each category
y_pred=model.predict(X_test_normalized)

# %%
y_pred

# %% [markdown]
# ### **Plotting the Confusion Matrix**

# %%
# Obtaining the categorical values from y_test_encoded and y_pred
y_pred_arg=np.argmax(y_pred,axis=1)
y_test_arg=np.argmax(y_test_encoded,axis=1)

# Plotting the Confusion Matrix using confusion matrix() function which is also predefined tensorflow module
confusion_matrix = tf.math.confusion_matrix(y_test_arg,y_pred_arg)
f, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(
    confusion_matrix,
    annot=True,
    linewidths=.4,
    fmt="d",
    square=True,
    ax=ax
)
plt.show()

# %% [markdown]
# We observe that most of the classes are predicted correctly. 
# 
# The misclassification mostly with the the digits 4 and 8, which are confused with digit 9.

