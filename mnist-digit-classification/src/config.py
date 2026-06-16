# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import random
import warnings

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Activation, LeakyReLU
from tensorflow.keras.optimizers import Adam, SGD

warnings.filterwarnings("ignore")

# %%
np.random.seed(1)
random.seed(1)
tf.random.set_seed(1)
