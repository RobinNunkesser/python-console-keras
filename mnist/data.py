import numpy as np
import os
import matplotlib.pyplot as plt

os.environ["KERAS_BACKEND"] = "jax"
import keras

np.random.seed(1337)

# Load the data and split it between train and test sets
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

plt.style.use('dark_background')
digit = x_train[4]
plt.imshow(digit,cmap='gray')
plt.show()