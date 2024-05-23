import numpy as np
import os

os.environ["KERAS_BACKEND"] = "jax"
import keras

np.random.seed(1337)

x_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_train = np.array([[0], [1], [1], [0]])

model = keras.models.Sequential()
model.add(keras.layers.Dense(2, activation='sigmoid'))
model.add(keras.layers.Dense(1, activation='linear'))

model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(x=x_train, y=y_train, epochs=10000, verbose=0)

keras.utils.plot_model(model, to_file='model_xor_complete.png', show_shapes=True, show_layer_names=True,
                       expand_nested=True, show_layer_activations=True)
model.summary()

print(model.predict(x_train))
