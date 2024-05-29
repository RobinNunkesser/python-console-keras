import numpy as np
import os
import matplotlib.pyplot as plt

os.environ["KERAS_BACKEND"] = "jax"
import keras

np.random.seed(1337)

x_train = np.array([0,1])
print(x_train.ndim)
print(x_train.shape)
y_train = np.array([1, 0])

inputs = keras.Input(shape=(1,))
outputs = keras.layers.Dense(1, activation='sigmoid')(inputs)
model = keras.Model(inputs, outputs)

#model = keras.models.Sequential()
#model.add(keras.layers.Input(shape=(1,)))
#model.add(keras.layers.Dense(1, activation='sigmoid'))

model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(x=x_train, y=y_train, epochs=10000, verbose=0)

keras.utils.plot_model(model, to_file='model_not_complete.png', show_shapes=True, show_layer_names=True,
                       expand_nested=True, show_layer_activations=True)
model.summary()

print(model.predict(x_train))

plt.style.use('dark_background')
x = np.linspace(0, 1, 100)
y = model.predict(x)
plt.plot(x, y)
plt.xlabel("Eingabewert $x$")
plt.ylabel("Ausgabewert $y$")
plt.tight_layout()
plt.savefig("predictions_not_10000.svg")

