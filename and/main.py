import numpy as np
import os
import matplotlib.pyplot as plt

os.environ["KERAS_BACKEND"] = "jax"
import keras

np.random.seed(1337)

x_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_train = np.array([0,0,0,1])

inputs = keras.layers.Input(shape=(2,))
outputs = keras.layers.Dense(1, activation='sigmoid')(inputs)
model = keras.Model(inputs, outputs)

#model = keras.models.Sequential()
#model.add(keras.layers.Dense(1, activation='sigmoid'))

model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(x=x_train, y=y_train, epochs=10000, verbose=0)

keras.utils.plot_model(model, to_file='model_and_complete.png', show_shapes=True, show_layer_names=True,
                       expand_nested=True, show_layer_activations=True)
model.summary()

print(model.predict(x_train))

x = np.linspace(0, 1, 100)
(X1_raster, X2_raster) = np.meshgrid(x, x)
X1_vektor = X1_raster.flatten()
X2_vektor = X2_raster.flatten()

eingangswerte_grafik = np.vstack((X1_vektor, X2_vektor)).T
ausgangswerte_grafik = model.predict(eingangswerte_grafik).reshape(X1_raster.shape)

(gewichte, bias) = model.layers[1].get_weights()

plt.style.use('dark_background')
plt.contourf(X1_raster, X2_raster, ausgangswerte_grafik, 100, cmap="jet")
plt.xlim(0, 1)
plt.ylim(0, 1)

plt.xlabel("Eingabewert $x_1$")
plt.ylabel("Eingabewert $x_2$")
plt.colorbar()

plt.tight_layout()
plt.savefig("predictions_and_10000.svg")
