import numpy as np
import os
import matplotlib.pyplot as plt

os.environ["KERAS_BACKEND"] = "jax"
import keras

np.random.seed(1337)

x_train = np.array([0,1])
y_train = np.array([1,0])
print(x_train.ndim)
print(x_train.shape)
print(y_train.ndim)
print(y_train.shape)

inputs = keras.Input(shape=(1,))
outputs = keras.layers.Dense(1)(inputs)
model = keras.Model(inputs, outputs)

model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(x=x_train, y=y_train, epochs=10000, verbose=0)

keras.utils.plot_model(model, to_file='model_not_complete.png', show_shapes=True, show_layer_names=True,
                       expand_nested=True, show_layer_activations=True)
model.summary()

print(model.predict(x_train))

(weights,biases) = model.layers[1].get_weights()
print(model.layers[1].get_weights())

plt.style.use('dark_background')
x = np.linspace(0, 1, 100)
y = model.predict(x)
plt.plot(x, weights[0]*x + biases[0])
#plt.plot(x, y)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel("Eingabewert $x$")
plt.ylabel("Ausgabewert $y$")
plt.tight_layout()
plt.savefig("predictions_linear_not_10000.svg")

plt.figure()

acc = history.history["accuracy"]
loss = history.history["loss"]
epochs = range(len(acc))

plt.plot(epochs, acc, label="Korrektklassifizierungsrate")
plt.plot(epochs, loss, label="Verlust")
plt.title("Korrektklassifizierungsrate und Verlust")
plt.legend()
plt.tight_layout()
plt.savefig("train_acc_loss_not.svg")

