import numpy as np
import os
import matplotlib.pyplot as plt

os.environ["KERAS_BACKEND"] = "jax"
import keras

np.random.seed(1337)

max_features = 10000
maxlen = 500
batch_size = 32
(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=max_features)
x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)

print(x_train.shape)

inputs = keras.layers.Input(shape=(500,))
embedding = keras.layers.Embedding(max_features, 32)
rnn = keras.layers.SimpleRNN(32)
outputs = keras.layers.Dense(1,activation='sigmoid')(rnn(embedding(inputs)))
model = keras.Model(inputs, outputs)

keras.utils.plot_model(model, to_file='model_imdb_complete.png', show_shapes=True, show_layer_names=True,
                       expand_nested=True, show_layer_activations=True)

model.summary()

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

history = model.fit(x=x_train, y=y_train, epochs=10, batch_size=128,validation_split=0.2)

acc = history.history["acc"]
val_acc = history.history["val_acc"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]
epochs = range(len(acc))

plt.style.use('dark_background')
plt.plot(epochs, acc, label="Training")
plt.plot(epochs, val_acc, label="Validierung")
plt.title("Korrektklassifizierungsrate Training/Validierung")
plt.legend()
plt.tight_layout()
plt.savefig("train_validation_acc_imdb.svg")
plt.figure()
plt.plot(epochs, loss, label="Verlust Training")
plt.plot(epochs, val_loss, label="Verlust Validierung")
plt.title("Wert der Verlustfunktion Training/Validierung")
plt.legend()
plt.tight_layout()
plt.savefig("train_validation_loss_imdb.svg")