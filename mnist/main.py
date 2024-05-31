import numpy as np
import os
import matplotlib.pyplot as plt

os.environ["KERAS_BACKEND"] = "jax"
import keras

np.random.seed(1337)

# Load the data and split it between train and test sets
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
print("x_train shape:", x_train.shape)
# Make sure images have shape (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")

# Model parameters
num_classes = 10

inputs = keras.layers.Input(shape=(28, 28, 1))
conv1 = keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu')
pool1 = keras.layers.MaxPooling2D(pool_size=(2, 2))
conv2 = keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu')
pool2 = keras.layers.GlobalAveragePooling2D()
dropout = keras.layers.Dropout(0.5)
outputs = keras.layers.Dense(num_classes, activation='softmax')(dropout(pool2(conv2(pool1(conv1(inputs))))))
model = keras.Model(inputs, outputs)

# model = keras.Sequential(
#     [
#         keras.layers.Input(shape=(28, 28, 1)),
#         keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
#         keras.layers.MaxPooling2D(pool_size=(2, 2)),
#         keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
#         keras.layers.MaxPooling2D(pool_size=(2, 2)),
#         keras.layers.Flatten(),
#         keras.layers.Dropout(0.5),
#         keras.layers.Dense(num_classes, activation="softmax"),
#     ]
# )

keras.utils.plot_model(model, to_file='model_mnist_complete.png', show_shapes=True, show_layer_names=True,
                       expand_nested=True, show_layer_activations=True)

model.summary()

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(),
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    metrics=[
        keras.metrics.SparseCategoricalAccuracy(name="acc"),
    ],
)

batch_size = 128
epochs = 15

callbacks = [
    keras.callbacks.ModelCheckpoint(filepath="model_at_epoch_{epoch}.keras"),
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=2),
]

history = model.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.15,
    callbacks=callbacks,
)

score = model.evaluate(x_test, y_test, verbose=0)

model.save("final_model.keras")

model = keras.saving.load_model("final_model.keras")

predictions = model.predict(x_test)

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
plt.savefig("train_validation_acc_mnist.svg")
plt.figure()
plt.plot(epochs, loss, label="Verlust Training")
plt.plot(epochs, val_loss, label="Verlust Validierung")
plt.title("Wert der Verlustfunktion Training/Validierung")
plt.legend()
plt.tight_layout()
plt.savefig("train_validation_loss_mnist.svg")
