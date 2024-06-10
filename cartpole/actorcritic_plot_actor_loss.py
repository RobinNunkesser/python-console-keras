import os

os.environ["KERAS_BACKEND"] = "tensorflow"
import keras
import numpy as np
import matplotlib.pyplot as plt
from keras import ops

x = np.linspace(-2.00, 2.00, 400)
(X1_raster, X2_raster) = np.meshgrid(x, x)
X1_vektor = X1_raster.flatten()
X2_vektor = X2_raster.flatten()
huber_loss = keras.losses.Huber()

def transformi(i):
    return -ops.log(i[1])*-i[0]

# Nutzt die gerasterten Eingabewerte und erzeugt Ausgabewerte
eingangswerte_grafik = np.vstack((X1_vektor, X2_vektor)).T
#applyall = np.vectorize(transformi)
#test = ops.expand_dims(eingangswerte_grafik, 1)
ausgangswerte_grafik = np.apply_along_axis(transformi, 1, eingangswerte_grafik).reshape(X1_raster.shape)


# Contourplot der gerasterten Ausgangswerte in leicht vergroessertem
# Bereich und Legende
plt.style.use('dark_background')
plt.contourf(X1_raster, X2_raster, ausgangswerte_grafik, 100, cmap="jet")
plt.xlim(-2, 2)
plt.ylim(0, 1)

plt.xlabel("Belohnungsüberschätzung Critic")
plt.ylabel("Vorhersagesicherheit Actor")
plt.colorbar()

plt.tight_layout()
plt.savefig("actor_loss.svg")
