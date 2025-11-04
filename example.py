import numpy as np
import matplotlib.pyplot as plt

# Dimensiones del mapa
size = 50  # 50x50 celdas
mapa = np.zeros((size, size))  # mapa acumulativo de experiencia

# Función para crear una "onda" a partir de un spike
def onda_spike(cx, cy, intensidad=1.0, sigma=3):
    """Crea una onda gaussiana centrada en (cx, cy)."""
    x = np.arange(0, size)
    y = np.arange(0, size)
    X, Y = np.meshgrid(x, y)
    gauss = intensidad * np.exp(-((X - cx)**2 + (Y - cy)**2) / (2 * sigma**2))
    return gauss

# Lista de spikes: (x, y, tiempo, intensidad)
spikes = [
    (10, 10, 1, 1.0),
    (15, 20, 2, 0.8),
    (30, 35, 3, 1.2),
    (25, 15, 4, 1.0),
    (40, 10, 5, 0.9),
]

# Simular paso del tiempo
for t in range(1, 6):
    # Buscar spikes en este instante
    for spike in [s for s in spikes if s[2] == t]:
        cx, cy, _, intensidad = spike
        mapa += onda_spike(cx, cy, intensidad=intensidad)

# Mostrar el mapa acumulado
plt.imshow(mapa, cmap="hot", interpolation="nearest")
plt.colorbar(label="Intensidad acumulada")
plt.title("Mapa de navegación generado por ondas de spikes")
plt.show()
