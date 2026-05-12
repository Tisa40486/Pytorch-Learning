
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z = np.sqrt(X**2 + Y**2)

fig, ax = plt.subplots()
contour = ax.contourf(X, Y, Z, levels=20, cmap='viridis')
ax.contour(X, Y, Z, levels=10, colors='black', linewidths=0.5)

plt.colorbar(contour, ax=ax)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Contour plot')

plt.show()