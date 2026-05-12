from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

#Create figure and 3D axis
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Data

# 1000 points between 0 and 10*pi
theta = np.linspace(-5, 10*np.pi, 1000)

# 1000 points between 0 and 10
z = np.linspace(0, 10, 1000)

# Create a spiral in 3D space
x = 5 * np.cos(theta)
y = 5 * np.sin(theta)

ax.plot(x, y, z, color='green', linewidth=6, marker='*', markersize=5, alpha=1, markeredgecolor='red', markerfacecolor='red')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Courbe 3D')

plt.show()