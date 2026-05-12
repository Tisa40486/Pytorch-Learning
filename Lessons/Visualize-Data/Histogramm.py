import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots()

données = np.random.normal(loc=100, scale=15, size=1000)

ax.hist(données, bins=30, color='green', alpha=0.7, edgecolor='black')

ax.set_xlabel('Valeurs')
ax.set_ylabel('Fréquence')
ax.set_title('Distribution normale')
ax.grid(axis='y', alpha=0.3)

plt.show()