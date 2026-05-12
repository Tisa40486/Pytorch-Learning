import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots()

données = [np.random.normal(0, std, 100) for std in range(1, 4)]

ax.boxplot(données, labels=['A', 'B', 'C'])
ax.set_ylabel('Valeurs')
ax.set_title('Diagramme en boîte')
ax.grid(axis='y', alpha=0.3)

plt.show()