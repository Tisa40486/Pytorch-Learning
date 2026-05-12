import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots()

données = [np.random.normal(0, std, 100) for std in range(1, 4)]
ax.violinplot(données, positions=[1, 2, 3], showmeans=True)

ax.set_xticks([1, 2, 3])
ax.set_xticklabels(['A', 'B', 'C'])
ax.set_ylabel('Valeurs')
ax.set_title('Violin plot')
ax.grid(axis='y', alpha=0.3)

plt.show()