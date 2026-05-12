import numpy as np
import matplotlib.pyplot as plt

data = np.random.rand(10, 10)

fig, ax = plt.subplots()
im = ax.imshow(data, cmap='coolwarm')

ax.set_xticks(np.arange(10))
ax.set_yticks(np.arange(10))
ax.set_xticklabels(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'])
ax.set_yticklabels(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'])

for i in range(10):
    for j in range(10):
        text = ax.text(j, i, f'{data[i, j]:.2f}',
                      ha="center", va="center", color="black")

plt.colorbar(im, ax=ax)
plt.show()
