import matplotlib.pyplot as plt
import numpy as np


fig, ax = plt.subplots()

# Create Categories and values
categories = ['Dog', 'Cat', 'Bird', 'Fish', 'Horse']
values = [10, 15, 7, 5, 12]

# Create the bar plot
ax.bar(categories, values, color=['blue', 'red', 'green', 'pink', 'purple'])

ax.set_ylabel('Valeurs')
ax.set_title('Graphique en barres')
ax.grid(axis='y', alpha=0.3)

plt.show()