import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots()

valeurs = [30, 25, 20, 25]
labels = ['A', 'B', 'C', 'D']
couleurs = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']

ax.pie(valeurs, labels=labels, colors=couleurs, autopct='%1.1f%%')
ax.set_title('Répartition')

plt.show()