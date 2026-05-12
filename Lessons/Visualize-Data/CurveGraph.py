import matplotlib.pyplot as plt
import numpy as np

# Data
x = np.linspace(-1, 30, 200) # 200 points between -1 and 30
y = np.sin(x) # Sinusoidal curve foreach x value

# Create the curve graph
fig, ax = plt.subplots(figsize=(10, 6)) # fig = window, ax = graph area, figsize in inches (width, height)

# Draw the curve
ax.plot(x,
        y,
        label='Sin(x)', 
        color='blue',
        marker='o',
        markersize=5,
        markeredgecolor='red',
        markerfacecolor='red',
        linewidth=7) # marker='o' adds points on the curve, markersize sets their size

# Customize the graph
ax.set_title('Curve Graph of Sin(x)', fontsize=16)
ax.set_xlabel('x', fontsize=14)
ax.set_ylabel('Sin(x)', fontsize=14)
ax.grid(True)
ax.legend()

plt.show()