import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots()

# 100 random x values between 0 and 10
x = np.random.rand(100) * 10 

# 100 random y values between 0 and 10
y = np.random.rand(100) * 10 

# 100 random colors
couleurs = np.random.rand(100) 

scatter = ax.scatter(x,
                     y,
                     c=couleurs, #c=couleurs sets the color of each point
                     cmap='viridis', # cmap='viridis' is a color map
                     s=100, # s=100 sets the size of the points
                     alpha=0.7) #  alpha=0.7 sets transparency


ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Scatter Plot with Random Colors')
cbar = plt.colorbar(scatter, ax=ax) # Add a color bar to show the color scale
cbar.set_label('Color Scale')


plt.show()