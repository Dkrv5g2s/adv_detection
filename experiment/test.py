import numpy as np
import matplotlib.pyplot as plt

# Define the range and interval for x and y (customizable)
x_start = 1  # Start of x range
x_end = 2    # End of x range
x_interval = 0.1  # Interval for x

y_start = 3.5  # Start of y range
y_end = 7      # End of y range
y_interval = 0.1  # Interval for y

# Generate x and y values based on the custom range and interval
x = np.arange(x_start, x_end + x_interval, x_interval)
y = np.arange(y_start, y_end + y_interval, y_interval)

# Create a meshgrid for x and y
X, Y = np.meshgrid(x, y)

# Calculate the difference y-x
Z = Y - X

# Plot the heatmap
plt.figure(figsize=(10, 8))
plt.imshow(Z, cmap='coolwarm', origin='lower', extent=[x_start, x_end, y_start, y_end])
plt.colorbar(label='y-x Difference')

# Add axis labels and title
plt.xticks(np.arange(x_start, x_end + 1, 1))
plt.yticks(np.arange(y_start, y_end + 1, 1))
plt.xlabel('x')
plt.ylabel('y')
plt.title('Heatmap of y-x Difference (Custom Range)')

# Ensure the aspect ratio is equal
plt.gca().set_aspect('equal', adjustable='box')

# Display the heatmap
plt.tight_layout()
plt.show()