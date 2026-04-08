import matplotlib.pyplot as plt
import numpy as np

# Generate skewed example data (normal distribution with a few outliers)
data = np.concatenate([np.random.normal(50, 10, 300), np.random.normal(120, 15, 10)])

# Calculate mean and standard deviation
mean = np.mean(data)
std = np.std(data)

# Create the plot
plt.figure(figsize=(12, 6))
plt.hist(data, bins=40, color='gray', edgecolor='black', alpha=0.7)

# Overlay Z-score lines
plt.axvline(mean, color='blue', linestyle='--', label=f'Mean: {mean:.1f}')
for z in [-3, -2, 2, 3]:
    color = 'green' if abs(z) == 2 else 'red'
    label = f'{z} SD' if abs(z) == 3 else None # Only label SD=3 for cleaner look
    plt.axvline(mean + z * std, color=color, linestyle='--', label=label)

plt.title('Histogram with Z-Score Boundaries', fontsize=14)
plt.xlabel('Value', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.show()