import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

path_data = r"C:\Users\corin\PycharmProjects\CNS-SS2425\cmake-build-debug\random_numbers.txt"
random_numbers = np.loadtxt(path_data, delimiter='\t')

mean_num = np.mean(random_numbers, axis=0)

print(mean_num)

# Perform PCA
pca = PCA(n_components=2)
pca.fit(random_numbers)

# Get the direction of the largest and smallest variance
largest_variance_direction = pca.components_[0]  # First principal component
smallest_variance_direction = pca.components_[1]  # Second principal component

print("Direction of largest variance:", largest_variance_direction)
print("Direction of smallest variance:", smallest_variance_direction)

# Plot the dataset
plt.scatter(random_numbers[:, 0], random_numbers[:, 1], label="Data Points", color="blue", alpha=0.5)

# Plot the principal component as an arrow starting from the mean
plt.quiver(mean_num[0], mean_num[1], largest_variance_direction[0], largest_variance_direction[1],
           angles='xy', scale_units='xy', scale=1, color='red', label="Largest Variance Direction")
plt.quiver(mean_num[0], mean_num[1], smallest_variance_direction[0], smallest_variance_direction[1],
           angles='xy', scale_units='xy', scale=1, color='green', label="Smallest Variance Direction")

# Set plot properties
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Dataset with Principal Component (Direction of Largest Variance)")
plt.legend()
plt.axis('equal')  # Keep the scale equal to see the direction clearly
plt.grid(True)
plt.show()