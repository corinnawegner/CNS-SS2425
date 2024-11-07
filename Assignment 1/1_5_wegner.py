import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

path_data = r"C:\Users\corin\PycharmProjects\CNS-SS2425\cmake-build-debug\random_numbers.txt"
random_numbers = np.loadtxt(path_data, delimiter='\t')

mean_num = np.mean(random_numbers, axis=0)

print(mean_num)

# Perform PCA
pca = PCA(n_components=1)  # We only need the first principal component
pca.fit(random_numbers)

# Get the direction of the largest variance (first principal component)
principal_component = pca.components_[0]

print("Direction of largest variance:", principal_component)

# Plot the dataset
plt.scatter(random_numbers[:, 0], random_numbers[:, 1], label="Data Points", color="blue", alpha=0.5)

# Plot the principal component as an arrow starting from the mean
plt.quiver(mean_num[0], mean_num[1], principal_component[0], principal_component[1],
           angles='xy', scale_units='xy', scale=1, color='red', label="Principal Component")

# Set plot properties
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Dataset with Principal Component (Direction of Largest Variance)")
plt.legend()
plt.axis('equal')  # Keep the scale equal to see the direction clearly
plt.grid(True)
plt.show()