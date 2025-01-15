import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Codebook vector data
codebook_vectors = [
    (0, 0, 2063, 129.80, 169.58, 207.15),
    (0, 1, 32178, 199.55, 141.00, 168.79),
    (0, 2, 13794, 129.23, 200.96, 127.36),
    (0, 3, 4126, 126.51, 198.04, 236.18),
    (1, 0, 177834, 225.58, 203.59, 213.76),
    (1, 1, 33530, 225.66, 148.01, 44.07),
    (1, 2, 466529, 241.00, 241.00, 241.00),
    (1, 3, 5511, 56.99, 200.13, 245.68),
    (2, 0, 11720, 176.50, 85.19, 128.07),
    (2, 1, 0, 3.00, 8.00, 14.00),
    (2, 2, 4093, 43.40, 196.77, 243.89),
    (2, 3, 81222, 148.21, 23.63, 82.00),
]

# Extract RGB and counts for visualization
reds = [vec[3] for vec in codebook_vectors]
greens = [vec[4] for vec in codebook_vectors]
blues = [vec[5] for vec in codebook_vectors]
counts = [vec[2] for vec in codebook_vectors]

# Normalize counts for marker size
marker_sizes = [count / max(counts) * 300 for count in counts]  # Scale for visualization

# Create 3D scatter plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(reds, greens, blues, c=[(r/255, g/255, b/255) for r, g, b in zip(reds, greens, blues)], s=marker_sizes, edgecolors="k")

# Label axes
ax.set_xlabel("Red")
ax.set_ylabel("Green")
ax.set_zlabel("Blue")

# Title and grid
ax.set_title("Codebook Vectors in RGB Space")
ax.grid(True)

plt.show()
