import numpy as np
from selforganizedmaps import *

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import random

# Load and preprocess images
def preprocess_image(filename):
    """
    Load an image, convert to RGB, and set Blue channel to zero.
    :param filename: Path to the image
    :return: Preprocessed numpy array
    """
    img = Image.open(filename).convert("RGB")
    img_array = np.array(img)
    img_array[:, :, 2] = 0  # Set Blue channel to zero
    return img_array

# Train SOM
def train_som(image_array, m, n, iterations):
    """
    Train a Self-Organizing Map (SOM) on the image data.
    :param image_array: Preprocessed image array
    :param m: Number of rows in SOM grid
    :param n: Number of columns in SOM grid
    :param iterations: Number of training iterations
    :return: Trained SOM grid
    """
    h, w, _ = image_array.shape
    input_vectors = image_array.reshape(h * w, 3).astype(np.float32)

    # Initialize SOM grid
    som_grid = np.random.uniform(0, 255, (m, n, 3)).astype(np.float32)
    som_grid[:, :, 2] = 0  # Ensure Blue channel is zero

    for t in range(iterations):
        learning_rate_t = learning_rate(t)
        input_vector = input_vectors[random.randint(0, len(input_vectors) - 1)]
        winner_pos = find_winning_neuron(som_grid, input_vector)
        update_weights(som_grid, winner_pos, input_vector, learning_rate_t)

    return som_grid

# Generate Voronoi tessellation
def generate_voronoi(image_array, som_grid, output_filename):
    """
    Generate a Voronoi tessellation for the given image.
    :param image_array: Input image array
    :param som_grid: Trained SOM grid
    :param output_filename: Path to save the Voronoi diagram
    """
    voronoi_image = np.zeros((255, 255, 3), dtype=np.uint8)

    for r in range(255):
        for g in range(255):
            input_vector = np.array([r, g, 0], dtype=np.float32)
            winner_pos = find_winning_neuron(som_grid, input_vector)
            winner_x, winner_y = winner_pos
            voronoi_image[r, g] = som_grid[winner_x, winner_y].astype(np.uint8)

    img = Image.fromarray(voronoi_image)
    img.save(output_filename)

# Visualize and overlay weight vectors
def plot_voronoi_with_weights(voronoi_filename, som_grid, output_filename):
    voronoi_image = np.array(Image.open(voronoi_filename))

    plt.figure(figsize=(10, 10))
    plt.imshow(voronoi_image)
    plt.title("Voronoi Tessellation in Red-Green Plane")
    plt.xlabel("Red Channel")
    plt.ylabel("Green Channel")

    # Overlay weight vectors
    m, n, _ = som_grid.shape
    for i in range(m):
        for j in range(n):
            red, green, _ = som_grid[i, j]
            plt.scatter(green, red, color="black", s=50, marker="o", edgecolors="white", linewidths=0.5)

    plt.savefig(output_filename)
    plt.show()

# Main process
if __name__ == "__main__":
    # Parameters
    som_size = (3, 4)  # Grid size
    iterations = 1000  # Number of iterations

    # Process gradient.png
    gradient_image = preprocess_image("gradient.png")
    som_gradient = train_som(gradient_image, som_size[0], som_size[1], iterations)
    generate_voronoi(gradient_image, som_gradient, "voronoi_gradient.png")
    plot_voronoi_with_weights("voronoi_gradient.png", som_gradient, "voronoi_gradient_with_weights.png")

    # Process gradient_nonlinear.png
    gradient_nonlinear_image = preprocess_image("gradient_nonlinear.png")
    som_gradient_nonlinear = train_som(gradient_nonlinear_image, som_size[0], som_size[1], iterations)
    generate_voronoi(gradient_nonlinear_image, som_gradient_nonlinear, "voronoi_gradient_nonlinear.png")
    plot_voronoi_with_weights("voronoi_gradient_nonlinear.png", som_gradient_nonlinear, "voronoi_gradient_nonlinear_with_weights.png")
