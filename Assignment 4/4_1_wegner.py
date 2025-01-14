import numpy as np
from PIL import Image
import random


# Step 1: Load an RGB image into a suitable data structure
def load_image(filename):
    """
    Load an image and convert it to an RGB array.
    :param filename: Path to the image file
    :return: numpy array of shape (height, width, 3)
    """
    img = Image.open(filename)
    img = img.convert("RGB")  # Ensure the image is in RGB mode
    return np.array(img)


# Step 2: Initialize a grid of neurons
def initialize_grid(m, n):
    """
    Initialize an m x n grid of neurons, each with random RGB weights.
    :param m: Number of rows
    :param n: Number of columns
    :return: numpy array of shape (m, n, 3) with random values in [0, 255]
    """
    return np.random.randint(0, 256, (m, n, 3), dtype=np.float32)


# Step 3: Shuffle the input RGB array
def generate_input_vectors(image_array):
    """
    Generate shuffled input vectors from the image.
    :param image_array: numpy array of shape (height, width, 3)
    :return: list of RGB tuples shuffled randomly
    """
    h, w, _ = image_array.shape
    input_vectors = image_array.reshape(h * w, 3)  # Flatten to a list of RGB values
    random.shuffle(input_vectors.tolist())        # Shuffle the list
    return input_vectors

def find_winning_neuron(som_grid, input_vector):
    """
    Find the position (x*, y*) of the neuron with the smallest Euclidean distance
    to the input vector.
    :param som_grid: SOM grid of shape (m, n, 3)
    :param input_vector: Input RGB vector of shape (3,)
    :return: Tuple (x, y) representing the position of the winning neuron
    """
    distances = np.linalg.norm(som_grid - input_vector, axis=2)  # Compute distances
    min_distance = np.min(distances)  # Find the minimum distance
    winners = np.argwhere(distances == min_distance)  # Get all positions with the min distance
    return tuple(winners[random.randint(0, len(winners) - 1)])  # Randomly pick one if multiple

def learning_rate(t):
    tau = 400
    return 0.01 + 0.4*np.exp(-t/tau)

def update_weights(som_grid, winner_pos, input_vector, learning_rate):
    """
    Update the weight vector of the winning neuron.
    :param som_grid: SOM grid of shape (m, n, 3)
    :param winner_pos: Tuple (x, y) of the winning neuron's position
    :param input_vector: Input RGB vector of shape (3,)
    :param learning_rate: Current learning rate α(t)
    """
    x, y = winner_pos
    #print(type(input_vector))
    #print(type(som_grid[x,y]))
    som_grid[x, y] += learning_rate * (input_vector - som_grid[x, y])  # Update weights


# Example usage
def main():

    input_image = load_image("testfile.png")
    print(input_image.shape)

    # Initialize SOM grid and parameters
    grid_rows, grid_cols = input_image.shape[0], input_image.shape[1]
    som_grid = np.random.randint(0, 256, (grid_rows, grid_cols, 3)).astype(np.float32)

    for step in range(400):
        for input_vector in input_image:

            # Step 1: Find the winning neuron
            winner_pos = find_winning_neuron(som_grid, input_vector)
            print(f"Winning neuron position: {winner_pos}")

            # Step 2: Update the weights of the winning neuron
            update_weights(som_grid, winner_pos, input_vector, learning_rate(step))
            print(f"Updated weights at {winner_pos}: {som_grid[winner_pos]}")

if __name__ == "__main__":
    main()

def learning_rate(t):
    tau = 400
    return 0.01 + 0.4 * np.exp(-t/tau)

if __name__ == "__main__":
    main()
