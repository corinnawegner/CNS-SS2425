import numpy as np
from PIL import Image
import random
import math

# Step 1: Load an RGB image into a suitable data structure
def load_image(filename):
    """
    Load an image and convert it to an RGB array.
    :param filename: Path to the image file
    :return: numpy array of shape (height, width, 3)
    """
    img = Image.open(filename)
    img = img.convert("RGB")  # Ensure the image is in RGB mode
    print(f"shape: {np.array(img).shape}")
    return np.array(img).astype(float)


# Step 2: Initialize a grid of neurons
def initialize_grid(m, n, init_blue_to_zero = False):
    """
    Initialize an m x n grid of neurons, each with random RGB weights.
    :param init_blue_to_zero: Sets all values for blue to zero.
    :param m: Number of rows
    :param n: Number of columns
    :return: numpy array of shape (m, n, 3) with random values in [0, 255]
    """
    grid = np.random.randint(0, 256, (m, n, 3), dtype=np.uint8)  # Generate integers
    if init_blue_to_zero:
        grid[:,:,2] = 0
    return grid.astype(np.float32)  # Convert to float32


# Step 3: Shuffle the input RGB array
def generate_input_vectors(image_array):
    """
    Generate shuffled input vectors from the image.
    :param image_array: numpy array of shape (height, width, 3)
    :return: list of RGB tuples shuffled randomly
    """
    h, w, _ = image_array.shape
    orig_img = image_array.copy()
    input_vectors = image_array.reshape(-1, 3)  # Flatten to a list of RGB values
    np.random.shuffle(input_vectors)         # Shuffle the list
    return input_vectors, orig_img


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
    #print(winners)
    winning_neuron = tuple(winners[random.randint(0, len(winners) - 1)])
    #print('winning neuron:', winning_neuron)
    return winning_neuron   # Randomly pick one if multiple

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
    som_grid[x, y] += learning_rate * (input_vector - som_grid[x, y])  # Update weights
    return som_grid

def save_codebook(som_grid, timestep, output_filename):
    """
    Save the codebook vectors to a file in the specified format.
    :param som_grid: SOM grid of shape (m, n, 3)
    :param timestep: Current timestep
    :param output_filename: File name to save the data
    """
    m, n, _ = som_grid.shape
    with open(output_filename, "a") as f:
        f.write(f"# Timestep: {timestep}\n")
        for x in range(m):
            for y in range(n):
                red, green, blue = som_grid[x, y]
                f.write(f"{x} {y} {red:.2f} {green:.2f} {blue:.2f}\n")
        f.write("\n")

def reconstruct_image(input_image, som_grid, output_filename):
    """
    Reconstruct the image by assigning each pixel the RGB values of the closest codebook vector.
    :param input_image: Original input image as a numpy array
    :param som_grid: SOM grid of shape (m, n, 3)
    :param output_filename: File name to save the reconstructed data
    """
    h, w, _ = input_image.shape
    m, n, _ = som_grid.shape

    # Initialize a counter for each neuron in the grid
    pixel_count = np.zeros((m, n), dtype=int)

    with open(output_filename, "w") as f:
        f.write("# Reconstructed Image Data\n")
        for x in range(h):
            for y in range(w):
                input_vector = input_image[x, y]
                winner_pos = find_winning_neuron(som_grid, input_vector)  # Find the closest neuron
                winner_x, winner_y = winner_pos

                # Write pixel data
                red, green, blue = som_grid[winner_x, winner_y]
                f.write(f"{x} {y} {red:.2f} {green:.2f} {blue:.2f}\n")

                # Update the count for this neuron
                pixel_count[winner_x, winner_y] += 1

        f.write("\n# Codebook Vector Statistics\n")
        for i in range(m):
            for j in range(n):
                red, green, blue = som_grid[i, j]
                count = pixel_count[i, j]
                f.write(f"{i} {j} {count} {red:.2f} {green:.2f} {blue:.2f}\n")

def reconstruct_compressed_image(reconstructed_filename, output_image_filename, width, height):
    """
    Reconstruct and save the compressed image from the reconstructed data.
    :param reconstructed_filename: File containing the reconstructed image data
    :param output_image_filename: File name to save the reconstructed image
    :param width: Width of the original image
    :param height: Height of the original image
    """
    # Initialize an empty image array
    reconstructed_image = np.zeros((height, width, 3), dtype=np.uint8)

    # Read the reconstructed data
    with open(reconstructed_filename, "r") as file:
        lines = file.readlines()

    # Process the lines that contain pixel data
    for line in lines:
        if line.startswith("#") or line.strip() == "":
            continue  # Skip comments and empty lines

        #print(line)

        parts = line.split()

        if len(parts) != 5:
            #print(f"Skipping malformed line: {line.strip()}")
            continue

        print(parts)
        x, y, red, green, blue = map(float, parts)
        reconstructed_image[int(x), int(y)] = [int(red), int(green), int(blue)]

    # Save the reconstructed image
    img = Image.fromarray(reconstructed_image)
    img.save(output_image_filename)
    print(f"Reconstructed image saved as {output_image_filename}")


def determine_neighborhood_neurons(som_grid, winner_pos, radius=5.0):
    """
    Determine the neurons within the circular neighborhood of the winning neuron
    using periodic boundary conditions (torus shape). Supports float radius.
    :param som_grid: SOM grid of shape (m, n, 3)
    :param winner_pos: Tuple (x*, y*) of the winning neuron's position
    :param radius: Radius of the neighborhood (can be a float)
    :return: List of (x, y) positions of neurons in the neighborhood
    """
    m, n, _ = som_grid.shape
    x_star, y_star = winner_pos

    # Generate all positions in the rectangular bounding box
    neighbors = []
    int_radius = math.ceil(radius)  # Integer radius to define the bounding box
    for dx in range(-int_radius, int_radius + 1):
        for dy in range(-int_radius, int_radius + 1):
            x = (x_star + dx) % m  # Apply periodic boundary conditions for x
            y = (y_star + dy) % n  # Apply periodic boundary conditions for y

            # Check if the neuron is within the circular radius
            distance_squared = (dx ** 2 + dy ** 2)
            if distance_squared <= radius ** 2:
                neighbors.append((x, y))

    return neighbors


def update_weights_kohonen(som_grid, neighbors, winner_pos, input_vector, learning_rate, sigma_t = 5.0):
    """
    Update the weights of neurons in the neighborhood of the winner.
    :param som_grid: SOM grid of shape (m, n, 3)
    :param neighbors: List of (x, y) positions of neurons in the neighborhood
    :param winner_pos: Tuple (x*, y*) of the winning neuron's position
    :param input_vector: Input RGB vector of shape (3,)
    :param learning_rate: Current learning rate α(t)
    """
    x_star, y_star = winner_pos

    # Update weights for each neuron in the neighborhood
    for neighbor in neighbors:
        x, y = neighbor

        # Calculate the Gaussian neighborhood function
        distance_squared = (x - x_star) ** 2 + (y - y_star) ** 2 #Todo
        h_ci = np.exp(-distance_squared / (2 * sigma_t ** 2))

        # Update the weight of the neuron
        som_grid[x, y] += learning_rate * h_ci * (input_vector - som_grid[x, y])
