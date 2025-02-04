from selforganizedmaps import *
from tqdm import tqdm

"""
def generate_input_vectors(image_array, copies=3):
    
    Generate shuffled input vectors from multiple copies of the image.
    :param image_array: numpy array of shape (height, width, 3)
    :param copies: Number of shuffled copies to use
    :return: List of RGB tuples shuffled randomly

    h, w, _ = image_array.shape
    input_vectors = np.tile(image_array.reshape(h * w, 3), (copies, 1))  # Create copies
    np.random.shuffle(input_vectors)  # Shuffle all copies together
    return input_vectors
"""

def main():
    input_image = load_image(r"testfile.png")

    for radius in [2]:
        # Initialize SOM grid and parameters
        grid_rows, grid_cols = 32, 32
        som_grid = initialize_grid(grid_rows, grid_cols)

        # Flatten and shuffle the input image into RGB vectors
        input_vectors = generate_input_vectors(input_image)

        # File to save codebook vectors
        codebook_filename = f"codebook_vectors_{radius}.txt"
        reconstructed_filename = f"reconstructed_image_{radius}.txt"

        # Determine neighborhood neurons

        # Empty codebook_vectors.txt
        with open(codebook_filename, "w") as f:
            pass

        for index in tqdm(range(3)):
            for step, input_vector in enumerate(input_vectors):

                winner_pos = find_winning_neuron(som_grid, input_vector)

                neighbors = determine_neighborhood_neurons(som_grid, winner_pos, radius = radius)

                # Update weights for the neighborhood
                update_weights_kohonen(som_grid, neighbors, winner_pos, input_vector, learning_rate(step))

            # Save codebook vectors every 100 timesteps
                if (step + 1) % 100 == 0:
                    save_codebook(som_grid, step + 1, codebook_filename)

        # Reconstruct the image and save data
        reconstruct_image(input_image, som_grid, reconstructed_filename)

        reconstructed_filename =f"reconstructed_image_{radius}.txt"
        output_image_filename = f"compressed_image_{radius}.png"

        # Example dimensions (adjust as needed)
        original_width = input_image.shape[1]
        original_height = input_image.shape[0]

        reconstruct_compressed_image(reconstructed_filename, output_image_filename, original_width, original_height)


if __name__ == "__main__":
    main()
