from selforganizedmaps import *
from tqdm import tqdm

def main():
    input_image_1 = load_image(r"gradient.png")
    input_image_2 = load_image(r"gradient_nonlinear.png")

    radius = 1.2

    names = ["gradient", "gradient_nonlinear"]

    for img, name in zip([input_image_1, input_image_2], names):
        # Initialize SOM grid and parameters
        grid_rows, grid_cols = 32, 32
        som_grid = initialize_grid(grid_rows, grid_cols)

        # Flatten and shuffle the input image into RGB vectors
        input_vectors = generate_input_vectors(img)

        # File to save codebook vectors
        codebook_filename = f"codebook_vectors_{name}.txt"
        reconstructed_filename = f"reconstructed_image_{name}.txt"

        # Determine neighborhood neurons

        # Empty codebook_vectors.txt
        with open(codebook_filename, "w") as f:
            pass

        for step in tqdm(range(100)):
            for input_vector in input_vectors:

                winner_pos = find_winning_neuron(som_grid, input_vector)

                neighbors = determine_neighborhood_neurons(som_grid, winner_pos, radius = radius)

                # Update weights for the neighborhood
                update_weights_kohonen(som_grid, neighbors, winner_pos, input_vector, learning_rate(0))

            # Save codebook vectors every 100 timesteps
            if (step + 1) % 100 == 0:
                save_codebook(som_grid, step + 1, codebook_filename)

        # Reconstruct the image and save data
        reconstruct_image(img, som_grid, reconstructed_filename)

        reconstructed_filename =f"reconstructed_image_{name}.txt"
        output_image_filename = f"compressed_image_{name}.png"

        # Example dimensions (adjust as needed)
        original_width = img.shape[1]
        original_height = img.shape[0]

        reconstruct_compressed_image(reconstructed_filename, output_image_filename, original_width, original_height)


if __name__ == "__main__":
    main()