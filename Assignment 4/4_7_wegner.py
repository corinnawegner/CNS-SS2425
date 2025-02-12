from selforganizedmaps import *
from tqdm import tqdm

def main():
    input_image = load_image(r"testfile.png")
    mode = "random"

    for radius in [1,1.3,2,3,4]:
        # Initialize SOM grid and parameters
        grid_rows, grid_cols = 32, 32
        som_grid = initialize_grid(grid_rows, grid_cols, mode=mode)

        visualize_rgb_grid(som_grid, filename=f"initial_map_kohonen_{mode}.png")

        input_vectors, orig_img = generate_input_vectors(input_image)

        codebook_filename = f"codebook_vectors_{radius}.txt"
        reconstructed_filename = f"reconstructed_image_{radius}.txt"
        output_image_filename = f"compressed_image_{radius}.png"

        # Empty codebook_filename
        with open(codebook_filename, "w") as f:
            pass

        for index in tqdm(range(3)):
            for step, input_vector in enumerate(input_vectors):

                winner_pos = find_winning_neuron(som_grid, input_vector)

                neighbors = determine_neighborhood_neurons(som_grid, winner_pos, radius = radius)

                # Update weights for the neighborhood using the radius as standard deviation
                som_grid = update_weights_kohonen(som_grid, neighbors, winner_pos, input_vector, learning_rate(step), sigma_t=radius)

            # Save codebook vectors every 100 timesteps
                if (step + 1) % 100 == 0:
                    save_codebook(som_grid, step + 1, codebook_filename)

        visualize_rgb_grid(som_grid, filename=f"kohonen_map_{radius}_{mode}.png")

        # Reconstruct the image and save data
        reconstruct_image(input_image, som_grid, reconstructed_filename)

        original_width = input_image.shape[1]
        original_height = input_image.shape[0]

        reconstruct_compressed_image(reconstructed_filename, output_image_filename, original_width, original_height)


if __name__ == "__main__":
    main()
