from selforganizedmaps import *
from tqdm import tqdm

def main():
    input_image_1 = load_image(r"gradient.png")
    input_image_2 = load_image(r"gradient_nonlinear.png")

    radius = 1.2

    names = ["gradient", "gradient_nonlinear"]

    for img, name in zip([input_image_1, input_image_2], names):
        # Initialize SOM grid and parameters
        grid_rows, grid_cols = 10, 10
        som_grid = initialize_grid(grid_rows, grid_cols, init_blue_to_zero=True)

        # Flatten and shuffle the input image into RGB vectors
        input_vectors, original_image = generate_input_vectors(img)

        # File to save codebook vectors
        codebook_filename = f"codebook_vectors_{name}.txt"
        reconstructed_filename = f"reconstructed_image_{name}.txt"

        # Determine neighborhood neurons

        # Empty codebook_vectors.txt
        with open(codebook_filename, "w") as f:
            pass

        for index in tqdm(range(3)):
            for step, input_vector in enumerate(input_vectors):

                winner_pos = find_winning_neuron(som_grid, input_vector)

                neighbors = determine_neighborhood_neurons(som_grid, winner_pos, radius=radius)

                # Update weights for the neighborhood using the radius as standard deviation
                som_grid = update_weights_kohonen(som_grid, neighbors, winner_pos, input_vector, learning_rate(step),
                                                  sigma_t=radius)

                # Save codebook vectors every 100 timesteps
                if (step + 1) % 100 == 0:
                    save_codebook(som_grid, step + 1, codebook_filename)

        visualize_rgb_grid(som_grid, filename=f"Kohonen_map_{name}.png")

        red_part = som_grid[:, :, 0].flatten()/256  # Flatten to 1D array
        green_part = som_grid[:, :, 1].flatten()/256  # Flatten to 1D array

        colors = np.column_stack((red_part, green_part, np.zeros_like(red_part)))  # B = 0

        # Create scatter plot in the red-green plane
        plt.scatter(red_part, green_part, c=colors, edgecolors='k')

        # Labels and title
        plt.xlabel("Red Channel")
        plt.ylabel("Green Channel")
        plt.title(f"Projection of codebook vectors from {name}")

        # Show the plot
        plt.show()




if __name__ == "__main__":
    main()