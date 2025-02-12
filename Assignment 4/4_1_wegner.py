from selforganizedmaps import *
from tqdm import tqdm

def main():
    input_image = load_image(r"gradient_nonlinear.png", set_blue_to_zero=True)

    # Initialize SOM grid and parameters
    grid_rows, grid_cols = 3, 4
    som_grid = initialize_grid(grid_rows, grid_cols, init_blue_to_zero=True)

    # Flatten and shuffle the input image into RGB vectors
    input_vectors, orig_img = generate_input_vectors(input_image)

    # File to save codebook vectors
    codebook_filename = "codebook_vectors.txt"
    reconstructed_filename = "reconstructed_image.txt"

    # Empty codebook_vectors.txt
    with open(codebook_filename, "w") as f:
        pass

    for iteration in tqdm(range(1)):
        for step, input_vector in enumerate(input_vectors):
            # Step 1: Find the winning neuron
            winner_pos = find_winning_neuron(som_grid, input_vector)

            # Step 2: Update the weights of the winning neuron
            som_grid = update_weights(som_grid, winner_pos, input_vector, learning_rate(step))

            # Save codebook vectors every 100 timesteps
            if (step + 1) % 100 == 0:
                save_codebook(som_grid, step + 1, codebook_filename)

    # Reconstruct the image and save data
    reconstruct_image(orig_img, som_grid, reconstructed_filename)

    reconstructed_filename = "reconstructed_image.txt"
    output_image_filename = "compressed_image.png"

    # Example dimensions (adjust as needed)
    original_width = orig_img.shape[1]
    original_height = orig_img.shape[0]

    reconstruct_compressed_image(reconstructed_filename, output_image_filename, original_width, original_height)

    rgb_grid_filename = "Codebook vectors of self-chosen image"
    visualize_rgb_grid(som_grid, filename=rgb_grid_filename)
    print(f"Codebook vectors saved as filename: {rgb_grid_filename}")

    iterations = 1000
    som_gradient = train_som(input_image, grid_rows, grid_cols, iterations)
    generate_voronoi(input_image, som_gradient, "voronoi_gradient.png")
    plot_voronoi_with_weights("voronoi_gradient.png", som_gradient, "voronoi_gradient_with_weights.png")


if __name__ == "__main__":
    main()
