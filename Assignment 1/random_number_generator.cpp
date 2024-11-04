//
// Created by corin on 11/4/2024.
//

#include <iostream>
#include <fstream>
#include <random>
#include <cmath>
#include <vector>

void generate_random_numbers(int n, double sx, double sy, double phi, double o,const std::string &filename) {
    std::default_random_engine generator;
    std::normal_distribution<double> dist_x(0.0, sx);
    std::normal_distribution<double> dist_y(0.0, sy);

    std::ofstream file(filename);
    if (!file) {
        std::cerr << "Error opening file for writing." << std::endl;
        return;
    }

    // Generate n random (x, y) pairs
    for (int i = 0; i < n; ++i) {
        double x = dist_x(generator);
        double y = dist_y(generator);

        // Rotate the vector (x, y) by angle phi
        double rotated_x = x * cos(phi) - y * sin(phi);
        double rotated_y = x * sin(phi) + y * cos(phi);

        // Shift vector by an offset o
        double offset_x = rotated_x + o;
        double offset_y = rotated_y + o;

        // Write to file
        file << offset_x << "\t" << offset_y << "\n";
    }

    file.close();
    std::cout << "Generated " << n << " random numbers and saved to " << filename << std::endl;
}


int main() {
    int n = 1000;                     // Number of random pairs
    double sx = 1.0;                  // Standard deviation for x
    double sy = 1.0;                  // Standard deviation for y
    double phi = M_PI / 4;            // Rotation angle in radians (45 degrees)
    double offset = 0;
    std::string filename = "random_numbers.txt"; // Output file name

    generate_random_numbers(n, sx, sy, phi, offset, filename);

    return 0;
}
