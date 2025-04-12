#include "../include/evaluate.hpp"
#include <iostream>
#include <filesystem>
#include <iomanip>

int main(int argc, char *argv[])
{
    try
    {
        int test_size = 10000; // default test size (tests over entire test set)
        std::string weights_file = "../weights.bin";

        if (argc > 1)
            test_size = std::stoi(argv[1]);
        if (argc > 2)
            weights_file = argv[2]; // only implemented this incase we need to evaluate different CNN models
        if (test_size < 1 || test_size > 10000)
        {
            std::cerr << "Error: test size must be between 1 and 10000\n";
            return 1;
        }
        if (!std::filesystem::exists("../data"))
        {
            std::cerr << "Error: data directory not found" << std::endl;
            return 1;
        }
        if (!std::filesystem::exists(weights_file))
        {
            std::cerr << "Error: weights file not found: " << weights_file << std::endl;
            return 1;
        }
        std::string test_images_file = "../data/t10k-images.idx3-ubyte";
        std::string test_labels_file = "../data/t10k-labels.idx1-ubyte";
        if (!std::filesystem::exists(test_images_file) || !std::filesystem::exists(test_labels_file))
        {
            std::cerr << "Error: MNIST test files not found" << std::endl;
            return 1;
        }

        std::cout << "Initializing CNN..." << std::endl;
        CNN cnn(0.01f);
        std::cout << "Loading weights from " << weights_file << "..." << std::endl;
        cnn.load_weights_from_file(weights_file);

        Evaluator evaluator(cnn);
        auto start = std::chrono::high_resolution_clock::now();
        evaluator.evaluate_model(test_images_file, test_labels_file, test_size);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;
        std::cout << "Successfully loaded " << test_size << " test images." << std::endl;
        std::cout << "Evaluation completed in " << std::fixed << std::setprecision(2) << duration.count() << " seconds" << std::endl;
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}