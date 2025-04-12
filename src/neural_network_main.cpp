#include "../include/neural_network.hpp"
#include <iostream>
#include <chrono>
#include <iomanip>
#include <filesystem>

int main(int argc, char *argv[])
{
    int train_size = 60000;
    int test_size = 10000;
    int epochs = 10;
    int batch_size = 32;

    for (int i = 1; i < argc; i++)
    {
        std::string arg = argv[i];
        if (i + 1 < argc)
        {
            if (arg == "--train-size")
            {
                train_size = std::stoi(argv[++i]);
                if (train_size < 1 || train_size > 60000)
                {
                    std::cerr << "Error: train-size must be between 1 and 60000\n";
                    return 1;
                }
            }
            else if (arg == "--test-size")
            {
                test_size = std::stoi(argv[++i]);
                if (test_size < 1 || test_size > 10000)
                {
                    std::cerr << "Error: test-size must be between 1 and 10000\n";
                    return 1;
                }
            }
            else if (arg == "--epochs")
            {
                epochs = std::stoi(argv[++i]);
                if (epochs < 1)
                {
                    std::cerr << "Error: epochs must be positive\n";
                    return 1;
                }
            }
            else if (arg == "--batch-size")
            {
                batch_size = std::stoi(argv[++i]);
                if (batch_size < 1)
                {
                    std::cerr << "Error: batch-size must be positive\n";
                    return 1;
                }
            }
        }
    }

    try
    {
        if (!std::filesystem::exists("../data"))
        {
            std::cerr << "Error: data directory not found!" << std::endl;
            std::cerr << "Please create a 'data' directory and place MNIST dataset files in it." << std::endl;
            return 1;
        }

        std::vector<std::string> required_files = {
            "../data/train-images.idx3-ubyte",
            "../data/train-labels.idx1-ubyte",
            "../data/t10k-images.idx3-ubyte",
            "../data/t10k-labels.idx1-ubyte"};

        bool files_missing = false;
        for (const auto &file : required_files)
        {
            if (!std::filesystem::exists(file))
            {
                std::cerr << "Error: Required file not found: " << file << std::endl;
                files_missing = true;
            }
        }

        if (files_missing)
        {
            std::cerr << "\nMissing Files!" << std::endl;
            return 1;
        }

        std::cout << "Initializing Neural Network..." << std::endl;
        NeuralNetwork nn(784, 10, 0.01f); // input size = 784, output size = 10, learning rate = 0.01

        std::cout << "Loading MNIST data..." << std::endl;
        std::cout << "Using " << train_size << " training images and " << test_size << " test images" << std::endl;

        auto [train_data, train_labels] = NeuralNetwork::load_mnist(
            "../data/train-images.idx3-ubyte",
            "../data/train-labels.idx1-ubyte",
            train_size);

        std::cout << "Successfully loaded training data." << std::endl;
        std::cout << "Number of training images: " << train_data.size() << std::endl;

        auto [test_data, test_labels] = NeuralNetwork::load_mnist(
            "../data/t10k-images.idx3-ubyte",
            "../data/t10k-labels.idx1-ubyte",
            test_size);

        std::cout << "Successfully loaded test data." << std::endl;
        std::cout << "Number of test images: " << test_data.size() << std::endl;

        std::cout << "\nTraining Neural Network..." << std::endl;
        std::cout << "Epochs: " << epochs << ", Batch size: " << batch_size << std::endl;
        auto start_time = std::chrono::high_resolution_clock::now();

        nn.train(train_data, train_labels, test_data, test_labels, epochs, batch_size);

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);

        std::cout << "Training completed in " << duration.count() << " seconds" << std::endl;

        std::cout << "\nSaving best weights to neural_network_weights.bin..." << std::endl;
        nn.save_weights("../neural_network_weights.bin");

        float accuracy = nn.evaluate(test_data, test_labels, false);
        std::cout << "\nOverall test accuracy: " << std::fixed << std::setprecision(2)
                  << (accuracy * 100) << "%" << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    catch (...)
    {
        std::cerr << "Unknown error occurred" << std::endl;
        return 1;
    }

    return 0;
}