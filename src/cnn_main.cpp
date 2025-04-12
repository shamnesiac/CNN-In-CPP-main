#include "../include/cnn.hpp"
#include <iostream>
#include <chrono>
#include <filesystem>
#include <iomanip>

int main(int argc, char *argv[])
{
    int train_size = 60000;
    int test_size = 10000;
    int epochs = 10;
    int batch_size = 32; // default values of the hyperparameters

    for (int i = 1; i < argc; i++)
    {
        std::string arg = argv[i];
        if (i + 1 < argc) // reads the hyperparameters from command line arguments
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

    try // checks for the existence of the right directory and files (for debugging)
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

        std::cout << "Initializing CNN..." << std::endl;
        CNN cnn(0.001f); // makes CNN with learning rate = 0.001

        std::cout << "Loading MNIST data..." << std::endl;
        std::cout << "Using " << train_size << " training images and " << test_size << " test images" << std::endl;

        auto [train_images, train_labels] = CNN::load_mnist(
            "../data/train-images.idx3-ubyte",
            "../data/train-labels.idx1-ubyte",
            train_size); // loads training set

        std::cout << "Successfully loaded training data." << std::endl;
        std::cout << "Number of training images: " << train_images.size() << std::endl;

        auto [test_images, test_labels] = CNN::load_mnist(
            "../data/t10k-images.idx3-ubyte",
            "../data/t10k-labels.idx1-ubyte",
            test_size); // loads testing set

        std::cout << "Successfully loaded test data." << std::endl;
        std::cout << "Number of test images: " << test_images.size() << std::endl;

        std::cout << "\nTraining CNN..." << std::endl;
        std::cout << "Epochs: " << epochs << ", Batch size: " << batch_size << std::endl;
        auto start_time = std::chrono::high_resolution_clock::now();

        cnn.train(train_images, train_labels, test_images, test_labels, epochs, batch_size);

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);

        std::cout << "Training completed in " << duration.count() << " seconds" << std::endl;

        std::cout << "\nSaving best weights to weights.bin..." << std::endl;
        cnn.save_weights_to_file("../weights.bin");

        float accuracy = cnn.evaluate(test_images, test_labels, false);
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