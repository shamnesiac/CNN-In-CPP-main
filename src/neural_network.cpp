#include "../include/neural_network.hpp"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <random>
#include <iomanip>
#include <chrono>

NeuralNetwork::NeuralNetwork(int input_size, int output_size, float learning_rate)
    : input_size(input_size), output_size(output_size), learning_rate(learning_rate)
{
    rng.seed(std::random_device{}());
    initialize_weights();
}

void NeuralNetwork::initialize_weights()
{
    std::normal_distribution<float> dist(0.0f, 0.1f);

    weights.resize(input_size, std::vector<float>(output_size));
    for (int i = 0; i < input_size; ++i)
    {
        for (int j = 0; j < output_size; ++j)
        {
            weights[i][j] = dist(rng);
        }
    }

    bias.resize(output_size, 0.0f);
    for (int i = 0; i < output_size; ++i)
    {
        bias[i] = dist(rng);
    }

    output.resize(output_size);
}

std::vector<float> NeuralNetwork::softmax(const std::vector<float> &x)
{
    std::vector<float> exp_x(x.size());
    float max_val = *std::max_element(x.begin(), x.end());
    float sum = 0.0f;

    for (size_t i = 0; i < x.size(); ++i)
    {
        exp_x[i] = std::exp(x[i] - max_val);
        sum += exp_x[i];
    }

    for (size_t i = 0; i < x.size(); ++i)
    {
        exp_x[i] /= sum;
    }

    return exp_x;
}

float NeuralNetwork::cross_entropy_loss(const std::vector<float> &predicted, const std::vector<float> &target)
{
    float loss = 0.0f;
    for (size_t i = 0; i < predicted.size(); ++i)
    {
        loss -= target[i] * std::log(predicted[i] + 1e-15f);
    }
    return loss;
}

std::vector<float> NeuralNetwork::forward(const std::vector<float> &input)
{
    for (int j = 0; j < output_size; ++j)
    {
        output[j] = bias[j];
        for (int i = 0; i < input_size; ++i)
        {
            output[j] += input[i] * weights[i][j];
        }
    }
    return softmax(output);
}

void NeuralNetwork::backward(const std::vector<float> &input, const std::vector<float> &target)
{
    std::vector<float> grad_output = output;
    for (int i = 0; i < output_size; ++i)
    {
        grad_output[i] -= target[i];
    }

    for (int i = 0; i < input_size; ++i)
    {
        for (int j = 0; j < output_size; ++j)
        {
            weights[i][j] -= learning_rate * grad_output[j] * input[i];
        }
    }

    for (int j = 0; j < output_size; ++j)
    {
        bias[j] -= learning_rate * grad_output[j];
    }
}

void NeuralNetwork::train(const std::vector<std::vector<float>> &training_data,
                          const std::vector<std::vector<float>> &training_labels,
                          const std::vector<std::vector<float>> &test_data,
                          const std::vector<std::vector<float>> &test_labels,
                          int epochs,
                          int batch_size)
{
    size_t num_samples = training_data.size();
    std::vector<size_t> indices(num_samples);
    std::iota(indices.begin(), indices.end(), 0);

    float best_accuracy = 0.0f;

    for (int epoch = 0; epoch < epochs; ++epoch)
    {
        float total_loss = 0.0f;
        std::shuffle(indices.begin(), indices.end(), rng);

        for (size_t batch_start = 0; batch_start < num_samples; batch_start += batch_size)
        {
            size_t current_batch_size = std::min(batch_size, static_cast<int>(num_samples - batch_start));
            float batch_loss = 0.0f;

            for (size_t i = batch_start; i < batch_start + current_batch_size; ++i)
            {
                size_t idx = indices[i];
                std::vector<float> predicted = forward(training_data[idx]);
                batch_loss += cross_entropy_loss(predicted, training_labels[idx]);
                backward(training_data[idx], training_labels[idx]);

                std::cout << "\rEpoch " << std::setw(1) << epoch + 1
                          << " - Processing image " << std::setw(5) << (i + 1) << "/" << std::setw(5) << num_samples
                          << " (Batch " << (batch_start / batch_size + 1) << "/" << (num_samples + batch_size - 1) / batch_size << ")"
                          << std::flush;
            }
            total_loss += batch_loss;
        }

        float avg_loss = total_loss / num_samples;
        float train_accuracy = evaluate(training_data, training_labels, false);
        float test_accuracy = evaluate(test_data, test_labels, false);

        std::cout << " - Loss: " << std::fixed << std::setprecision(4) << avg_loss
                  << " - Train Acc: " << std::fixed << std::setprecision(2) << (train_accuracy * 100) << "%"
                  << " - Test Acc: " << std::fixed << std::setprecision(2) << (test_accuracy * 100) << "%";

        if (test_accuracy > best_accuracy)
        {
            best_accuracy = test_accuracy;
            save_best_model();
            std::cout << " (Best)" << std::endl;
        }
        else
        {
            std::cout << std::endl;
        }
    }

    restore_best_model();
    std::cout << "\nRestored best model with test accuracy: " << std::fixed << std::setprecision(2)
              << (best_accuracy * 100) << "%" << std::endl;
}

std::pair<std::vector<std::vector<float>>, std::vector<std::vector<float>>>
NeuralNetwork::load_mnist(const std::string &images_file, const std::string &labels_file, int num_images)
{
    std::ifstream images(images_file, std::ios::binary);
    std::ifstream labels(labels_file, std::ios::binary);

    if (!images || !labels)
    {
        throw std::runtime_error("Could not open MNIST files");
    }

    images.seekg(16);
    labels.seekg(8);

    std::vector<std::vector<float>> data(num_images, std::vector<float>(784));
    std::vector<std::vector<float>> labels_one_hot(num_images, std::vector<float>(10, 0.0f));

    for (int i = 0; i < num_images; ++i)
    {
        unsigned char pixel;
        for (int j = 0; j < 784; ++j)
        {
            images.read(reinterpret_cast<char *>(&pixel), 1);
            data[i][j] = static_cast<float>(pixel) / 255.0f;
        }
        unsigned char label;
        labels.read(reinterpret_cast<char *>(&label), 1);
        labels_one_hot[i][label] = 1.0f;
    }

    std::cout << "Successfully loaded " << num_images << " images and labels" << std::endl;
    return {data, labels_one_hot};
}

float NeuralNetwork::evaluate(const std::vector<std::vector<float>> &test_data,
                              const std::vector<std::vector<float>> &test_labels,
                              bool print_predictions)
{
    int correct = 0;
    int total = test_data.size();

    for (size_t i = 0; i < test_data.size(); ++i)
    {
        std::vector<float> pred = forward(test_data[i]);
        int pred_class = std::max_element(pred.begin(), pred.end()) - pred.begin();
        int true_class = std::max_element(test_labels[i].begin(), test_labels[i].end()) - test_labels[i].begin();

        if (pred_class == true_class)
        {
            correct++;
        }

        if (print_predictions)
        {
            std::cout << "Sample " << i << ": Predicted=" << pred_class << ", True=" << true_class << std::endl;
        }
    }

    return static_cast<float>(correct) / total;
}

void NeuralNetwork::save_best_model()
{
    best_weights = weights;
    best_bias = bias;
}

void NeuralNetwork::restore_best_model()
{
    weights = best_weights;
    bias = best_bias;
}

void NeuralNetwork::save_weights(const std::string &filename) const
{
    std::ofstream file(filename, std::ios::binary);
    if (!file)
    {
        throw std::runtime_error("Could not open file for saving weights");
    }
    file.write(reinterpret_cast<const char *>(&input_size), sizeof(input_size));
    file.write(reinterpret_cast<const char *>(&output_size), sizeof(output_size));

    for (const auto &row : weights)
    {
        file.write(reinterpret_cast<const char *>(row.data()), row.size() * sizeof(float));
    }
    file.write(reinterpret_cast<const char *>(bias.data()), bias.size() * sizeof(float));
}

void NeuralNetwork::load_weights(const std::string &filename)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file)
    {
        throw std::runtime_error("Could not open file for loading weights");
    }

    file.read(reinterpret_cast<char *>(&input_size), sizeof(input_size));
    file.read(reinterpret_cast<char *>(&output_size), sizeof(output_size));
    weights.resize(input_size, std::vector<float>(output_size));
    for (auto &row : weights)
    {
        file.read(reinterpret_cast<char *>(row.data()), row.size() * sizeof(float));
    }
    bias.resize(output_size);
    file.read(reinterpret_cast<char *>(bias.data()), bias.size() * sizeof(float));
}