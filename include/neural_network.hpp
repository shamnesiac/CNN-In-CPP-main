#ifndef NEURAL_NETWORK_HPP
#define NEURAL_NETWORK_HPP

#include <vector>
#include <random>
#include <string>
#include <fstream>

class NeuralNetwork
{
private:
    int input_size;
    int output_size;
    float learning_rate;
    std::mt19937 rng;
    std::vector<std::vector<float>> weights;
    std::vector<float> bias;
    std::vector<float> output;
    std::vector<std::vector<float>> best_weights;
    std::vector<float> best_bias;
    void initialize_weights();
    void save_best_model();
    void restore_best_model();
    std::vector<float> softmax(const std::vector<float> &x);
    float cross_entropy_loss(const std::vector<float> &predicted, const std::vector<float> &target);

public:
    NeuralNetwork(int input_size, int output_size, float learning_rate = 0.01f);
    void train(const std::vector<std::vector<float>> &training_data,
               const std::vector<std::vector<float>> &training_labels,
               const std::vector<std::vector<float>> &test_data,
               const std::vector<std::vector<float>> &test_labels,
               int epochs,
               int batch_size = 32);
    std::vector<float> forward(const std::vector<float> &input);
    void backward(const std::vector<float> &input, const std::vector<float> &target);
    float evaluate(const std::vector<std::vector<float>> &test_data,
                   const std::vector<std::vector<float>> &test_labels,
                   bool print_predictions = false);
    void load_weights(const std::string &filename);
    void save_weights(const std::string &filename) const;
    static std::pair<std::vector<std::vector<float>>, std::vector<std::vector<float>>>
    load_mnist(const std::string &images_file, const std::string &labels_file, int num_images);
};

#endif