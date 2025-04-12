#ifndef CNN_HPP
#define CNN_HPP

#include <vector>
#include <random>
#include <cmath>
#include <string>

class CNN
{
private:
    float learning_rate;
    std::mt19937 rng;

    std::vector<std::vector<std::vector<std::vector<float>>>> conv1_weights; // 6 filters of 5x5
    std::vector<float> conv1_biases;
    std::vector<std::vector<std::vector<std::vector<float>>>> conv2_weights; // 16 filters of 5x5
    std::vector<float> conv2_biases;
    std::vector<std::vector<float>> fc1_weights; // 400 x 120
    std::vector<float> fc1_biases;
    std::vector<std::vector<float>> fc2_weights; // 120 x 84
    std::vector<float> fc2_biases;
    std::vector<std::vector<float>> output_weights; // 84 x 10
    std::vector<float> output_biases;

    std::vector<std::vector<std::vector<std::vector<float>>>> best_conv1_weights;
    std::vector<float> best_conv1_biases;
    std::vector<std::vector<std::vector<std::vector<float>>>> best_conv2_weights;
    std::vector<float> best_conv2_biases;
    std::vector<std::vector<float>> best_fc1_weights;
    std::vector<float> best_fc1_biases;
    std::vector<std::vector<float>> best_fc2_weights;
    std::vector<float> best_fc2_biases;
    std::vector<std::vector<float>> best_output_weights;
    std::vector<float> best_output_biases;
    float best_accuracy;

    std::vector<std::vector<std::vector<float>>> conv1_output;
    std::vector<std::vector<std::vector<float>>> pool1_output;
    std::vector<std::vector<std::vector<float>>> conv2_output;
    std::vector<std::vector<std::vector<float>>> pool2_output;
    std::vector<float> fc1_output;
    std::vector<float> fc2_output;
    std::vector<float> final_output;
    void initialize_weights();
    void save_best_model();
    void restore_best_model();
    float relu(float x);
    float relu_derivative(float x);
    std::vector<float> softmax(const std::vector<float> &x);
    float cross_entropy_loss(const std::vector<float> &predicted, const std::vector<float> &target);
    std::vector<std::vector<std::vector<float>>> convolution(
        const std::vector<std::vector<std::vector<float>>> &input,
        const std::vector<std::vector<std::vector<std::vector<float>>>> &filters,
        const std::vector<float> &biases);
    std::vector<std::vector<std::vector<float>>> max_pooling(
        const std::vector<std::vector<std::vector<float>>> &input);

public:
    CNN(float lr = 0.001);
    void train(const std::vector<std::vector<std::vector<float>>> &training_data,
               const std::vector<std::vector<float>> &training_labels,
               const std::vector<std::vector<std::vector<float>>> &test_data,
               const std::vector<std::vector<float>> &test_labels,
               int epochs,
               int batch_size = 32);
    std::vector<float> forward(const std::vector<std::vector<float>> &input);
    void backward(const std::vector<std::vector<float>> &input, const std::vector<float> &target);
    float evaluate(const std::vector<std::vector<std::vector<float>>> &test_data,
                   const std::vector<std::vector<float>> &test_labels,
                   bool print_predictions = false);
    int predict(const std::vector<std::vector<float>> &input);
    void save_weights_to_file(const std::string &filename) const;
    void load_weights_from_file(const std::string &filename);
    static std::pair<std::vector<std::vector<std::vector<float>>>, std::vector<std::vector<float>>>
    load_mnist(const std::string &images_file, const std::string &labels_file, int num_images);
};

#endif