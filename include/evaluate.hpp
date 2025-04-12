#ifndef EVALUATE_HPP
#define EVALUATE_HPP

#include "cnn.hpp"
#include <vector>
#include <string>

class Evaluator
{
private:
    CNN &cnn;
    int num_threads;
    std::vector<std::vector<int>> confusion_matrix;

    void compute_confusion_matrix(
        const std::vector<std::vector<std::vector<float>>> &test_data,
        const std::vector<std::vector<float>> &test_labels);
    void print_confusion_matrix() const;
    void print_progress(size_t current, size_t total) const;

public:
    explicit Evaluator(CNN &cnn_model);
    void evaluate_model(
        const std::string &test_images_file,
        const std::string &test_labels_file,
        int test_size = 10000);
    const std::vector<std::vector<int>> &get_confusion_matrix() const { return confusion_matrix; }
    float get_overall_accuracy() const;
    std::vector<float> get_class_accuracies() const;
};

#endif