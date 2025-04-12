#include "../include/evaluate.hpp"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <filesystem>
#include <omp.h>
#include <algorithm>

Evaluator::Evaluator(CNN &cnn_model) : cnn(cnn_model)
{
    num_threads = omp_get_max_threads();
    omp_set_num_threads(num_threads);
    confusion_matrix = std::vector<std::vector<int>>(10, std::vector<int>(10, 0));
}

void Evaluator::print_progress(size_t current, size_t total) const
{
    std::cout << "\rProcessed " << current << "/" << total << " samples" << std::flush;
}

void Evaluator::compute_confusion_matrix(
    const std::vector<std::vector<std::vector<float>>> &test_data,
    const std::vector<std::vector<float>> &test_labels)
{
    confusion_matrix = std::vector<std::vector<int>>(10, std::vector<int>(10, 0)); // initializing confusion matrix
    for (size_t i = 0; i < test_data.size(); ++i)                                  // processing every test sample
    {
        std::vector<float> predicted = cnn.forward(test_data[i]);
        int predicted_label = std::max_element(predicted.begin(), predicted.end()) - predicted.begin();
        int true_label = std::max_element(test_labels[i].begin(), test_labels[i].end()) - test_labels[i].begin();
        confusion_matrix[true_label][predicted_label]++; // update confusion matrix

        if ((i + 1) % 100 == 0)
        {
            std::cout << "\rProcessed " << i + 1 << "/" << test_data.size() << " samples" << std::flush;
        }
    }
    std::cout << std::endl;
    std::cout << "\nConfusion Matrix:\n";
    std::cout << "X-Axis: Predicted labels (0-9)\n";
    std::cout << "Y-Axis: Actual labels (0-9)\n\n";
    std::cout << "     ";
    for (int i = 0; i < 10; i++)
    {
        std::cout << std::setw(6) << i;
    }
    std::cout << "  │ Class Acc." << std::endl;
    std::cout << "   ┌";
    for (int i = 0; i < 62; i++)
        std::cout << "─";
    std::cout << "┐" << std::endl;
    for (int i = 0; i < 10; i++)
    {
        std::cout << std::setw(2) << i << " │";
        int row_total = 0;
        for (int j = 0; j < 10; j++)
        {
            std::cout << std::setw(6) << confusion_matrix[i][j];
            row_total += confusion_matrix[i][j];
        }

        float row_accuracy = row_total > 0 ? (float)confusion_matrix[i][i] / row_total * 100 : 0.0f;
        std::cout << "  │" << std::setw(7) << std::fixed << std::setprecision(1) << row_accuracy << "%" << std::endl;
    }

    std::cout << "   └";
    for (int i = 0; i < 62; i++)
        std::cout << "─";
    std::cout << "┘" << std::endl;

    std::cout << "Acc│";
    for (int j = 0; j < 10; j++)
    {
        int col_total = 0;
        for (int i = 0; i < 10; i++)
        {
            col_total += confusion_matrix[i][j];
        }
        float col_accuracy = col_total > 0 ? (float)confusion_matrix[j][j] / col_total * 100 : 0.0f;
        std::cout << std::setw(6) << std::fixed << std::setprecision(1) << col_accuracy;
    }
    std::cout << "% │" << std::endl;
    int correct = 0;
    int total = 0;
    for (int i = 0; i < 10; i++)
    {
        for (int j = 0; j < 10; j++)
        {
            if (i == j)
                correct += confusion_matrix[i][j];
            total += confusion_matrix[i][j];
        }
    }
    float accuracy = (float)correct / total * 100;
    std::cout << "\nOverall Accuracy: " << std::fixed << std::setprecision(2) << accuracy << "%" << std::endl;
    std::cout << "Total samples: " << total << std::endl;
}

void Evaluator::print_confusion_matrix() const
{
    std::cout << "\nConfusion Matrix:\n";
    std::cout << "Predicted →\n";
    std::cout << "   ";
    for (int i = 0; i < 10; ++i)
    {
        std::cout << std::setw(5) << i;
    }
    std::cout << "  ← True\n";
    for (int i = 0; i < 10; ++i)
    {
        std::cout << std::setw(2) << i << " ";
        for (int j = 0; j < 10; ++j)
        {
            std::cout << std::setw(5) << confusion_matrix[i][j];
        }
        std::cout << std::endl;
    }
    std::cout << "\nPer-class accuracy:\n";
    auto class_accuracies = get_class_accuracies();
    for (int i = 0; i < 10; ++i)
    {
        std::cout << "Class " << i << ": " << std::fixed << std::setprecision(2)
                  << class_accuracies[i] << "%" << std::endl;
    }
    std::cout << "\nOverall accuracy: " << std::fixed << std::setprecision(2)
              << get_overall_accuracy() << "%" << std::endl;
}

float Evaluator::get_overall_accuracy() const
{
    int total_correct = 0;
    int total_samples = 0;

    for (int i = 0; i < 10; ++i)
    {
        for (int j = 0; j < 10; ++j)
        {
            if (i == j)
                total_correct += confusion_matrix[i][j];
            total_samples += confusion_matrix[i][j];
        }
    }
    return total_samples > 0 ? (static_cast<float>(total_correct) / total_samples * 100) : 0.0f;
}

std::vector<float> Evaluator::get_class_accuracies() const
{
    std::vector<float> accuracies(10);
    for (int i = 0; i < 10; ++i)
    {
        int class_total = 0;
        for (int j = 0; j < 10; ++j)
        {
            class_total += confusion_matrix[i][j];
        }
        accuracies[i] = class_total > 0 ? (static_cast<float>(confusion_matrix[i][i]) / class_total * 100) : 0.0f;
    }
    return accuracies;
}

void Evaluator::evaluate_model(
    const std::string &test_images_file,
    const std::string &test_labels_file,
    int test_size)
{
    std::cout << "Loading test data..." << std::endl;
    auto [test_data, test_labels] = CNN::load_mnist(test_images_file, test_labels_file, test_size);
    std::cout << "Loaded " << test_data.size() << " test images" << std::endl;
    std::cout << "Computing confusion matrix..." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();
    compute_confusion_matrix(test_data, test_labels);
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    std::cout << "\nEvaluation completed in " << duration.count() << " seconds" << std::endl;
}