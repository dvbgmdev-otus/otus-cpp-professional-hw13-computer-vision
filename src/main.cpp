#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>

#include "io.h"
#include "mlp_classifier.h"

namespace {
constexpr std::size_t kInputDim = 784;
constexpr std::size_t kHiddenDim = 128;
constexpr std::size_t kOutputDim = 10;
}  // namespace

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <test.csv> <model directory>\n";
        return EXIT_FAILURE;
    }
    try {
        const std::filesystem::path model_directory{ argv[2] };
        const auto w1 =
            mnist::read_mat_from_file(kInputDim, kHiddenDim, (model_directory / "w1.txt").string());
        const auto w2 = mnist::read_mat_from_file(
            kHiddenDim, kOutputDim, (model_directory / "w2.txt").string());
        const mnist::MlpClassifier classifier{ w1.transpose(), w2.transpose() };
        std::ifstream test_data{ argv[1] };
        if (!test_data.is_open()) {
            throw std::runtime_error{ "Unable to open test data file: " + std::string{ argv[1] } };
        }
        mnist::Classifier::features_t features;
        std::size_t expected_class = 0;
        std::size_t samples_count = 0;
        std::size_t correct_predictions_count = 0;
        while (mnist::read_sample(test_data, expected_class, features)) {
            if (features.size() != kInputDim) {
                throw std::runtime_error{ "Unexpected feature count in sample " +
                                          std::to_string(samples_count + 1) };
            }
            if (expected_class >= classifier.num_classes()) {
                throw std::runtime_error{ "Unexpected class in sample " +
                                          std::to_string(samples_count + 1) };
            }
            if (classifier.predict(features) == expected_class) {
                ++correct_predictions_count;
            }
            ++samples_count;
        }
        if (samples_count == 0) {
            throw std::runtime_error{ "Test data file is empty" };
        }
        const auto accuracy = static_cast<double>(correct_predictions_count) / samples_count;
        std::cout << std::fixed << std::setprecision(6) << accuracy << '\n';
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
