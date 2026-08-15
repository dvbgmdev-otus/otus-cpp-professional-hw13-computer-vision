#include "io.h"

#include <algorithm>
#include <fstream>
#include <iterator>
#include <sstream>
#include <stdexcept>

namespace mnist {

Eigen::MatrixXf read_mat_from_stream(size_t rows, size_t cols, std::istream& stream) {
    Eigen::MatrixXf res(rows, cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            float val = 0.0f;
            if (!(stream >> val)) {
                throw std::runtime_error{ "Not enough values in model matrix" };
            }
            res(i, j) = val;
        }
    }
    return res;
}

Eigen::MatrixXf read_mat_from_file(size_t rows, size_t cols, const std::string& filepath) {
    std::ifstream stream{ filepath };
    if (!stream.is_open()) {
        throw std::runtime_error{ "Unable to open model file: " + filepath };
    }
    auto matrix = read_mat_from_stream(rows, cols, stream);
    stream >> std::ws;
    if (!stream.eof()) {
        throw std::runtime_error{ "Unexpected data in model file: " + filepath };
    }
    return matrix;
}

bool read_features(std::istream& stream, Classifier::features_t& features) {
    std::string line;
    std::getline(stream, line);

    features.clear();
    std::istringstream linestream{ line };
    double value;
    while (linestream >> value) {
        features.push_back(value);
    }
    return stream.good();
}

bool read_sample(std::istream& stream,
                 std::size_t& expected_class,
                 Classifier::features_t& features) {
    std::string line;
    if (!std::getline(stream, line)) {
        return false;
    }
    std::replace(line.begin(), line.end(), ',', ' ');
    std::istringstream line_stream{ line };
    long long parsed_class = 0;
    if (!(line_stream >> parsed_class) || parsed_class < 0) {
        throw std::invalid_argument{ "Invalid sample class" };
    }
    expected_class = static_cast<std::size_t>(parsed_class);
    features.clear();
    float value = 0.0f;
    while (line_stream >> value) {
        features.push_back(value);
    }
    if (!line_stream.eof()) {
        throw std::invalid_argument{ "Invalid sample feature" };
    }
    return true;
}

std::vector<float> read_vector(std::istream& stream) {
    std::vector<float> result;

    std::copy(std::istream_iterator<float>(stream),
              std::istream_iterator<float>(),
              std::back_inserter(result));
    return result;
}

}  // namespace mnist
