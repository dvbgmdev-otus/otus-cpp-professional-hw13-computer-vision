#include <Eigen/Dense>
#include <cstddef>
#include <istream>
#include <string>

#include "classifier.h"

namespace mnist {

Eigen::MatrixXf read_mat_from_stream(size_t rows, size_t cols, std::istream&);

Eigen::MatrixXf read_mat_from_file(size_t rows, size_t cols, const std::string&);

bool read_features(std::istream& stream, mnist::Classifier::features_t& features);

bool read_sample(std::istream& stream,
                 std::size_t& expected_class,
                 Classifier::features_t& features);

std::vector<float> read_vector(std::istream&);

}  // namespace mnist
