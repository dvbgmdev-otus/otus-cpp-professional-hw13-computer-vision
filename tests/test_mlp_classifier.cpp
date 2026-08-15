#include <gtest/gtest.h>

#include <fstream>

#include "io.h"
#include "mlp_classifier.h"

namespace {
constexpr std::size_t kInputDim = 784;
constexpr std::size_t kHiddenDim = 128;
constexpr std::size_t kOutputDim = 10;
class MlpClassifierTest : public testing::Test {
protected:
    MlpClassifierTest()
        : m_classifier{
              mnist::read_mat_from_file(kInputDim, kHiddenDim, "model/w1.txt").transpose(),
              mnist::read_mat_from_file(kHiddenDim, kOutputDim, "model/w2.txt").transpose()
          } {}
    mnist::MlpClassifier m_classifier;
};
}  // namespace

#if (1)  // Part 1. Проверка классификатора MLP
// Test 1.1. Классификатор возвращает количество классов Fashion MNIST
TEST_F(MlpClassifierTest, Metadata_WhenModelLoaded_ReturnsTenClasses) {
    EXPECT_EQ(kOutputDim, m_classifier.num_classes());
}
// Test 1.2. Классификатор повторяет контрольные предсказания Python-модели
TEST_F(MlpClassifierTest, Prediction_WhenReferenceDataProvided_MatchesPythonModel) {
    std::ifstream test_data{ "data/test_data_mlp.txt" };
    ASSERT_TRUE(test_data.is_open());
    mnist::MlpClassifier::features_t features;
    std::size_t expected_class = 0;
    std::size_t samples_count = 0;
    while (mnist::read_sample(test_data, expected_class, features)) {
        ASSERT_EQ(kInputDim, features.size());
        EXPECT_EQ(expected_class, m_classifier.predict(features));
        ++samples_count;
    }
    EXPECT_EQ(10u, samples_count);
}
#endif

#if (1)  // Part 2. Проверка чтения данных Fashion MNIST
// Test 2.1. CSV-строка преобразуется в класс и 784 признака
TEST(DataReader, Sample_WhenCsvRowProvided_ReturnsClassAndFeatures) {
    std::ifstream test_data{ "data/test.csv" };
    ASSERT_TRUE(test_data.is_open());
    mnist::Classifier::features_t features;
    std::size_t expected_class = 0;
    ASSERT_TRUE(mnist::read_sample(test_data, expected_class, features));
    EXPECT_EQ(7u, expected_class);
    EXPECT_EQ(kInputDim, features.size());
}
#endif
