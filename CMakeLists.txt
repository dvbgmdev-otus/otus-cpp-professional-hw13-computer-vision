cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

project(fashion_mnist LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(GTest REQUIRED)

add_library(mnist
    src/io.cpp
    src/io.h
    src/classifier.h
    src/mlp_classifier.cpp
    src/mlp_classifier.h
)

target_include_directories(mnist PUBLIC
    ${PROJECT_SOURCE_DIR}/src
)

target_link_libraries(mnist PUBLIC
    Eigen3::Eigen
)

enable_testing()

add_executable(test_mnist
    tests/test_mlp_classifier.cpp
)

target_link_libraries(test_mnist PRIVATE
    mnist
    GTest::gtest_main
)

add_test(NAME test_mnist COMMAND test_mnist)
set_tests_properties(test_mnist PROPERTIES
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
)
