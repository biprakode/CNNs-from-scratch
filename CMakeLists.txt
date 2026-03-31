cmake_minimum_required(VERSION 3.20)
project(CNNs_from_scratch C)

set(CMAKE_C_STANDARD 11)

include_directories(csrc)

add_executable(tensor_test
        csrc/tensor.c
        csrc/tensor.h
        csrc/c_tests/tensor_tests.c)

add_executable(matmul_test csrc/matmul.c
        csrc/matmul.h)

add_executable(conv2d_test csrc/conv2d.c csrc/matmul.c
        csrc/matmul.h)

add_library(cnn SHARED csrc/matmul.c csrc/conv2d.c csrc/activation.c)

include(CTest)
find_program(MEMORYCHECK_COMMAND valgrind)
set(MEMORYCHECK_COMMAND_OPTIONS "--leak-check=full --error-exitcode=1")
add_test(NAME tensor_test COMMAND tensor_test)

