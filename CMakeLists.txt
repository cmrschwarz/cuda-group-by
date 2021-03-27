cmake_minimum_required(VERSION 3.8)
project(gpu_db_predicate LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 11)

add_executable(cuda-group-by benchmark.cu)
