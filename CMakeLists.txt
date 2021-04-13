cmake_minimum_required(VERSION 3.8)
project(cuda_group_by LANGUAGES CXX CUDA)

# cub complains that it needs c++14 if we do this (just on the server though)
# set(CMAKE_CXX_STANDARD 11)

add_executable(cuda-group-by benchmark.cu)

option(
    CUDA_GROUP_BY_USE_CUB_SUBMODULE 
    "use the cub sumodulle located in ./deps/cub instead of the system version" 
    ON
)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()


# we use this for some ifdefs to make 
# nvcc benchmark.cu work without any parameters
target_compile_definitions(cuda-group-by PRIVATE CUDA_GROUP_BY_CMAKE_BUILD)

if(${CUDA_GROUP_BY_USE_CUB_SUBMODULE})
    target_include_directories(
        cuda-group-by 
        PRIVATE 
        "${CMAKE_CURRENT_SOURCE_DIR}/deps/cub"
    )
endif()
