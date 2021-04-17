cmake_minimum_required(VERSION 3.8)
project(cuda_group_by LANGUAGES CXX CUDA)

# required by cub, the rest should be fine with C++11
set(CMAKE_CXX_STANDARD 14)

# determine Cuda Compute Capability
    execute_process(
        COMMAND "nvcc" "${CMAKE_CURRENT_SOURCE_DIR}/get_cc.cu" "-o" "get_cc"
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        RESULT_VARIABLE "PROC_RES"
    )
    if(NOT "${PROC_RES}" EQUAL "0")
        message(FATAL_ERROR "failed to get cc by compiling get_cc.cu: ${PROC_RES}")
    endif()

    execute_process(
        COMMAND "${CMAKE_CURRENT_BINARY_DIR}/get_cc"
        OUTPUT_VARIABLE "CUDA_COMP_CAP"
        RESULT_VARIABLE "PROC_RES"
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(NOT "${PROC_RES}" EQUAL "0")
        message(FATAL_ERROR "failed to get cc by running get_cc: ${PROC_RES}")
    endif()

    message(STATUS "Cuda Compute Capability: ${CUDA_COMP_CAP}")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --generate-code arch=compute_${CUDA_COMP_CAP},code=sm_${CUDA_COMP_CAP}")

# add our benchmark binary and its dependencies
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
    # 'nvcc benchmark.cu' work without any parameters for RAD
    target_compile_definitions(cuda-group-by PRIVATE CUDA_GROUP_BY_CMAKE_BUILD)

    if(${CUDA_GROUP_BY_USE_CUB_SUBMODULE})
        target_include_directories(
            cuda-group-by 
            PRIVATE 
            "${CMAKE_CURRENT_SOURCE_DIR}/deps/cub"
        )
    endif()
