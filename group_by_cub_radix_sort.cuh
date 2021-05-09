#pragma once

#include "cuda_group_by.cuh"

// Disable the "dynamic initialization in unreachable code" warning message
// thrown inside from inside cub
#pragma diag_suppress initialization_not_reachable

#ifdef CUDA_GROUP_BY_CMAKE_BUILD
#    include <cub/cub.cuh>
#else
// always use the submodule version when we are not building with cmake
// and don't have a proper include path setup
#    include "./deps/cub/cub/cub.cuh"
#endif

static void* cub_radix_sort_temp_storage;
static size_t cub_radix_sort_temp_storage_size;
static uint64_t* cub_radix_sort_sorted_group_col;
static uint64_t* cub_radix_sort_sorted_aggregate_col;

static size_t* cub_radix_sort_num_runs_dev_ptr;
__device__ size_t cub_radix_sort_num_runs;

static inline void group_by_cub_radix_sort_get_mem_requirements(
    size_t max_groups, size_t max_rows, size_t* zeroed, size_t* uninitialized)
{
    size_t reduce_max_storage = 0;
    cub::DeviceReduce::ReduceByKey(
        NULL, reduce_max_storage, (uint64_t*)NULL, (uint64_t*)NULL,
        (uint64_t*)NULL, (uint64_t*)NULL, (size_t*)NULL, cub::Sum(), max_rows);

    size_t sort_max_storage = 0;
    cub::DeviceRadixSort::SortPairs(
        NULL, sort_max_storage, (uint64_t*)NULL, (uint64_t*)NULL,
        (uint64_t*)NULL, (uint64_t*)NULL, max_rows);

    cub_radix_sort_temp_storage_size = ceil_to_mult(
        std::max(reduce_max_storage, sort_max_storage),
        CUDA_MAX_CACHE_LINE_SIZE);
    *zeroed = 0;
    *uninitialized =
        cub_radix_sort_temp_storage_size + 2 * max_rows * sizeof(uint64_t);
}

void group_by_cub_radix_sort_init(
    size_t max_groups, size_t max_rows, void* zeroed_mem,
    void* uninitialized_mem)
{
    cub_radix_sort_temp_storage = uninitialized_mem;
    cub_radix_sort_sorted_group_col =
        (size_t*)ptradd(uninitialized_mem, cub_radix_sort_temp_storage_size);
    cub_radix_sort_sorted_aggregate_col =
        cub_radix_sort_sorted_group_col + max_rows;
    CUDA_TRY(cudaGetSymbolAddress(
        (void**)&cub_radix_sort_num_runs_dev_ptr, cub_radix_sort_num_runs));
}

void group_by_cub_radix_sort_fin()
{
}

static inline bool approach_cub_radix_sort_available(
    int group_bits, size_t row_count, int grid_dim, int block_dim,
    int stream_count)
{
    return grid_dim == 0 && block_dim == 0 && stream_count == 0;
}

template <int MAX_GROUP_BITS>
void group_by_cub_radix_sort(
    gpu_data* gd, int grid_dim, int block_dim, int stream_count,
    cudaStream_t* streams, cudaEvent_t* events, cudaEvent_t start_event,
    cudaEvent_t end_event)
{

    CUDA_TRY(cudaEventRecord(start_event));
    int end_bit;

#if GROUP_COUNT_EQUALS_GROUP_MAX_VAL
    end_bit = MAX_GROUP_BITS;
#else
    end_bit = sizeof(uint64_t) * 8;
#endif
    size_t mem;
    cub::DeviceRadixSort::SortPairs(
        NULL, mem, (uint64_t*)NULL, (uint64_t*)NULL, (uint64_t*)NULL,
        (uint64_t*)NULL, gd->input.row_count);
    RELASE_ASSERT(mem <= cub_radix_sort_temp_storage_size);
    cub::DeviceRadixSort::SortPairs(
        cub_radix_sort_temp_storage, cub_radix_sort_temp_storage_size,
        gd->input.group_col, cub_radix_sort_sorted_group_col,
        gd->input.aggregate_col, cub_radix_sort_sorted_aggregate_col,
        gd->input.row_count, 0, end_bit);
    cub::DeviceReduce::ReduceByKey(
        cub_radix_sort_temp_storage, cub_radix_sort_temp_storage_size,
        cub_radix_sort_sorted_group_col, gd->output.group_col,
        cub_radix_sort_sorted_aggregate_col, gd->output.aggregate_col,
        cub_radix_sort_num_runs_dev_ptr, cub::Sum(), gd->input.row_count);
    CUDA_TRY(cudaEventRecord(end_event));
    CUDA_TRY(cudaGetLastError());
    cudaMemcpy(
        (void*)&gd->output.row_count, cub_radix_sort_num_runs_dev_ptr,
        sizeof(size_t), cudaMemcpyDeviceToHost);
}
