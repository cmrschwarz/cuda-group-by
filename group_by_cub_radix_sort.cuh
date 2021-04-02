#pragma once
#include "./deps/cub/cub/cub.cuh"
#include "cuda_group_by.cuh"

static void* cub_radix_sort_temp_storage;
static size_t cub_radix_sort_temp_storage_size;
static uint64_t* cub_radix_sort_sorted_group_col;
static uint64_t* cub_radix_sort_sorted_aggregate_col;

static size_t* cub_radix_sort_num_runs_dev_ptr;
__device__ size_t cub_radix_sort_num_runs;

void group_by_cub_radix_sort_init(size_t max_row_count)
{
    size_t reduce_max_storage;
    cub::DeviceReduce::ReduceByKey(
        NULL, reduce_max_storage, (uint64_t*)NULL, (uint64_t*)NULL,
        (uint64_t*)NULL, (uint64_t*)NULL, (size_t*)NULL, cub::Sum(),
        max_row_count);

    size_t sort_max_storage;
    cub::DeviceRadixSort::SortPairs(
        NULL, sort_max_storage, (uint64_t*)NULL, (uint64_t*)NULL,
        (uint64_t*)NULL, (uint64_t*)NULL, max_row_count);

    cub_radix_sort_temp_storage_size =
        std::max(reduce_max_storage, sort_max_storage);
    CUDA_TRY(cudaMalloc(
        &cub_radix_sort_temp_storage, cub_radix_sort_temp_storage_size));
    CUDA_TRY(cudaMalloc(
        &cub_radix_sort_sorted_group_col, max_row_count * sizeof(uint64_t)));
    CUDA_TRY(cudaMalloc(
        &cub_radix_sort_sorted_aggregate_col,
        max_row_count * sizeof(uint64_t)));
    CUDA_TRY(cudaGetSymbolAddress(
        (void**)&cub_radix_sort_num_runs_dev_ptr, cub_radix_sort_num_runs));
}

void group_by_cub_radix_sort_fin()
{
    CUDA_TRY(cudaFree(cub_radix_sort_sorted_aggregate_col));
    CUDA_TRY(cudaFree(cub_radix_sort_sorted_group_col));
    CUDA_TRY(cudaFree(cub_radix_sort_temp_storage));
}

static inline bool approach_cub_radix_sort_available(
    int group_bits, int row_count, int grid_size, int block_size,
    int stream_count)
{
    static bool grid_block_values_locked = false;
    static int locked_in_grid_size;
    static int locked_in_block_size;
    if (!grid_block_values_locked) {
        grid_block_values_locked = true;
        locked_in_grid_size = grid_size;
        locked_in_block_size = block_size;
        return true;
    }
    return grid_size == locked_in_grid_size &&
           block_size == locked_in_block_size && stream_count == 0;
}

template <int MAX_GROUP_BITS>
void group_by_cub_radix_sort(
    gpu_data* gd, int grid_size, int block_size, int stream_count,
    cudaStream_t* streams, cudaEvent_t* events, cudaEvent_t start_event,
    cudaEvent_t end_event)
{

    CUDA_TRY(cudaEventRecord(start_event));
    cub::DeviceRadixSort::SortPairs(
        cub_radix_sort_temp_storage, cub_radix_sort_temp_storage_size,
        gd->input.group_col, cub_radix_sort_sorted_group_col,
        gd->input.aggregate_col, cub_radix_sort_sorted_aggregate_col,
        gd->input.row_count);
    cub::DeviceReduce::ReduceByKey(
        cub_radix_sort_temp_storage, cub_radix_sort_temp_storage_size,
        cub_radix_sort_sorted_group_col, gd->output.group_col,
        cub_radix_sort_sorted_aggregate_col, gd->output.aggregate_col,
        cub_radix_sort_num_runs_dev_ptr, cub::Sum(), gd->input.row_count);
    cudaMemcpy(
        (void*)&gd->output.row_count, cub_radix_sort_num_runs_dev_ptr,
        sizeof(size_t), cudaMemcpyDeviceToHost);
    CUDA_TRY(cudaEventRecord(end_event));
    CUDA_TRY(cudaGetLastError());
}
