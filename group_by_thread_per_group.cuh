#pragma once
#include "cuda_group_by.cuh"

#define TPG_MAX_GROUPS 512
#define TPG_EMPTY_GROUP_VALUE 0

__device__ size_t tpg_group_count;
__device__ bool tpg_empty_group_used;
__device__ uint64_t tpg_empty_group_aggregate;
__device__ uint64_t tpg_groups[TPG_MAX_GROUPS];
__device__ uint64_t tpg_aggregates[TPG_MAX_GROUPS];

uint64_t* tpg_groups_dev_ptr;
uint64_t* tpg_aggregates_dev_ptr;

static inline void group_by_thread_per_group_init()
{
    cudaGetSymbolAddress((void**)&tpg_groups_dev_ptr, "tpg_groups");
    cudaGetSymbolAddress((void**)&tpg_aggregates_dev_ptr, "tpg_aggregates");

    // same reasoning as in hashtable_init
    assert(TPG_EMPTY_GROUP_VALUE == 0);
    cudaMemset(tpg_groups_dev_ptr, 0, sizeof(tpg_groups));
    cudaMemset(tpg_aggregates_dev_ptr, 0, sizeof(tpg_groups));
    bool f = false;
    cudaMemcpyToSymbol(
        tpg_empty_group_used, &f, sizeof(f), 0, cudaMemcpyHostToDevice);
    uint64_t zero = 0;
    cudaMemcpyToSymbol(
        tpg_empty_group_used, &zero, sizeof(zero), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(
        tpg_empty_group_aggregate, &zero, sizeof(zero), 0,
        cudaMemcpyHostToDevice);
}

static inline void group_by_thread_per_group_fin()
{
}

static inline bool approach_thread_per_group_available(
    int group_bits, int row_count, int grid_size, int block_size,
    int stream_count)
{
    return (1 << group_bits) < TPG_MAX_GROUPS;
}

template <int MAX_GROUP_BITS>
__global__ void
kernel_thread_per_group(gpu_data gd, int stream_idx, int stream_count)
{
    constexpr size_t GROUP_COUNT = (1 << MAX_GROUP_BITS);
    __shared__ uint64_t groups[GROUP_COUNT];
    __shared__ uint64_t aggregates[GROUP_COUNT];

    // the availability function guarantees us that this is divisible
    const int groups_per_thread = blockDim.x / GROUP_COUNT;

    uint64_t thread_groups[groups_per_thread];
    uint64_t thread_aggregates[groups_per_thread];
    // TODO...
}

template <int MAX_GROUP_BITS>
void group_by_thread_per_group(
    gpu_data* gd, int grid_size, int block_size, int stream_count,
    cudaStream_t* streams, cudaEvent_t* events, cudaEvent_t start_event,
    cudaEvent_t end_event)
{
    assert(approach_thread_per_group_available(
        MAX_GROUP_BITS, gd->input.row_count, grid_size, block_size,
        stream_count));
    constexpr size_t MAX_GROUPS = 1 << MAX_GROUP_BITS;
    int actual_stream_count = stream_count ? stream_count : 1;
    for (int i = 0; i < actual_stream_count; i++) {
        cudaStream_t stream = stream_count ? streams[i] : 0;
        kernel_thread_per_group<MAX_GROUP_BITS>
            <<<block_size, grid_size, 0, stream>>>(*gd, actual_stream_count, i);
    }
}
