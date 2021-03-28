#pragma once
#include "cuda_group_by.cuh"

#define TPG_MAX_GROUPS 512

bool approach_thread_per_group_available(
    int group_bits, int row_count, int grid_size, int block_size,
    int stream_count)
{
    return (1 << group_bits) < TPG_MAX_GROUPS;
}

template <int MAX_GROUP_BITS> void group_by_thread_per_group()
{
    constexpr size_t MAX_GROUPS = 1 << MAX_GROUP_BITS;
    static_assert(
        GPU_MAX_THREADS_PER_BLOCK >= MAX_GROUPS,
        "strategy not available, MAX_GROUPS exceeds "
        "GPU_MAX_THREADS_PER_BLOCK");
}
