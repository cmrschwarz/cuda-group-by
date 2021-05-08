#pragma once
#include "group_by_hashtable.cuh"

#define PER_THREAD_HT_EMPTY_GROUP_VAL ((uint64_t)0)
#define PER_THREAD_HT_OVERSIZE_BITS 1
// 16 bytes -> 4 bits
#define PER_THREAD_HT_ENTRY_BITS 4

struct per_thread_ht_entry {
    uint64_t group;
    uint64_t aggregate;
};

GROUP_BY_HASHTABLE_FORWARD_REQUIREMENTS(group_by_per_thread_hashtable)

static inline bool approach_per_thread_hashtable_available(
    int group_bits, int row_count, int grid_dim, int block_dim,
    int stream_count)
{
    if (!grid_dim || !block_dim) return false;
    int block_dim_bits = (int)log2(block_dim);
    if (block_dim_bits + group_bits + PER_THREAD_HT_OVERSIZE_BITS +
            PER_THREAD_HT_ENTRY_BITS >
        CUDA_SHARED_MEM_BITS_PER_BLOCK) {
        return false;
    }
    return true;
}

template <int MAX_GROUP_BITS>
__global__ void kernel_per_thread_hashtable_bank_optimized(
    db_table input, group_ht_entry<>* hashtable, int stream_count,
    int stream_idx)
{
    // the ternaries guards against template instantiations that would
    // cause ptxas error during compilations by requiring
    // too much shared memory even if these instantiations are never used
    constexpr bool SANE_INSTANCE =
        (CUDA_SHARED_MEM_BITS_PER_BLOCK - MAX_GROUP_BITS -
         PER_THREAD_HT_OVERSIZE_BITS - CUDA_WARP_SIZE_BITS -
         PER_THREAD_HT_ENTRY_BITS) >= 0;

    constexpr int MAX_BLOCK_BITS =
        SANE_INSTANCE
            ? CUDA_SHARED_MEM_BITS_PER_BLOCK - MAX_GROUP_BITS -
                  PER_THREAD_HT_OVERSIZE_BITS - PER_THREAD_HT_ENTRY_BITS
            : 1;
    constexpr int MAX_BLOCK_SIZE = 1 << MAX_BLOCK_BITS;

    constexpr int PER_THREAD_HT_CAPACITY =
        SANE_INSTANCE ? (int)1 << (MAX_GROUP_BITS + PER_THREAD_HT_OVERSIZE_BITS)
                      : 1;
    constexpr int PER_THREAD_HT_MASK = PER_THREAD_HT_CAPACITY - 1;

    // shorter names for very common vars
    const int HT_STRIDE = blockDim.x;
    int tid = threadIdx.x;

    bool empty_group_used = false;
    uint64_t empty_group_aggregate = 0;
    __shared__ uint32_t
        group_high_bytes[PER_THREAD_HT_CAPACITY * MAX_BLOCK_SIZE];
    __shared__ uint32_t
        group_low_bytes[PER_THREAD_HT_CAPACITY * MAX_BLOCK_SIZE];
    __shared__ uint32_t
        aggregate_high_bytes[PER_THREAD_HT_CAPACITY * MAX_BLOCK_SIZE];
    __shared__ uint32_t
        aggregate_low_bytes[PER_THREAD_HT_CAPACITY * MAX_BLOCK_SIZE];
    int tid_flat = threadIdx.x + blockIdx.x * blockDim.x +
                   stream_idx * blockDim.x * gridDim.x;
    int stride = blockDim.x * gridDim.x * stream_count;

    for (int i = 0; i < PER_THREAD_HT_CAPACITY; i++) {
        group_low_bytes[i * HT_STRIDE + tid] =
            (uint32_t)PER_THREAD_HT_EMPTY_GROUP_VAL;
        group_high_bytes[i * HT_STRIDE + tid] =
            (uint32_t)(PER_THREAD_HT_EMPTY_GROUP_VAL >> 32);
        aggregate_low_bytes[i * HT_STRIDE + tid] = 0;
        aggregate_high_bytes[i * HT_STRIDE + tid] = 0;
    }
    for (size_t i = tid_flat; i < input.row_count; i += stride) {
        uint64_t group = input.group_col[i];
        uint64_t aggregate = input.aggregate_col[i];
        if (group == PER_THREAD_HT_EMPTY_GROUP_VAL) {
            empty_group_used = true;
            empty_group_aggregate += aggregate;
            continue;
        }
        int idx = group & PER_THREAD_HT_MASK;
        while (true) {
            uint64_t cmp_group = group_low_bytes[idx * HT_STRIDE + tid];
            cmp_group |=
                (((uint64_t)group_high_bytes[idx * HT_STRIDE + tid]) << 32);
            if (group == cmp_group) {
                break;
            }
            if (cmp_group == PER_THREAD_HT_EMPTY_GROUP_VAL) {
                group_low_bytes[idx * HT_STRIDE + tid] = (uint32_t)group;
                group_high_bytes[idx * HT_STRIDE + tid] =
                    (uint32_t)(group >> 32);
                break;
            }
            if (idx != PER_THREAD_HT_CAPACITY - 1) {
                idx++;
            }
            else {
                idx = 0;
            }
        }
        uint64_t stored_agg = aggregate_low_bytes[idx * HT_STRIDE + tid];
        stored_agg |= ((uint64_t)aggregate_high_bytes[idx * HT_STRIDE + tid])
                      << 32;
        stored_agg += aggregate;
        aggregate_low_bytes[idx * HT_STRIDE + tid] = (uint32_t)stored_agg;
        aggregate_high_bytes[idx * HT_STRIDE + tid] =
            (uint32_t)(stored_agg >> 32);
    }
    if (empty_group_used) {
        group_ht_insert<MAX_GROUP_BITS, false>(
            hashtable, PER_THREAD_HT_EMPTY_GROUP_VAL, empty_group_aggregate);
    }
    for (int i = 0; i < PER_THREAD_HT_CAPACITY; i++) {
        uint64_t group = group_low_bytes[i * HT_STRIDE + tid];
        group |= ((uint64_t)group_high_bytes[i * HT_STRIDE + tid]) << 32;
        if (group != PER_THREAD_HT_EMPTY_GROUP_VAL) {
            uint64_t aggregate = aggregate_low_bytes[i * HT_STRIDE + tid];
            aggregate |= ((uint64_t)aggregate_high_bytes[i * HT_STRIDE + tid])
                         << 32;
            group_ht_insert<MAX_GROUP_BITS, false>(hashtable, group, aggregate);
        }
    }
}

template <int MAX_GROUP_BITS>
__global__ void kernel_per_thread_hashtable(
    db_table input, group_ht_entry<>* hashtable, int stream_count,
    int stream_idx)
{
    // guards against template instantiations that would
    // cause ptxas error during compilations by requiring
    // too much shared memory even if these instantiations are never used
    constexpr bool SANE_INSTANCE =
        (CUDA_SHARED_MEM_BITS_PER_BLOCK - MAX_GROUP_BITS -
         PER_THREAD_HT_OVERSIZE_BITS - CUDA_WARP_SIZE_BITS -
         PER_THREAD_HT_ENTRY_BITS) >= 0;

    constexpr int PER_THREAD_HT_CAPACITY =
        SANE_INSTANCE ? (int)1 << (MAX_GROUP_BITS + PER_THREAD_HT_OVERSIZE_BITS)
                      : 1;
    constexpr int PER_THREAD_HT_MASK = PER_THREAD_HT_CAPACITY - 1;

    bool empty_group_used = false;
    uint64_t empty_group_aggregate = 0;
    per_thread_ht_entry per_thread_ht[PER_THREAD_HT_CAPACITY];

    int tid = threadIdx.x + blockIdx.x * blockDim.x +
              stream_idx * blockDim.x * gridDim.x;
    int stride = blockDim.x * gridDim.x * stream_count;

    for (int i = 0; i < PER_THREAD_HT_CAPACITY; i++) {
        per_thread_ht[i].group = PER_THREAD_HT_EMPTY_GROUP_VAL;
        per_thread_ht[i].aggregate = 0;
    }

    for (size_t i = tid; i < input.row_count; i += stride) {
        uint64_t group = input.group_col[i];
        uint64_t aggregate = input.aggregate_col[i];
        if (group == PER_THREAD_HT_EMPTY_GROUP_VAL) {
            empty_group_used = true;
            empty_group_aggregate += aggregate;
            continue;
        }
        per_thread_ht_entry* hte = &per_thread_ht[group & PER_THREAD_HT_MASK];
        while (true) {
            if (hte->group == group) break;
            if (hte->group == PER_THREAD_HT_EMPTY_GROUP_VAL) {
                hte->group = group;
                break;
            }
            if (hte != &per_thread_ht[PER_THREAD_HT_CAPACITY - 1]) {
                hte++;
            }
            else {
                hte = &per_thread_ht[0];
            }
        }
        hte->aggregate += aggregate;
    }
    if (empty_group_used) {
        group_ht_insert<MAX_GROUP_BITS, false>(
            hashtable, PER_THREAD_HT_EMPTY_GROUP_VAL, empty_group_aggregate);
    }
    for (int i = 0; i < PER_THREAD_HT_CAPACITY; i++) {
        if (per_thread_ht[i].group != PER_THREAD_HT_EMPTY_GROUP_VAL) {
            group_ht_insert<MAX_GROUP_BITS, false>(
                hashtable, per_thread_ht[i].group, per_thread_ht[i].aggregate);
        }
    }
}

template <int MAX_GROUP_BITS, bool BANK_OPTIMIZED = false>
void group_by_per_thread_hashtable(
    gpu_data* gd, int grid_dim, int block_dim, int stream_count,
    cudaStream_t* streams, cudaEvent_t* events, cudaEvent_t start_event,
    cudaEvent_t end_event)
{
    CUDA_TRY(cudaEventRecord(start_event));
    // reset number of groups found
    size_t zero = 0;
    cudaMemcpyToSymbol(
        group_ht_groups_found, &zero, sizeof(zero), 0, cudaMemcpyHostToDevice);
    // for stream_count 0 we use the default stream,
    // but thats actually still one stream not zero
    int actual_stream_count = stream_count ? stream_count : 1;
    for (int i = 0; i < actual_stream_count; i++) {
        cudaStream_t stream = stream_count ? streams[i] : 0;
        if (BANK_OPTIMIZED) {
            kernel_per_thread_hashtable_bank_optimized<MAX_GROUP_BITS>
                <<<grid_dim, block_dim, 0, stream>>>(
                    gd->input, group_ht_entry<>::table, actual_stream_count, i);
        }
        else {
            kernel_per_thread_hashtable<MAX_GROUP_BITS>
                <<<grid_dim, block_dim, 0, stream>>>(
                    gd->input, group_ht_entry<>::table, actual_stream_count, i);
        }
        // if we have only one stream there is no need for waiting events
        if (stream_count > 1) cudaEventRecord(events[i], stream);
    }
    group_by_hashtable_writeout<MAX_GROUP_BITS>(
        gd, grid_dim, block_dim, stream_count, streams, events, start_event,
        end_event);
}
