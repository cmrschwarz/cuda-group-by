#pragma once
#include "group_by_global_array.cuh"

// 8 bytes for the aggregate value + 1 bit (0.125 bytes) for the occurs flag
#define PER_THREAD_ARRAY_ENTRY_SIZE 8.125
#define PER_THREAD_ENTRY_BITS 3
#define PER_THREAD_ARRAY_SHARED_MEM_SUFFICIENT(GROUP_BITS, BLOCK_DIM)          \
    (CUDA_SHARED_MEM_PER_BLOCK >= ((((size_t)1) << (GROUP_BITS)) *             \
                                   PER_THREAD_ARRAY_ENTRY_SIZE * BLOCK_DIM))

GROUP_BY_GLOBAL_ARRAY_FORWARD_REQUIREMENTS(group_by_per_thread_array);

static inline bool approach_per_thread_array_available(
    int group_bits, int row_count, int grid_dim, int block_dim,
    int stream_count)
{
    if (!grid_dim || !block_dim) return false;
    return PER_THREAD_ARRAY_SHARED_MEM_SUFFICIENT(group_bits, block_dim);
}

template <int MAX_GROUP_BITS>
__global__ void kernel_per_thread_array_bank_optimized(
    db_table input, uint64_t* global_array, bool* global_occurance_array,
    int stream_count, int stream_idx)
{
    // ternary guards against template instantiations that would
    // cause ptxas error during compilations by requiring
    // too much shared memory even if these instantiations are never used
    constexpr bool SANE_INSTANCE =
        PER_THREAD_ARRAY_SHARED_MEM_SUFFICIENT(MAX_GROUP_BITS, CUDA_WARP_SIZE);
    constexpr int GROUP_COUNT = SANE_INSTANCE ? (int)1 << MAX_GROUP_BITS : 1;

    // for ex. on the 1080ti, 48 KiB of sm per block is 32Kib -> pow of 2
    // + 16 KiB overhang that more than suffice to store all the flag bits
    constexpr bool FLAGS_FIT_IN_NON_POW2_SHARED_MEM_OVERHANG =
        (CUDA_SHARED_MEM_PER_BLOCK -
         (((size_t)1) << CUDA_SHARED_MEM_BITS_PER_BLOCK)) >=
        (GROUP_COUNT * CUDA_MAX_BLOCK_SIZE / 8);

    constexpr int MAX_BLOCK_BITS =
        SANE_INSTANCE ? CUDA_SHARED_MEM_BITS_PER_BLOCK - MAX_GROUP_BITS -
                            PER_THREAD_ENTRY_BITS -
                            !FLAGS_FIT_IN_NON_POW2_SHARED_MEM_OVERHANG
                      : 1;
    constexpr int MAX_BLOCK_SIZE = 1 << MAX_BLOCK_BITS;
    constexpr int OCC_FLAGS_PACK_SIZE = 8 * sizeof(uint32_t);
    constexpr int OCC_FLAGS_ARRAY_SIZE = GROUP_COUNT > OCC_FLAGS_PACK_SIZE
                                             ? GROUP_COUNT / OCC_FLAGS_PACK_SIZE
                                             : 1;
    // we will always have sufficient occurance flag bits since
    // GROUP_COUNT must be a power of two -> smaller than or divisible
    // by OCC_FLAGS_PACK_SIZE
    static_assert(OCC_FLAGS_ARRAY_SIZE * OCC_FLAGS_PACK_SIZE >= GROUP_COUNT);

    const int ARR_STRIDE = blockDim.x;

    int tid = threadIdx.x + blockIdx.x * blockDim.x +
              stream_idx * blockDim.x * gridDim.x;

    int stride = blockDim.x * gridDim.x * stream_count;

    __shared__ uint32_t aggregate_low_bytes[GROUP_COUNT * MAX_BLOCK_SIZE];
    __shared__ uint32_t aggregate_high_bytes[GROUP_COUNT * MAX_BLOCK_SIZE];
    __shared__ uint32_t occurance_flags[OCC_FLAGS_ARRAY_SIZE * MAX_BLOCK_SIZE];

    for (int i = 0; i < GROUP_COUNT; i++) {
        aggregate_low_bytes[i * ARR_STRIDE + threadIdx.x] = 0;
        aggregate_high_bytes[i * ARR_STRIDE + threadIdx.x] = 0;
    }
    for (int i = 0; i < OCC_FLAGS_ARRAY_SIZE; i++) {
        occurance_flags[i * ARR_STRIDE + threadIdx.x] = 0;
    }

    for (size_t i = tid; i < input.row_count; i += stride) {
        uint64_t group = input.group_col[i];
        uint64_t aggregate = input.aggregate_col[i];
        int arr_idx = group * ARR_STRIDE + threadIdx.x;
        uint64_t stored_agg = aggregate_low_bytes[arr_idx];
        stored_agg |= ((uint64_t)aggregate_high_bytes[arr_idx]) << 32;
        stored_agg += aggregate;
        aggregate_low_bytes[arr_idx] = (uint32_t)stored_agg;
        aggregate_high_bytes[arr_idx] = (uint32_t)(stored_agg >> 32);

        occurance_flags
            [(group / OCC_FLAGS_PACK_SIZE) * ARR_STRIDE + threadIdx.x] |=
            (uint32_t)1 << (group % OCC_FLAGS_PACK_SIZE);
    }
    for (int i = 0; i < GROUP_COUNT; i += OCC_FLAGS_PACK_SIZE) {
        int end = i + OCC_FLAGS_PACK_SIZE <= GROUP_COUNT ? OCC_FLAGS_PACK_SIZE
                                                         : GROUP_COUNT - i;
        uint32_t occ_pack = occurance_flags
            [(i / OCC_FLAGS_PACK_SIZE) * ARR_STRIDE + threadIdx.x];
        for (int j = 0; j < end; j++) {
            if ((occ_pack >> j) & 0x1 == 1) {
                uint64_t group = i + j;
                size_t arr_idx = group * ARR_STRIDE + threadIdx.x;
                uint64_t aggregate = aggregate_low_bytes[arr_idx];
                aggregate |= ((uint64_t)aggregate_high_bytes[arr_idx]) << 32;
                global_array_insert<true>(
                    global_array, global_occurance_array, group, aggregate);
            }
        }
    }
}

template <int MAX_GROUP_BITS>
__global__ void kernel_per_thread_array(
    db_table input, uint64_t* global_array, bool* global_occurance_array,
    int stream_count, int stream_idx)
{
    // ternary guards against template instantiations that would
    // cause ptxas error during compilations by requiring
    // too much shared memory even if these instantiations are never used

    constexpr int GROUP_COUNT =
        PER_THREAD_ARRAY_SHARED_MEM_SUFFICIENT(MAX_GROUP_BITS, CUDA_WARP_SIZE)
            ? (int)1 << MAX_GROUP_BITS
            : 1;
    constexpr int OCC_FLAGS_PACK_SIZE = 8 * sizeof(uint32_t);
    constexpr int OCC_FLAGS_ARRAY_SIZE = GROUP_COUNT > OCC_FLAGS_PACK_SIZE
                                             ? GROUP_COUNT / OCC_FLAGS_PACK_SIZE
                                             : 1;
    // we will always have sufficient occurance flag bits since GROUP_COUNT
    // must be a power of two -> smaller than or divisible by 64
    static_assert(OCC_FLAGS_ARRAY_SIZE * OCC_FLAGS_PACK_SIZE >= GROUP_COUNT);
    uint64_t per_thread_array[GROUP_COUNT];
    uint32_t per_thread_occurance_flags[OCC_FLAGS_ARRAY_SIZE];

    int tid = threadIdx.x + blockIdx.x * blockDim.x +
              stream_idx * blockDim.x * gridDim.x;
    int stride = blockDim.x * gridDim.x * stream_count;

    for (int i = 0; i < GROUP_COUNT; i++) {
        per_thread_array[i] = 0;
    }
    for (int i = 0; i < OCC_FLAGS_ARRAY_SIZE; i++) {
        per_thread_occurance_flags[i] = 0;
    }

    for (size_t i = tid; i < input.row_count; i += stride) {
        uint64_t group = input.group_col[i];
        uint64_t aggregate = input.aggregate_col[i];
        per_thread_array[group] += aggregate;
        per_thread_occurance_flags[group / OCC_FLAGS_PACK_SIZE] |=
            (uint64_t)1 << (group % OCC_FLAGS_PACK_SIZE);
    }
    for (int i = 0; i < GROUP_COUNT; i += OCC_FLAGS_PACK_SIZE) {
        int end = i + OCC_FLAGS_PACK_SIZE < GROUP_COUNT ? OCC_FLAGS_PACK_SIZE
                                                        : GROUP_COUNT - i;
        uint32_t occ_pack = per_thread_occurance_flags[i / OCC_FLAGS_PACK_SIZE];
        for (int j = 0; j < end; j++) {
            if ((occ_pack >> j) & 0x1) {
                uint64_t group = i + j;
                global_array_insert<true>(
                    global_array, global_occurance_array, group,
                    per_thread_array[group]);
            }
        }
    }
}

template <int MAX_GROUP_BITS, bool BANK_OPTIMIZED = false>
void group_by_per_thread_array(
    gpu_data* gd, int grid_dim, int block_dim, int stream_count,
    cudaStream_t* streams, cudaEvent_t* events, cudaEvent_t start_event,
    cudaEvent_t end_event)
{
    CUDA_TRY(cudaEventRecord(start_event));
    // reset number of groups found
    size_t zero = 0;
    cudaMemcpyToSymbol(
        global_array_groups_found, &zero, sizeof(zero), 0,
        cudaMemcpyHostToDevice);
    // for stream_count 0 we use the default stream,
    // but thats actually still one stream not zero
    int actual_stream_count = stream_count ? stream_count : 1;
    for (int i = 0; i < actual_stream_count; i++) {
        cudaStream_t stream = stream_count ? streams[i] : 0;
        if (BANK_OPTIMIZED) {
            kernel_per_thread_array_bank_optimized<MAX_GROUP_BITS>
                <<<grid_dim, block_dim, 0, stream>>>(
                    gd->input, global_array, global_array_occurance_flags,
                    actual_stream_count, i);
        }
        else {
            kernel_per_thread_array<MAX_GROUP_BITS>
                <<<grid_dim, block_dim, 0, stream>>>(
                    gd->input, global_array, global_array_occurance_flags,
                    actual_stream_count, i);
        }
        // if we have only one stream there is no need for waiting events
        if (stream_count > 1) cudaEventRecord(events[i], stream);
    }
    group_by_global_array_writeout<MAX_GROUP_BITS>(
        gd, grid_dim, block_dim, stream_count, streams, events, start_event,
        end_event);
}
