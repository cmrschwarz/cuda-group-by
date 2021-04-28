#pragma once
#include "group_by_global_array.cuh"

// it's actually a bit less because the occurance_flag array storing ints
// but we just round up
#define SHARED_MEM_ARR_ENTRY_BITS 4
#define SHARED_MEM_ARR_MAX_GROUP_BITS                                          \
    (CUDA_SHARED_MEM_BITS_PER_BLOCK - SHARED_MEM_ARR_ENTRY_BITS)

static inline void group_by_shared_mem_array_init(size_t max_groups)
{
    group_by_global_array_init(max_groups);
}

static inline void group_by_shared_mem_array_fin()
{
    group_by_global_array_fin();
}

static inline bool approach_shared_mem_array_available(
    int group_bits, int row_count, int grid_dim, int block_dim,
    int stream_count)
{
    if (!grid_dim || !block_dim) return false;
    return group_bits <= SHARED_MEM_ARR_MAX_GROUP_BITS;
}

template <int MAX_GROUP_BITS>
__global__ void kernel_shared_mem_array(
    db_table input, uint64_t* global_array, bool* global_occurance_array,
    int stream_count, int stream_idx)
{
    // the ternary guards against template instantiations that would
    // cause ptxas error during compilations by requiring
    // too much shared memory even if these instantiations are never used
    constexpr size_t MAX_GROUPS =
        (MAX_GROUP_BITS <= SHARED_MEM_ARR_MAX_GROUP_BITS)
            ? (size_t)1 << MAX_GROUP_BITS
            : 1;
    __shared__ uint64_t shared_mem_array[MAX_GROUPS];
    __shared__ bool shared_mem_array_occurance[MAX_GROUPS];

    int tid = threadIdx.x + blockIdx.x * blockDim.x +
              stream_idx * blockDim.x * gridDim.x;
    int stride = blockDim.x * gridDim.x * stream_count;

    for (int i = threadIdx.x; i < MAX_GROUPS; i += blockDim.x) {
        shared_mem_array[i] = 0;
        shared_mem_array_occurance[i] = false;
    }
    __syncthreads();

    for (size_t i = tid; i < input.row_count; i += stride) {
        uint64_t group = input.group_col[i];
        uint64_t aggregate = input.aggregate_col[i];
        atomicAdd((cudaUInt64_t*)&shared_mem_array[group], aggregate);
        shared_mem_array_occurance[group] = true;
    }

    __syncthreads();
    for (uint64_t i = threadIdx.x; i < MAX_GROUPS; i += blockDim.x) {
        if (!shared_mem_array_occurance[i]) continue;
        global_array_insert<false>(
            global_array, global_occurance_array, i, shared_mem_array[i]);
    }
}

template <int MAX_GROUP_BITS>
__global__ void kernel_shared_mem_array_optimistic(
    db_table input, uint64_t* global_array, bool* global_occurance_array,
    int stream_count, int stream_idx)
{
    // the ternary guards against template instantiations that would
    // cause ptxas error during compilations by requiring
    // too much shared memory even if these instantiations are never used
    constexpr size_t MAX_GROUPS =
        (MAX_GROUP_BITS <= SHARED_MEM_ARR_MAX_GROUP_BITS)
            ? (size_t)1 << MAX_GROUP_BITS
            : 1;

    __shared__ uint64_t shared_mem_array[MAX_GROUPS];
    // datatypes smaller than int have no atomicOr :/,
    // otherwise bool would suffice
    __shared__ int shared_mem_array_occurance[MAX_GROUPS];
    __shared__ int unfound_groups;

    if (threadIdx.x == 0) {
        unfound_groups = 0;
    }

    int tid = threadIdx.x + blockIdx.x * blockDim.x +
              stream_idx * blockDim.x * gridDim.x;
    int stride = blockDim.x * gridDim.x * stream_count;

    for (int i = threadIdx.x; i < MAX_GROUPS; i += blockDim.x) {
        shared_mem_array[i] = 0;
        shared_mem_array_occurance[i] = 0;
    }
    __syncthreads();
    size_t i = tid;
    for (; i < input.row_count; i += stride) {
        uint64_t group = input.group_col[i];
        uint64_t aggregate = input.aggregate_col[i];
        atomicAdd((cudaUInt64_t*)&shared_mem_array[group], aggregate);
        if (atomicOr(&shared_mem_array_occurance[group], 1) == 0) {
            atomicAdd((int*)&unfound_groups, 1);
        }
        if (tid - threadIdx.x & 0xFF == 0) {
            if (unfound_groups == MAX_GROUPS) break;
        }
    }
    for (; i < input.row_count; i += stride) {
        uint64_t group = input.group_col[i];
        uint64_t aggregate = input.aggregate_col[i];
        atomicAdd((cudaUInt64_t*)&shared_mem_array[group], aggregate);
    }
    __syncthreads();
    if (unfound_groups == MAX_GROUPS) {
        for (int i = threadIdx.x; i < MAX_GROUPS; i += blockDim.x) {
            global_array_insert<false>(
                global_array, global_occurance_array, i, shared_mem_array[i]);
        }
    }
    else {
        for (int i = threadIdx.x; i < MAX_GROUPS; i += blockDim.x) {
            if (!shared_mem_array_occurance[i]) continue;
            global_array_insert<false>(
                global_array, global_occurance_array, i, shared_mem_array[i]);
        }
    }
}

template <int MAX_GROUP_BITS, bool OPTIMISTIC>
void group_by_shared_mem_array(
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
        if (OPTIMISTIC) {
            kernel_shared_mem_array_optimistic<MAX_GROUP_BITS>
                <<<grid_dim, block_dim, 0, stream>>>(
                    gd->input, global_array, global_array_occurance_flags,
                    actual_stream_count, i);
        }
        else {
            kernel_shared_mem_array<MAX_GROUP_BITS>
                <<<grid_dim, block_dim, 0, stream>>>(
                    gd->input, global_array, global_array_occurance_flags,
                    actual_stream_count, i);
        }
        // if we have only one stream there is no need for waiting events
        if (stream_count > 1) cudaEventRecord(events[i], stream);
    }
    for (int i = 0; i < actual_stream_count; i++) {
        cudaStream_t stream = stream_count ? streams[i] : 0;
        if (stream_count > 1) {
            // every write out kernel needs to wait on every fill kernel
            for (int j = 0; j < stream_count; j++) {
                // the stream doesn't need to wait on itself
                if (j == i) continue;
                cudaStreamWaitEvent(stream, events[j], 0);
            }
        }
        kernel_write_out_global_array<MAX_GROUP_BITS>
            <<<grid_dim, block_dim, 0, stream>>>(
                gd->output, global_array, global_array_occurance_flags,
                actual_stream_count, i);
    }
    CUDA_TRY(cudaEventRecord(end_event));
    CUDA_TRY(cudaGetLastError());
    // read out number of groups found
    // this waits for the kernels to complete since it's in the default stream
    cudaMemcpyFromSymbol(
        &gd->output.row_count, global_array_groups_found, sizeof(size_t), 0,
        cudaMemcpyDeviceToHost);
}
