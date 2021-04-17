#pragma once
#include "group_by_hashtable.cuh"

#define SHARED_MEM_HT_ENTRY_BITS 4
#define SHARED_MEM_HT_EMPTY_GROUP_VAL ((uint64_t)0)
#define SHARED_MEM_HT_OVERSIZE_BITS 1
#define SHARED_MEM_HT_MAX_GROUP_BITS                                           \
    (CUDA_SHARED_MEM_BITS_PER_BLOCK - SHARED_MEM_HT_OVERSIZE_BITS -            \
     SHARED_MEM_HT_ENTRY_BITS)

struct shared_mem_ht_entry {
    uint64_t group;
    uint64_t aggregate;
};

static inline void group_by_shared_mem_hashtable_init(size_t max_groups)
{
    group_by_hashtable_init(max_groups);
}

static inline void group_by_shared_mem_hashtable_fin()
{
    group_by_hashtable_fin();
}

static inline bool approach_shared_mem_hashtable_available(
    int group_bits, int row_count, int grid_dim, int block_dim,
    int stream_count)
{
    if (!grid_dim || !block_dim) return false;
    return group_bits <= SHARED_MEM_HT_MAX_GROUP_BITS;
}

template <int MAX_GROUP_BITS>
__global__ void kernel_shared_mem_ht(
    db_table input, group_ht_entry<false>* hashtable, int stream_count,
    int stream_idx)
{
    // the ternary guards against template instantiations that would
    // cause ptxas error during compilations by requiring
    // too much shared memory even if these instantiations are never used
    constexpr size_t SHARED_MEM_HT_CAPACITY =
        (MAX_GROUP_BITS <= SHARED_MEM_HT_MAX_GROUP_BITS)
            ? (size_t)1 << (MAX_GROUP_BITS + SHARED_MEM_HT_OVERSIZE_BITS)
            : 1;
    constexpr size_t SHARED_MEM_HT_MASK = SHARED_MEM_HT_CAPACITY - 1;

    __shared__ bool empty_group_used;
    __shared__ uint64_t empty_group_aggregate;
    __shared__ shared_mem_ht_entry shared_mem_ht[SHARED_MEM_HT_CAPACITY];

    int tid = threadIdx.x + blockIdx.x * blockDim.x +
              stream_idx * blockDim.x * gridDim.x;
    int stride = blockDim.x * gridDim.x * stream_count;

    if (threadIdx.x == 0) {
        empty_group_used = false;
        empty_group_aggregate = 0;
    }

    for (int i = threadIdx.x; i < SHARED_MEM_HT_CAPACITY; i += blockDim.x) {
        shared_mem_ht[i].group = SHARED_MEM_HT_EMPTY_GROUP_VAL;
        shared_mem_ht[i].aggregate = 0;
    }
    __syncthreads();

    for (size_t i = tid; i < input.row_count; i += stride) {
        uint64_t group = input.group_col[i];
        uint64_t aggregate = input.aggregate_col[i];
        if (group == SHARED_MEM_HT_EMPTY_GROUP_VAL) {
            empty_group_used = true;
            atomicAdd((cudaUInt64_t*)&empty_group_aggregate, aggregate);
            continue;
        }
        shared_mem_ht_entry* hte = &shared_mem_ht[group & SHARED_MEM_HT_MASK];
        while (true) {
            if (hte->group == group) break;
            if (hte->group == SHARED_MEM_HT_EMPTY_GROUP_VAL) {
                uint64_t found = atomicCAS(
                    (cudaUInt64_t*)&hte->group, SHARED_MEM_HT_EMPTY_GROUP_VAL,
                    group);
                if (found == SHARED_MEM_HT_EMPTY_GROUP_VAL || found == group) {
                    break;
                }
            }
            if (hte != &shared_mem_ht[SHARED_MEM_HT_CAPACITY - 1]) {
                hte++;
            }
            else {
                hte = &shared_mem_ht[0];
            }
        }
        atomicAdd((cudaUInt64_t*)&hte->aggregate, aggregate);
    }

    __syncthreads();
    if (threadIdx.x == 0 && empty_group_used) {
        group_ht_insert<MAX_GROUP_BITS, false>(
            hashtable, SHARED_MEM_HT_EMPTY_GROUP_VAL, empty_group_aggregate);
    }
    for (int i = threadIdx.x; i < SHARED_MEM_HT_CAPACITY; i += blockDim.x) {
        if (shared_mem_ht[i].group != SHARED_MEM_HT_EMPTY_GROUP_VAL) {
            group_ht_insert<MAX_GROUP_BITS, false>(
                hashtable, shared_mem_ht[i].group, shared_mem_ht[i].aggregate);
        }
    }
}

template <int MAX_GROUP_BITS>
void group_by_shared_mem_hashtable(
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
        kernel_shared_mem_ht<MAX_GROUP_BITS>
            <<<grid_dim, block_dim, 0, stream>>>(
                gd->input, group_ht_entry<false>::table, actual_stream_count,
                i);
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
        kernel_write_out_group_ht<MAX_GROUP_BITS, false>
            <<<grid_dim, block_dim, 0, stream>>>(
                gd->output, group_ht_entry<false>::table, actual_stream_count,
                i);
    }
    CUDA_TRY(cudaEventRecord(end_event));
    CUDA_TRY(cudaGetLastError());
    // read out number of groups found
    // this waits for the kernels to complete since it's in the default stream
    cudaMemcpyFromSymbol(
        &gd->output.row_count, group_ht_groups_found, sizeof(size_t), 0,
        cudaMemcpyDeviceToHost);
}
