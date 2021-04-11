#pragma once
#include "group_by_hashtable.cuh"

#define PER_THREAD_HT_MAX_BITS 7
#define PER_THREAD_HT_EMPTY_GROUP_VAL ((uint64_t)0)
#define PER_THREAD_HT_OVERSIZE_BITS 1

struct per_thread_ht_entry {
    uint64_t group;
    uint64_t aggregate;
};

static inline void group_by_per_thread_hashtable_init(size_t max_groups)
{
    group_by_hashtable_init(max_groups);
}

static inline void group_by_per_thread_hashtable_fin()
{
    group_by_hashtable_fin();
}

static inline bool approach_per_thread_hashtable_available(
    int group_bits, int row_count, int grid_dim, int block_dim,
    int stream_count)
{
    if (!grid_dim || !block_dim) return false;
    return group_bits <= (PER_THREAD_HT_MAX_BITS - PER_THREAD_HT_OVERSIZE_BITS);
}

template <int MAX_GROUP_BITS>
__global__ void kernel_per_thread_hashtable(
    db_table input, group_ht_entry<false>* hashtable, int stream_count,
    int stream_idx)
{
    // the ternary guards against template instantiations that would
    // cause ptxas error during compilations by requiring
    // too much shared memory even if these instantiations are never used
    constexpr size_t PER_THREAD_HT_CAPACITY =
        (MAX_GROUP_BITS + PER_THREAD_HT_OVERSIZE_BITS > PER_THREAD_HT_MAX_BITS)
            ? 1
            : (size_t)1 << (MAX_GROUP_BITS + PER_THREAD_HT_MAX_BITS);
    constexpr size_t PER_THREAD_HT_MASK = PER_THREAD_HT_CAPACITY - 1;

    bool empty_group_used = false;
    uint64_t empty_group_aggregate = 0;
    per_thread_ht_entry per_thread_ht[PER_THREAD_HT_CAPACITY];

    int tid = threadIdx.x + blockIdx.x * blockDim.x +
              stream_idx * blockDim.x * gridDim.x;
    int stride = blockDim.x * gridDim.x * stream_count;

    for (int i = 0; i < PER_THREAD_HT_CAPACITY; i++) {
        per_thread_ht[i].group = PER_THREAD_HT_MAX_BITS;
        per_thread_ht[i].aggregate = 0;
    }

    for (size_t i = tid; i < input.row_count; i += stride) {
        uint64_t group = input.group_col[i];
        uint64_t aggregate = input.aggregate_col[i];
        if (group == PER_THREAD_HT_MAX_BITS) {
            empty_group_used = true;
            empty_group_aggregate += aggregate;
            continue;
        }
        per_thread_ht_entry* hte = &per_thread_ht[group & PER_THREAD_HT_MASK];
        while (true) {
            if (hte->group == group) break;
            if (hte->group == PER_THREAD_HT_MAX_BITS) {
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
            hashtable, PER_THREAD_HT_MAX_BITS, empty_group_aggregate);
    }
    for (int i = 0; i < PER_THREAD_HT_CAPACITY; i++) {
        if (per_thread_ht[i].group != PER_THREAD_HT_MAX_BITS) {
            group_ht_insert<MAX_GROUP_BITS, false>(
                hashtable, per_thread_ht[i].group, per_thread_ht[i].aggregate);
        }
    }
}

template <int MAX_GROUP_BITS>
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
        kernel_per_thread_hashtable<MAX_GROUP_BITS>
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
