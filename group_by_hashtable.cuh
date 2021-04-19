#pragma once
#include <type_traits>
#include "cuda_group_by.cuh"

// number of bits to 'oversize' the hashtable by relative to MAX_GROUP_BITS
// we use a shift instead of a multiplier to make sure we keep the nice power of
// two properties
#define GROUP_HT_SIZE_SHIFT 1
#define GROUP_HT_EMPTY_VALUE 0

__device__ size_t group_ht_groups_found = 0;
__device__ bool group_ht_empty_group_used = false;
// since other approaches reuse the hashtable from this we have to avoid
// multi initialization
static bool group_ht_initialized = false;

struct group_ht_entry_base {
    uint64_t group;
    uint64_t aggregate;
};

template <bool EAGER_OUT_IDX>
struct group_ht_entry : public group_ht_entry_base {
    static group_ht_entry<EAGER_OUT_IDX>* table;
    __device__ void eager_inc_out_idx()
    {
        // does nothing, since this is the non eager specialization of this
    }
    // ONLY CALL THIS ONCE!
    __device__ size_t aquire_out_idx()
    {
        return atomicAdd((cudaUInt64_t*)&group_ht_groups_found, 1);
    }
};

template <> struct group_ht_entry<true> : public group_ht_entry_base {
    static group_ht_entry<true>* table;
    size_t output_idx;
    __device__ void eager_inc_out_idx()
    {
        output_idx = atomicAdd((cudaUInt64_t*)&group_ht_groups_found, 1);
    }
    __device__ size_t aquire_out_idx()
    {
        return output_idx;
    }
};
template <> group_ht_entry<false>* group_ht_entry<false>::table = nullptr;
group_ht_entry<true>* group_ht_entry<true>::table = nullptr;

static inline void group_by_hashtable_init(size_t max_groups)
{
    if (group_ht_initialized) return;
    group_ht_initialized = true;
    // if we want to use a different empty value (0 is probably a common
    // group) we will need to properly initialize this, since the memset
    // will no longer suffice
    assert(GROUP_HT_EMPTY_VALUE == 0);

    const size_t ht_size_eager =
        (max_groups << GROUP_HT_SIZE_SHIFT) * sizeof(group_ht_entry<true>);
    CUDA_TRY(cudaMalloc(&group_ht_entry<true>::table, ht_size_eager));
    CUDA_TRY(cudaMemset(group_ht_entry<true>::table, 0, ht_size_eager));

    const size_t ht_size_lazy =
        (max_groups << GROUP_HT_SIZE_SHIFT) * sizeof(group_ht_entry<false>);
    CUDA_TRY(cudaMalloc(&group_ht_entry<false>::table, ht_size_lazy));
    CUDA_TRY(cudaMemset(group_ht_entry<false>::table, 0, ht_size_lazy));
}
static inline void group_by_hashtable_fin()
{
    if (!group_ht_initialized) return;
    group_ht_initialized = false;
    CUDA_TRY(cudaFree(group_ht_entry<false>::table));
    CUDA_TRY(cudaFree(group_ht_entry<true>::table));
}

template <int MAX_GROUP_BITS, bool EAGER_OUT_IDX>
__device__ void group_ht_insert(
    group_ht_entry<EAGER_OUT_IDX>* hashtable, uint64_t group,
    uint64_t aggregate)
{
    constexpr size_t GROUP_HT_SIZE = ((size_t)1)
                                     << (MAX_GROUP_BITS + GROUP_HT_SIZE_SHIFT);
    constexpr size_t GROUPS_MASK = GROUP_HT_SIZE - 1;
    group_ht_entry<EAGER_OUT_IDX>* hte;
    if (group == GROUP_HT_EMPTY_VALUE) {
        hte = &hashtable[0];
        group_ht_empty_group_used = true;
    }
    else {
        size_t hash = group & GROUPS_MASK;
        if (hash == 0) hash++; // skip the EMPTY_VALUE slot
        hte = &hashtable[hash];
        while (true) {
            if (hte->group == group) break;
            if (hte->group == GROUP_HT_EMPTY_VALUE) {
                uint64_t found = atomicCAS(
                    (cudaUInt64_t*)&hte->group, GROUP_HT_EMPTY_VALUE, group);
                if (found == GROUP_HT_EMPTY_VALUE) {
                    // atomicInc doesn't exist for 64 bit...
                    if (EAGER_OUT_IDX) {
                        hte->eager_inc_out_idx();
                        // DEBUG
                        /*printf(
                            "assigning index %" PRIu64 " to group %" PRIu64
                            " in ht slot %" PRIu64 "\n",
                            hte->output_idx, hte->group, hte - hashtable);*/
                    }
                    break;
                }
                if (found == group) break;
            }
            if (hte != &hashtable[GROUP_HT_SIZE - 1]) {
                hte++;
            }
            else {
                // restart from the beginning of the hashtable,
                // skipping the EMPTY_VALUE in slot 0
                hte = &hashtable[1];
            }
        }
    }
    atomicAdd((cudaUInt64_t*)&hte->aggregate, aggregate);
}

template <int MAX_GROUP_BITS, bool EAGER_OUT_IDX>
__global__ void kernel_fill_group_ht(
    db_table input, group_ht_entry<EAGER_OUT_IDX>* hashtable, int stream_count,
    int stream_idx)
{
    int stride = blockDim.x * gridDim.x * stream_count;
    int tid = threadIdx.x + blockIdx.x * blockDim.x +
              stream_idx * blockDim.x * gridDim.x;
    for (size_t i = tid; i < input.row_count; i += stride) {
        uint64_t group = input.group_col[i];
        uint64_t agg = input.aggregate_col[i];
        group_ht_insert<MAX_GROUP_BITS, EAGER_OUT_IDX>(hashtable, group, agg);
    }
}

template <int MAX_GROUP_BITS, bool EAGER_OUT_IDX>
__global__ void kernel_write_out_group_ht(
    db_table output, group_ht_entry<EAGER_OUT_IDX>* hashtable, int stream_count,
    int stream_idx)
{
    constexpr size_t GROUP_HT_SIZE = ((size_t)1)
                                     << (MAX_GROUP_BITS + GROUP_HT_SIZE_SHIFT);
    int tid = threadIdx.x + blockIdx.x * blockDim.x +
              stream_idx * blockDim.x * gridDim.x;
    int stride = blockDim.x * gridDim.x * stream_count;
    if (tid == 0) {
        group_ht_entry<EAGER_OUT_IDX>* hte = &hashtable[0];
        if (group_ht_empty_group_used) {
            size_t idx;
            if (EAGER_OUT_IDX) {
                // if the EMPTY_VAL group actually occured, increment groups
                // found no race here, nobody else is currently interested in
                // this value
                idx = group_ht_groups_found++;
            }
            else {
                idx = atomicAdd((cudaUInt64_t*)&group_ht_groups_found, 1);
            }
            output.group_col[idx] = GROUP_HT_EMPTY_VALUE;
            output.aggregate_col[idx] = hte->aggregate;
            hte->aggregate = 0;
            group_ht_empty_group_used = false;
        }
        tid += stride;
    }
    for (size_t i = tid; i < GROUP_HT_SIZE; i += stride) {
        group_ht_entry<EAGER_OUT_IDX>* hte = &hashtable[i];
        if (hte->group != GROUP_HT_EMPTY_VALUE) {
            size_t out_idx = hte->aquire_out_idx();
            output.group_col[out_idx] = hte->group;
            output.aggregate_col[out_idx] = hte->aggregate;

            // DEBUG
            /*printf(
                "writing out group %" PRIu64 " in index %" PRIu64 "\n",
                hte->group, hte->output_idx);*/

            // reset for the next run
            hte->group = GROUP_HT_EMPTY_VALUE;
            hte->aggregate = 0;
        }
    }
}

static inline bool approach_hashtable_available(
    int group_bits, int row_count, int grid_dim, int block_dim,
    int stream_count)
{
    if (!grid_dim || !block_dim) return false;
    return true;
}

template <int MAX_GROUP_BITS, bool EAGER_OUT_IDX>
void group_by_hashtable(
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
        kernel_fill_group_ht<MAX_GROUP_BITS, EAGER_OUT_IDX>
            <<<grid_dim, block_dim, 0, stream>>>(
                gd->input, group_ht_entry<EAGER_OUT_IDX>::table,
                actual_stream_count, i);
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
        kernel_write_out_group_ht<MAX_GROUP_BITS, EAGER_OUT_IDX>
            <<<grid_dim, block_dim, 0, stream>>>(
                gd->output, group_ht_entry<EAGER_OUT_IDX>::table,
                actual_stream_count, i);
    }
    CUDA_TRY(cudaEventRecord(end_event));
    CUDA_TRY(cudaGetLastError());
    // read out number of groups found
    // this waits for the kernels to complete since it's in the default stream
    cudaMemcpyFromSymbol(
        &gd->output.row_count, group_ht_groups_found, sizeof(size_t), 0,
        cudaMemcpyDeviceToHost);
}
