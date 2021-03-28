#pragma once
#include <cstddef>
#include <cstdint>
#include <cassert>

struct db_table {
    uint64_t* group_col;
    uint64_t* aggregate_col;
    size_t row_count;
};

// we need this stupid type since c++ doesn't see
// uint64_t and unsigned long long int as the same type
// (even though they are on literally any plattform)
// this causes e.g. atomicAdd to complain about parameter types
typedef unsigned long long int cudaUInt64_t;

// number of bits to 'oversize' the hashtable by relative to MAX_GROUP_BITS
// we use a shift instead of a multiplier to make sure we keep the nice power of
// two properties
#define GROUP_HT_SIZE_SHIFT 1
#define GROUP_HT_OUTPUT_IDX_DUMMY_VAL ((size_t)-1)
#define GROUP_HT_EMPTY_VALUE 0

struct group_ht_entry {
    uint64_t group;
    uint64_t aggregate;
    size_t output_idx;
};

struct gpu_data {
    db_table input;
    db_table output;
    group_ht_entry* group_table;
};

#define GPU_MAX_THREADS_PER_BLOCK 1024

__device__ size_t groups_found = 0;

void alloc_db_table_gpu(db_table* t, uint64_t row_count)
{
    CUDA_TRY(cudaMalloc(&t->group_col, row_count * sizeof(uint64_t)));
    CUDA_TRY(cudaMalloc(&t->aggregate_col, row_count * sizeof(uint64_t)));
    t->row_count = row_count;
}

void free_db_table_gpu(db_table* t)
{
    CUDA_TRY(cudaFree(t->group_col));
    CUDA_TRY(cudaFree(t->aggregate_col));
}

void gpu_data_alloc(gpu_data* gd, size_t max_rows, size_t max_groups)
{
    alloc_db_table_gpu(&gd->input, max_rows);
    alloc_db_table_gpu(&gd->output, max_groups);
    const size_t ht_size =
        (max_groups << GROUP_HT_SIZE_SHIFT) * sizeof(group_ht_entry);
    CUDA_TRY(cudaMalloc(&gd->group_table, ht_size));
    CUDA_TRY(cudaMemset(gd->group_table, 0, ht_size));
    // if we want to use a different empty value (0 is probably a common group)
    // we will need to properly initialize this, since the memset will no longer
    // suffice
    assert(GROUP_HT_EMPTY_VALUE == 0);
}

void gpu_data_free(gpu_data* gd)
{
    CUDA_TRY(cudaFree(gd->group_table));
    free_db_table_gpu(&gd->output);
    free_db_table_gpu(&gd->input);
}

template <int MAX_GROUP_BITS>
__global__ void
kernel_fill_group_ht(gpu_data gd, int stream_count, int stream_idx)
{
    constexpr size_t GROUP_HT_SIZE = ((size_t)1)
                                     << (MAX_GROUP_BITS + GROUP_HT_SIZE_SHIFT);
    constexpr size_t GROUPS_MASK = GROUP_HT_SIZE - 1;
    int stride = blockDim.x * gridDim.x * stream_count;
    int tid = threadIdx.x + blockIdx.x * blockDim.x +
              stream_idx * blockDim.x * gridDim.x;
    for (size_t i = tid; i < gd.input.row_count; i += stride) {
        uint64_t group = gd.input.group_col[i];
        uint64_t agg = gd.input.aggregate_col[i];
        group_ht_entry* hte;
        if (group == GROUP_HT_EMPTY_VALUE) {
            hte = &gd.group_table[0];
            hte->output_idx = GROUP_HT_OUTPUT_IDX_DUMMY_VAL;
        }
        else {
            size_t hash = group & GROUPS_MASK;
            if (hash == 0) hash++; // skip the EMPTY_VALUE slot
            hte = &gd.group_table[hash];
            while (true) {
                if (hte->group == group) break;
                if (hte->group == GROUP_HT_EMPTY_VALUE) {
                    uint64_t found = atomicCAS(
                        (cudaUInt64_t*)&hte->group, GROUP_HT_EMPTY_VALUE,
                        group);
                    if (found == GROUP_HT_EMPTY_VALUE) {
                        // atomicInc doesn't exist for 64 bit...
                        hte->output_idx =
                            atomicAdd((cudaUInt64_t*)&groups_found, 1);
                        /*
                        // DEBUG
                        printf(
                            "assigning index %llu to group %llu in ht slot "
                            "%llu\n",
                            hte->output_idx, hte->group, hte - gd.group_table);
                        */
                        break;
                    }
                    if (found == group) break;
                }
                if (hte != &gd.group_table[GROUP_HT_SIZE - 1]) {
                    hte++;
                }
                else {
                    // restart from the beginning of the hashtable,
                    // skipping the EMPTY_VALUE in slot 0
                    hte = &gd.group_table[1];
                }
            }
        }
        atomicAdd((cudaUInt64_t*)&hte->aggregate, agg);
    }
}
template <int MAX_GROUP_BITS>
__global__ void
kernel_write_out_group_ht(gpu_data gd, int stream_count, int stream_idx)
{
    constexpr size_t GROUP_HT_SIZE = ((size_t)1)
                                     << (MAX_GROUP_BITS + GROUP_HT_SIZE_SHIFT);
    int tid = threadIdx.x + blockIdx.x * blockDim.x +
              stream_idx * blockDim.x * gridDim.x;
    int stride = blockDim.x * gridDim.x * stream_count;
    if (tid == 0) {
        group_ht_entry* hte = &gd.group_table[0];
        if (hte->output_idx == GROUP_HT_OUTPUT_IDX_DUMMY_VAL) {
            size_t idx = groups_found;
            // if the EMPTY_VAL group actually occured, increment groups found
            // no race here, nobody else is currently interested in this value
            groups_found++;
            gd.output.group_col[idx] = GROUP_HT_EMPTY_VALUE;
            gd.output.aggregate_col[idx] = hte->aggregate;
            hte->output_idx = 0;
            hte->aggregate = 0;
        }
        tid += stride;
    }
    for (size_t i = tid; i < GROUP_HT_SIZE; i += stride) {
        group_ht_entry* hte = &gd.group_table[i];
        if (hte->group != GROUP_HT_EMPTY_VALUE) {
            gd.output.group_col[hte->output_idx] = hte->group;
            gd.output.aggregate_col[hte->output_idx] = hte->aggregate;
            /*
            // DEBUG
            printf(
                "writing out group %llu in index %llu\n", hte->group,
                hte->output_idx);
            */

            // reset for the next run
            hte->group = GROUP_HT_EMPTY_VALUE;
            hte->aggregate = 0;
        }
    }
}
template <int MAX_GROUP_BITS> void group_by_thread_per_group()
{
    constexpr size_t MAX_GROUPS = 1 << MAX_GROUP_BITS;
    static_assert(
        GPU_MAX_THREADS_PER_BLOCK >= MAX_GROUPS,
        "strategy not available, MAX_GROUPS exceeds "
        "GPU_MAX_THREADS_PER_BLOCK");
}

template <int MAX_GROUP_BITS>
void group_by_hashtable(
    gpu_data* gd, int grid_size, int block_size, int stream_count,
    cudaStream_t* streams, cudaEvent_t* events, cudaEvent_t start_event,
    cudaEvent_t end_event)
{
    constexpr size_t MAX_GROUPS = 1 << MAX_GROUP_BITS;
    // reset number of groups found
    size_t zero = 0;
    cudaMemcpyToSymbol(
        groups_found, &zero, sizeof(zero), 0, cudaMemcpyHostToDevice);

    CUDA_TRY(cudaEventRecord(start_event));
    for (int i = 0; i < stream_count; i++) {
        cudaStream_t stream = (stream_count > 1) ? streams[i] : 0;
        kernel_fill_group_ht<MAX_GROUP_BITS>
            <<<block_size, grid_size, 0, stream>>>(*gd, stream_count, i);
        if (stream) cudaEventRecord(events[i], stream);
    }
    // since it's likely that our block / grid dims are overkill
    // for the write out kernel we reduce them a bit
    if (block_size * grid_size * stream_count > MAX_GROUPS) {
        grid_size = MAX_GROUPS / (stream_count * block_size);
        if (!grid_size) grid_size = 1;
    }
    for (int i = 0; i < stream_count; i++) {
        cudaStream_t stream = (stream_count > 1) ? streams[i] : 0;
        if (stream) {
            // every write out kernel needs to wait on every fill kernel
            for (int j = 0; j < stream_count; j++) {
                cudaStreamWaitEvent(stream, events[j], 0);
            }
        }
        kernel_write_out_group_ht<MAX_GROUP_BITS>
            <<<block_size, grid_size, 0, stream>>>(*gd, stream_count, i);
    }
    CUDA_TRY(cudaEventRecord(end_event));
    CUDA_TRY(cudaGetLastError());
    // read out number of groups found
    // this waits for the kernels to complete since it's in the default stream
    cudaMemcpyFromSymbol(
        &gd->output.row_count, groups_found, sizeof(size_t), 0,
        cudaMemcpyDeviceToHost);
}
