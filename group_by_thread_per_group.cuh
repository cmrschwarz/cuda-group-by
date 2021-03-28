#pragma once
#include "cuda_group_by.cuh"

#define TPG_MAX_GROUPS 512
#define TPG_MAX_BLOCK_DIM 1024
#define TPG_EMPTY_GROUP_VALUE 0

__device__ size_t tpg_group_count;
__device__ bool tpg_empty_group_used;
__device__ uint64_t tpg_empty_group_aggregate;

static inline void group_by_thread_per_group_init()
{
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
    const size_t group_count = (1 << group_bits);
    if (group_count > TPG_MAX_GROUPS) return false;
    if (group_count < block_size) return false;
    return true;
}

template <int MAX_GROUPS>
__device__ void kernel_thread_per_group_more_threads(
    db_table input, db_table output, int stream_idx, int stream_count)
{
}

__device__ void swap_remove_thread_group(
    uint64_t* thread_groups, uint64_t* thread_aggregates, int* groups_in_thread,
    int idx_to_remove)
{
    thread_groups[idx_to_remove] = thread_groups[*groups_in_thread - 1];
    thread_aggregates[idx_to_remove] = thread_aggregates[*groups_in_thread - 1];
    *groups_in_thread--;
}

template <int MAX_GROUPS>
__global__ void kernel_thread_per_group_more_groups(
    db_table input, db_table output, int stream_idx, int stream_count)
{
    __shared__ uint64_t groups[TPG_MAX_BLOCK_DIM];
    __shared__ uint64_t aggregates[TPG_MAX_BLOCK_DIM];

    __shared__ int groups_found;
    __shared__ int rows_read;
    __shared__ bool row_consumed;
    __shared__ bool race_won;

    // initialize shared variables
    if (threadIdx.x == 0) {
        groups_found = blockDim.x;
        rows_read = blockDim.x;
        row_consumed = false;
        race_won = false;
    }
    __syncthreads();
    size_t idx = (size_t)threadIdx.x + blockIdx.x * blockDim.x +
                 stream_idx * blockDim.x * gridDim.x;
    size_t stride = (size_t)blockDim.x * gridDim.x * stream_count;

    // the availability function guarantees us that this is divisible
    const int groups_per_thread = blockDim.x / MAX_GROUPS;

    uint64_t thread_groups[groups_per_thread];
    uint64_t thread_aggregates[groups_per_thread];

    int groups_in_thread = 0;
    int empty_group_slot = -1;

    while (rows_read == blockDim.x) {
        // read in one row per thread
        if (idx >= input.row_count) {
            // the thread that exactly hits the end of the input
            // reduces the rows_read field
            if (idx == input.row_count) {
                rows_read = threadIdx.x;
            }
        }
        groups[threadIdx.x] = input.group_col[idx];
        aggregates[threadIdx.x] = input.aggregate_col[idx];
        idx += stride;
        __syncthreads();

        // consume on row at a time
        // TODO: maybe consume multiple to reduce the number of syncthreads
        for (int i = 0; i < rows_read; i++) {
            uint64_t group = groups[i];
            for (int j = 0; j < groups_in_thread; j++) {
                if (group == thread_groups[j]) {
                    row_consumed = true;
                    thread_aggregates[j] += aggregates[i];
                    // no use in breaking here since the other threads
                    // would continue anyways
                    // we could break on row consumed but that's probably
                    // slower?
                }
            }
            __syncthreads();
            if (row_consumed) {
                if (threadIdx.x == 0) {
                    row_consumed = false;
                }
                __syncthreads();
                continue;
            }
            // one thread gets assigned the newly found group
            if (groups_found % blockDim.x == threadIdx.x) {
                thread_groups[groups_in_thread] = group;
                thread_aggregates[groups_in_thread] = aggregates[i];
                if (group == TPG_EMPTY_GROUP_VALUE) {
                    empty_group_slot = groups_in_thread;
                }
                groups_in_thread++;
                groups_found++;
            }
        }
    }
    // combine the local aggregates in global memory

    // handle the empty group...
    if (empty_group_slot != -1) {
        atomicAdd(
            (cudaUInt64_t*)tpg_empty_group_aggregate,
            aggregates[empty_group_slot]);
        tpg_empty_group_used = true;
        swap_remove_thread_group(
            thread_groups, thread_aggregates, &groups_in_thread,
            empty_group_slot);
    }

    for (int i = 0; i < MAX_GROUPS; i++) {
        if (groups_found == 0) break;
        // the hope is that with this loop ordering the global memory reads
        // will be somwhat broadcasted
        // maybe investigate __shared__ here??
        uint64_t group = output.group_col[i];
        // if the empty group occurs
        // we can be assured that this will happen at some point
        // since we have MAX_GROUPS slots but only insert MAX_GROUPS-1
        // groups since we store the empty group externally
        // if the empty group doesn't occur and this branch isn't taken
        // we know that all groups must have been sorted in and
        // we're fine too
        if (group == TPG_EMPTY_GROUP_VALUE) {
            // TODO: maybe figure out which thread from the block
            // should go first and then do this only for one ?

            if (groups_in_thread) {
                int last_group_idx = groups_in_thread - 1;
                group = atomicCAS(
                    (cudaUInt64_t*)&output.group_col[i], TPG_EMPTY_GROUP_VALUE,
                    thread_groups[last_group_idx]);
                if (group == TPG_EMPTY_GROUP_VALUE) {
                    // this must be atomic because we are already racing
                    // against other blocks
                    atomicAdd(
                        (cudaUInt64_t*)&output.aggregate_col[i],
                        thread_aggregates[last_group_idx]);
                    swap_remove_thread_group(
                        thread_groups, thread_aggregates, &groups_in_thread, 0);
                    race_won = true;
                    groups_in_thread--;
                    groups_found--;
                }
            }
            __syncthreads();
            // if somebody else won group is now a newly added group
            // that we need to test again
            if (race_won) {
                race_won = false;
                __syncthreads();
                continue;
            }
        }
        for (int j = 0; j < groups_in_thread; j++) {
            if (thread_groups[j] == group) {
                atomicAdd(
                    (cudaUInt64_t*)&output.aggregate_col[i],
                    thread_aggregates[j]);
                // swap remove the found group from thread_groups
                // we know that groups_in_thread >= 1 since otherwise
                // we wouldn't be in this loop
                swap_remove_thread_group(
                    thread_groups, thread_aggregates, &groups_in_thread, j);
                // INVESTIGATE: the hope is that this is not racing since there
                // is a syncthreads afterwards and only one thread can get here
                // per iteration
                groups_found--;
                // again, debatable whether or not we should break here
                // definitely no need for j-- here though, since the other
                // groups cannot match on this i
            }
        }
        // so groups_found in the next iteration is not racing
        // again, is this correct???
        // also, is it even worth the optimization that we get from
        // the early exit for groups_found=0 ?
        __syncthreads();
    }
}

template <int MAX_GROUPS>
void group_by_thread_per_group(
    gpu_data* gd, int grid_size, int block_size, int stream_count,
    cudaStream_t* streams, cudaEvent_t* events, cudaEvent_t start_event,
    cudaEvent_t end_event)
{

    assert(approach_thread_per_group_available(
        MAX_GROUPS, gd->input.row_count, grid_size, block_size, stream_count));
    int actual_stream_count = stream_count ? stream_count : 1;

    // same reasoning as in hashtable_init
    assert(TPG_EMPTY_GROUP_VALUE == 0);
    cudaMemset(gd->output.group_col, 0, MAX_GROUPS * sizeof(uint64_t));
    cudaMemset(gd->output.aggregate_col, 0, MAX_GROUPS * sizeof(uint64_t));
    for (int i = 0; i < actual_stream_count; i++) {
        cudaStream_t stream = stream_count ? streams[i] : 0;
        kernel_thread_per_group_more_groups<MAX_GROUPS>
            <<<block_size, grid_size, 0, stream>>>(*gd, actual_stream_count, i);
    }
}
