#pragma once
#include "cuda_group_by.cuh"
#include <type_traits>
#include "group_by_hashtable.cuh"

#define TPG_MAX_GROUPS 1024
// the minimum number of groups where we use the
// kernel_thread_per_group_more_groups kernel instead of the
// more_threads version
#define TPG_MIN_GROUPS_FOR_MORE_GROUPS_KERNEL 32
#define TPG_MIN_BLOCK_DIM 32
#define TPG_MAX_BLOCK_DIM 1024
#define TPG_MAX_THREADS_PER_GROUP (TPG_MAX_BLOCK_DIM / 2)
#define TPG_EMPTY_GROUP_VALUE 0

__device__ bool tpg_empty_group_used;
__device__ size_t tpg_group_count;
__device__ uint64_t tpg_empty_group_aggregate;

static bool TPG_DEBUG_ONCE = false;

static inline void group_by_thread_per_group_init(size_t max_groups)
{
    bool f = false;
    CUDA_TRY(cudaMemcpyToSymbol(
        tpg_empty_group_used, &f, sizeof(f), 0, cudaMemcpyHostToDevice));
    uint64_t zero = 0;
    CUDA_TRY(cudaMemcpyToSymbol(
        tpg_empty_group_aggregate, &zero, sizeof(zero), 0,
        cudaMemcpyHostToDevice));
    group_by_hashtable_init(max_groups);
}

static inline void group_by_thread_per_group_fin()
{
    group_by_hashtable_fin();
}

static inline bool approach_thread_per_group_available(
    int group_bits, int row_count, int grid_dim, int block_dim,
    int stream_count)
{
    if (!grid_dim || !block_dim) return false;
    const size_t group_count = (1 << group_bits);
    if (group_count > TPG_MAX_GROUPS) return false;
    if (group_count < block_dim) return false;
    if (block_dim < TPG_MIN_BLOCK_DIM) return false;
    if (block_dim > TPG_MAX_BLOCK_DIM) return false;
    return true;
}

template <int MAX_GROUP_BITS, bool NAIVE_WRITEOUT>
__global__ void kernel_thread_per_group_more_threads(
    db_table input, db_table output, group_ht_entry<true>* hashtable,
    int stream_count, int stream_idx)
{
    int threads_per_group = blockDim.x / (1 << MAX_GROUP_BITS);
}

__device__ void thread_per_group_swap_remove_thread_group(
    uint64_t* thread_groups, uint64_t* thread_aggregates, int* groups_in_thread,
    int idx_to_remove)
{
    int last = *groups_in_thread - 1;
    thread_groups[idx_to_remove] = thread_groups[last];
    thread_aggregates[idx_to_remove] = thread_aggregates[last];
    *groups_in_thread = last;
}

template <int MAX_GROUP_BITS>
__device__ void thread_per_group_naive_write_out(
    db_table output, int* groups_found, int groups_in_thread,
    uint64_t* thread_groups, uint64_t* thread_aggregates, int empty_group_slot)
{
    __shared__ uint64_t curr_group;
    __shared__ bool race_won;
    if (threadIdx.x == 0) {
        race_won = false;
    }
    __syncthreads();

    // combine the local aggregates in global memory

    // handle the empty group...
    if (empty_group_slot != -1) {
        atomicAdd(
            (cudaUInt64_t*)&tpg_empty_group_aggregate,
            thread_aggregates[empty_group_slot]);
        tpg_empty_group_used = true;
        thread_per_group_swap_remove_thread_group(
            thread_groups, thread_aggregates, &groups_in_thread,
            empty_group_slot);
        // again, max one thread can get here, so no race
        groups_found--;
    }
    for (size_t i = 0; i < (size_t)1 << MAX_GROUP_BITS; i++) {
        __syncthreads();
        if (groups_found == 0) break;
        if (threadIdx.x == 0) {
            curr_group = output.group_col[i];
        }
        __syncthreads();
        uint64_t group = curr_group;
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
                    race_won = true;
                    groups_in_thread--;
                    groups_found--;
                    atomicAdd((cudaUInt64_t*)&tpg_group_count, 1);
                }
            }
            __syncthreads();
            // if somebody else won group is now a newly added group
            // that we need to test again
            if (race_won) {
                __syncthreads();
                if (threadIdx.x == 0) {
                    race_won = false;
                }
                continue;
            }
            else {
                if (threadIdx.x == 0) {
                    curr_group = output.group_col[i];
                }
                __syncthreads();
                group = curr_group;
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
                thread_per_group_swap_remove_thread_group(
                    thread_groups, thread_aggregates, &groups_in_thread, j);
                // INVESTIGATE: the hope is that this is not racing since
                // there is a syncthreads afterwards and only one thread can
                // get here per iteration
                groups_found--;
                // again, debatable whether or not we should break here
                // definitely no need for j-- here though, since the other
                // groups cannot match on this i
                /*printf(
                    "increased group %llu: (%i %i %i) [out idx: %i, gf: %i,
                   " "git: %i]\n", group, threadIdx.x, blockIdx.x,
                   stream_idx, i, groups_found, groups_in_thread);*/
            }
        }
        // so groups_found in the next iteration is not racing
        // again, is this correct???
        // also, is it even worth the optimization that we get from
        // the early exit for groups_found=0 ?
        __syncthreads();
    }
}

template <int MAX_GROUP_BITS>
__device__ void thread_per_group_hashmap_write_out(
    db_table output, group_ht_entry<true>* hashtable, int* groups_found,
    int groups_in_thread, uint64_t* thread_groups, uint64_t* thread_aggregates)
{
    // no need for syncthreads since we are sure to have all the groups
    // that we are responsible for anyways
    for (int i = 0; i < groups_in_thread; i++) {
        group_ht_insert<MAX_GROUP_BITS, true>(
            hashtable, thread_groups[i], thread_aggregates[i]);
    }
}

template <int MAX_GROUP_BITS, bool NAIVE_WRITEOUT>
__global__ void kernel_thread_per_group_more_groups(
    db_table input, db_table output, group_ht_entry<true>* hashtable,
    int stream_count, int stream_idx)
{
    __shared__ uint64_t groups[TPG_MAX_BLOCK_DIM];
    __shared__ uint64_t aggregates[TPG_MAX_BLOCK_DIM];

    __shared__ int groups_found;
    __shared__ int rows_read;
    __shared__ bool row_consumed;

    size_t base_idx =
        (size_t)blockIdx.x * blockDim.x + stream_idx * blockDim.x * gridDim.x;

    size_t idx = threadIdx.x + base_idx;
    size_t stride = (size_t)blockDim.x * gridDim.x * stream_count;

    // initialize shared variables
    if (threadIdx.x == 0) {
        groups_found = 0;
        rows_read = blockDim.x;
        row_consumed = false;
    }
    __syncthreads();

    constexpr int MAX_GROUPS_PER_THREAD =
        (1 << MAX_GROUP_BITS) / TPG_MIN_BLOCK_DIM;
    uint64_t thread_groups[MAX_GROUPS_PER_THREAD];
    uint64_t thread_aggregates[MAX_GROUPS_PER_THREAD];

    int groups_in_thread = 0;
    int empty_group_slot = -1;
    (void)empty_group_slot; // prevent unused warning for !NAIVE_WRITEOUT
    while (rows_read == blockDim.x) {
        // read in one row per thread
        if (idx >= input.row_count) {
            // the last thread is guaranteed to overflow if any thread does
            if (threadIdx.x == blockDim.x - 1) {
                int threads_over_row_count = idx - input.row_count + 1;
                if (threads_over_row_count > blockDim.x) {
                    rows_read = 0;
                }
                else {
                    rows_read = blockDim.x - threads_over_row_count;
                }
            }
        }
        else {
            groups[threadIdx.x] = input.group_col[idx];
            aggregates[threadIdx.x] = input.aggregate_col[idx];
            idx += stride;
        }
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
                __syncthreads();
                if (threadIdx.x == 0) {
                    row_consumed = false;
                }
                __syncthreads();
                continue;
            }
            // one thread gets assigned the newly found group
            bool responsible = (groups_found % blockDim.x == threadIdx.x);
            __syncthreads();
            if (responsible) {
                thread_groups[groups_in_thread] = group;
                thread_aggregates[groups_in_thread] = aggregates[i];
                if (NAIVE_WRITEOUT) {
                    if (group == TPG_EMPTY_GROUP_VALUE) {
                        empty_group_slot = groups_in_thread;
                    }
                }
                groups_in_thread++;
                groups_found++;
            }
        }
    }
    if (NAIVE_WRITEOUT) {
        thread_per_group_naive_write_out<MAX_GROUP_BITS>(
            output, &groups_found, groups_in_thread, thread_groups,
            thread_aggregates, empty_group_slot);
    }
    else {
        thread_per_group_hashmap_write_out<MAX_GROUP_BITS>(
            output, hashtable, &groups_found, groups_in_thread, thread_groups,
            thread_aggregates);
    }
}

__global__ void kernel_thread_per_group_insert_empty_group(db_table output)
{
    if (tpg_empty_group_used) {
        // the group_col is already correctly set to TPG_EMPTY_GROUP_VALUE
        output.aggregate_col[tpg_group_count] = tpg_empty_group_aggregate;
        tpg_group_count++;
        // reset for the next run
        tpg_empty_group_used = false;
        tpg_empty_group_aggregate = 0;
    }
}

// we specialize a template based on whether
// MAX_GROUPS is >= TPG_MIN_GROUPS_FOR_MORE_GROUPS_KERNEL
// this way we don't get zero sized type errors in our kernels
// (and avoid unnecessary template instantiations)
template <
    int MAX_GROUP_BITS, bool NAIVE_WRITEOUT,
    bool MORE_GROUPS =
        ((1 << MAX_GROUP_BITS) >= TPG_MIN_GROUPS_FOR_MORE_GROUPS_KERNEL)>
struct group_based_kernel_dispatch {
};

template <int MAX_GROUP_BITS, bool NAIVE_WRITEOUT>
struct group_based_kernel_dispatch<MAX_GROUP_BITS, NAIVE_WRITEOUT, true> {
    static void call(
        int grid_dim, int block_dim, cudaStream_t stream, db_table input,
        db_table output, group_ht_entry<true>* hashtable, int stream_count,
        int stream_idx)
    {
        kernel_thread_per_group_more_groups<MAX_GROUP_BITS, NAIVE_WRITEOUT>
            <<<grid_dim, block_dim, 0, stream>>>(
                input, output, hashtable, stream_count, stream_idx);
    }
};

template <int MAX_GROUP_BITS, bool NAIVE_WRITEOUT>
struct group_based_kernel_dispatch<MAX_GROUP_BITS, NAIVE_WRITEOUT, false> {
    static void call(
        int grid_dim, int block_dim, cudaStream_t stream, db_table input,
        db_table output, group_ht_entry<true>* hashtable, int stream_count,
        int stream_idx)
    {
        kernel_thread_per_group_more_threads<MAX_GROUP_BITS, NAIVE_WRITEOUT>
            <<<grid_dim, block_dim, 0, stream>>>(
                input, output, hashtable, stream_count, stream_idx);
    }
};

template <int MAX_GROUP_BITS, bool NAIVE_WRITEOUT>
void group_by_thread_per_group(
    gpu_data* gd, int grid_dim, int block_dim, int stream_count,
    cudaStream_t* streams, cudaEvent_t* events, cudaEvent_t start_event,
    cudaEvent_t end_event)
{
    constexpr size_t MAX_GROUPS = (size_t)1 << MAX_GROUP_BITS;
    int actual_stream_count = stream_count ? stream_count : 1;
    uint64_t zero = 0;

    CUDA_TRY(cudaEventRecord(start_event));
    if (NAIVE_WRITEOUT) {
        // same reasoning as in hashtable_init
        assert(TPG_EMPTY_GROUP_VALUE == 0);
        CUDA_TRY(
            cudaMemset(gd->output.group_col, 0, MAX_GROUPS * sizeof(uint64_t)));
        CUDA_TRY(cudaMemcpyToSymbol(
            tpg_group_count, &zero, sizeof(zero), 0, cudaMemcpyHostToDevice));
        CUDA_TRY(cudaMemset(
            gd->output.aggregate_col, 0, MAX_GROUPS * sizeof(uint64_t)));
    }
    else {
        cudaMemcpyToSymbol(
            group_ht_groups_found, &zero, sizeof(zero), 0,
            cudaMemcpyHostToDevice);
    }
    for (int i = 0; i < actual_stream_count; i++) {
        cudaStream_t stream = stream_count ? streams[i] : 0;
        group_based_kernel_dispatch<MAX_GROUP_BITS, NAIVE_WRITEOUT>::call(
            grid_dim, block_dim, stream, gd->input, gd->output,
            group_ht_entry<true>::table, actual_stream_count, i);
        if (!NAIVE_WRITEOUT) {
            if (stream_count > 1) cudaEventRecord(events[i], stream);
        }
    }
    if (NAIVE_WRITEOUT) {
        kernel_thread_per_group_insert_empty_group<<<1, 1>>>(gd->output);
    }
    else {
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
            kernel_write_out_group_ht<MAX_GROUP_BITS, true>
                <<<grid_dim, block_dim, 0, stream>>>(
                    gd->output, group_ht_entry<true>::table,
                    actual_stream_count, i);
        }
    }
    CUDA_TRY(cudaEventRecord(end_event));
    CUDA_TRY(cudaGetLastError());
    if (NAIVE_WRITEOUT) {
        cudaMemcpyFromSymbol(
            &gd->output.row_count, tpg_group_count, sizeof(size_t), 0,
            cudaMemcpyDeviceToHost);
    }
    else {
        cudaMemcpyFromSymbol(
            &gd->output.row_count, group_ht_groups_found, sizeof(size_t), 0,
            cudaMemcpyDeviceToHost);
    }
}
