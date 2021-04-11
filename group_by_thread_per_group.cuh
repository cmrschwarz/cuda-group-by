#pragma once
#include "cuda_group_by.cuh"
#include <type_traits>
#include "group_by_hashtable.cuh"

#define TPG_MAX_GROUPS 1024
// the minimum number of groups where we use the
// kernel_thread_per_group_more_groups kernel instead of the
// more_threads version
#define CUDA_WARP_SIZE 32
#define TPG_MIN_GROUPS_FOR_MORE_GROUPS_KERNEL (CUDA_WARP_SIZE * 2)
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
    const size_t group_count = (1 << group_bits);
    // if (group_bits != 8 || row_count != 8192) return false; // DEBUG
    if (!grid_dim || !block_dim) return false;
    if (group_count > TPG_MAX_GROUPS) return false;
    if (block_dim < TPG_MIN_BLOCK_DIM) return false;
    if (block_dim > TPG_MAX_BLOCK_DIM) return false;
    if (group_count >= TPG_MIN_GROUPS_FOR_MORE_GROUPS_KERNEL) {
        if (group_count < block_dim) return false;
    }
    return true;
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

template <int MAX_GROUP_BITS, bool GROUPS_DISTINCT>
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
        if (GROUPS_DISTINCT) {
            groups_found--;
        }
        else {
            atomicSub(groups_found, 1);
        }
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
                    if (GROUPS_DISTINCT) {
                        groups_found--;
                    }
                    else {
                        atomicSub(groups_found, 1);
                    }
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
                if (GROUPS_DISTINCT) continue;
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
                if (GROUPS_DISTINCT) {
                    // this is not racing since
                    // there is a syncthreads afterwards and only one thread can
                    // get here per iteration
                    groups_found--;
                }
                else {
                    atomicSub(groups_found, 1);
                }
                //  debatable whether or not we should break here
                // definitely no need for j-- here though, since the other
                // groups cannot match on this i
            }
        }
        // so the groups_found check in the next iteration is not racing
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
__global__ void kernel_thread_per_group_more_threads(
    db_table input, db_table output, group_ht_entry<true>* hashtable,
    int stream_count, int stream_idx)
{
    size_t base_idx = (size_t)blockIdx.x * blockDim.x +
                      (size_t)stream_idx * blockDim.x * gridDim.x;
    size_t idx = threadIdx.x + base_idx;
    size_t stride = (size_t)blockDim.x * gridDim.x * stream_count;
    // this number is one of {2,4,8,16,32}
    constexpr int BATCH_SIZE = (1 << MAX_GROUP_BITS);
    int batch_idx = threadIdx.x / BATCH_SIZE;
    int batch_base = batch_idx * BATCH_SIZE;
    int next_batch_base = batch_base + BATCH_SIZE;
    bool group_assigned = false;
    uint64_t assigned_group;
    uint64_t assigned_aggregate;
    __shared__ uint64_t groups[TPG_MAX_BLOCK_DIM];
    __shared__ uint64_t aggregates[TPG_MAX_BLOCK_DIM];
    __shared__ bool row_handled[TPG_MAX_BLOCK_DIM];
    __shared__ int handout_ids[TPG_MAX_BLOCK_DIM];
    __shared__ int handout_counters[TPG_MAX_BLOCK_DIM / BATCH_SIZE];

    size_t rowcount = input.row_count;
    int prev_handout_counter = batch_base;
    if (threadIdx.x == batch_base) {
        handout_counters[batch_idx] = batch_base;
    }
    handout_ids[threadIdx.x] = -1;
    __syncwarp();

    // group aquiring phase
    // some threads in the warp have no assigned group, must handle the handout
    while (base_idx < rowcount) {
        // read input into shared memory, 1 element per thread
        if (idx < rowcount) {
            groups[threadIdx.x] = input.group_col[idx];
            aggregates[threadIdx.x] = input.aggregate_col[idx];
            row_handled[threadIdx.x] = false;
        }
        else {
            row_handled[threadIdx.x] = true;
        }

        // each thread checks each input to see if it's responsible
        __syncwarp();
        for (int i = 0; i < BATCH_SIZE; i++) {
            if (groups[batch_base + i] == assigned_group) {
                if (group_assigned && base_idx + i < rowcount) {
                    row_handled[batch_base + i] = true;
                    assigned_aggregate += aggregates[batch_base + i];
                }
            }
        }
        idx += stride;
        base_idx += stride;

        // if no thread was responsible for the input this thread read in,
        // hand out this group to one of the unassigned threads
        __syncwarp();
        if (!row_handled[threadIdx.x]) {
            int hand_out_to_tid = atomicAdd(&handout_counters[batch_idx], 1);
            if (hand_out_to_tid >= next_batch_base) {
                hand_out_to_tid -= BATCH_SIZE;
            }
            handout_ids[hand_out_to_tid] = threadIdx.x;
        }

        // check if we have to deal with handouts
        __syncwarp();
        int new_handout_counter = handout_counters[batch_idx];
        if (new_handout_counter == prev_handout_counter) continue;

        // if our thread got handed out a group responsibility,
        // check if nobody else before us got assigned the same group
        // if not, take on responsibility, if yes, yield resp. to the lower tid
        __syncwarp();
        int handout_id = handout_ids[threadIdx.x];
        bool handout_assigned = false;
        if (handout_id != -1) {
            handout_assigned = true;
            uint64_t handed_out_group = groups[handout_id];
            int i = prev_handout_counter;
            while (i != threadIdx.x) {
                int prev_handout_id = handout_ids[i];
                if (groups[prev_handout_id] == handed_out_group) {
                    atomicSub(&handout_counters[batch_idx], 1);
                    atomicAdd(
                        (cudaUInt64_t*)&aggregates[prev_handout_id],
                        aggregates[handout_id]);
                    handout_assigned = false;
                    break;
                }
                i++;
                if (i == next_batch_base) i = batch_base;
            }
        }

        // if we yielded, reset up our handout id
        // we have to do this after the actual yield since
        // other threads may still index based on it during yield detection
        __syncwarp();
        if (handout_id != -1 && handout_assigned == false) {
            handout_ids[threadIdx.x] = -1;
        }

        // if we yielded our responsibility to a lower tid, but a higher tid
        // didn't yield, steal its responsibility
        // the i'th thread gaining responsibility steals from the i'th non
        // yielding one
        __syncwarp();
        int steal_from_tid = -1;
        if (handout_id != -1) {
            new_handout_counter = handout_counters[batch_idx];
            if (!handout_assigned && new_handout_counter > threadIdx.x &&
                !group_assigned) {
                handout_assigned = true;
                int handout_priority = 0;
                // since the lowest tid can't yield, start from the next
                for (int i = prev_handout_counter + 1; i < threadIdx.x; i++) {
                    if (handout_ids[i] == -1) handout_priority++;
                }
                steal_from_tid = new_handout_counter;
                while (true) {
                    if (steal_from_tid == next_batch_base) {
                        steal_from_tid = batch_base;
                    }
                    handout_id = handout_ids[steal_from_tid];
                    if (handout_id != -1) {
                        if (handout_priority == 0) break;
                        handout_priority--;
                    }
                    steal_from_tid++;
                }
            }
        }

        // inform the tid that we stole from that it's no longer responsible
        __syncwarp();
        if (steal_from_tid != -1) {
            handout_ids[steal_from_tid] = -1;
            handout_ids[threadIdx.x] = handout_id;
        }

        // finally assign group if we weren't stolen from,
        __syncwarp();
        if (handout_assigned && handout_id != -1 &&
            handout_ids[threadIdx.x] != -1) {
            handout_ids[threadIdx.x] = -1;
            group_assigned = true;
            assigned_group = groups[handout_id];
            assigned_aggregate = aggregates[handout_id];
        }

        // if the last thread in the warp was assigned a group, advance
        // to the satisfied warp phase
        prev_handout_counter = handout_counters[batch_idx];
        if (prev_handout_counter == next_batch_base) {
            break;
        }
    }

    // satisfied warp phase
    // every thread has been assigned a group, no need to check anymore
    while (base_idx < rowcount) {
        // read input into shared memory, 1 element per thread
        if (idx < rowcount) {
            groups[threadIdx.x] = input.group_col[idx];
            aggregates[threadIdx.x] = input.aggregate_col[idx];
        }

        __syncwarp();
        for (int i = 0; i < BATCH_SIZE; i++) {
            if (groups[batch_base + i] == assigned_group) {
                if (base_idx + i < rowcount) {
                    assigned_aggregate += aggregates[batch_base + i];
                }
            }
        }

        idx += stride;
        base_idx += stride;
    }

    // the input is processed, proceed to writeout
    if (NAIVE_WRITEOUT) {
        __shared__ int groups_found;
        if (threadIdx.x == 0) {
            groups_found = 0;
        }
        __syncthreads();
        if (threadIdx.x % CUDA_WARP_SIZE == 0) {
            atomicAdd(&groups_found, handout_counters[batch_idx] - batch_base);
        }
        __syncthreads();
        int empty_group_slot = -1;
        if (group_assigned && assigned_group == TPG_EMPTY_GROUP_VALUE) {
            empty_group_slot = 0;
        }
        thread_per_group_naive_write_out<MAX_GROUP_BITS, false>(
            output, &groups_found, group_assigned ? 1 : 0, &assigned_group,
            &assigned_aggregate, empty_group_slot);
    }
    else {
        if (group_assigned) {
            group_ht_insert<MAX_GROUP_BITS, true>(
                hashtable, assigned_group, assigned_aggregate);
        }
    }
}

template <int MAX_GROUP_BITS, bool NAIVE_WRITEOUT>
__global__ void kernel_thread_per_group_more_groups(
    db_table input, db_table output, group_ht_entry<true>* hashtable,
    int stream_count, int stream_idx)
{
    size_t base_idx = (size_t)blockIdx.x * blockDim.x +
                      (size_t)stream_idx * blockDim.x * gridDim.x;
    size_t idx = threadIdx.x + base_idx;
    size_t stride = (size_t)blockDim.x * gridDim.x * stream_count;
    constexpr int MAX_GROUPS = 1 << MAX_GROUP_BITS;
    constexpr int MAX_GROUPS_PER_THREAD = MAX_GROUPS / TPG_MIN_BLOCK_DIM;
    const int RT_MAX_GROUPS_PER_THREAD = MAX_GROUPS / blockDim.x;
    uint64_t thread_groups[MAX_GROUPS_PER_THREAD];
    uint64_t thread_aggregates[MAX_GROUPS_PER_THREAD];
    int thread_group_count = 0;

    __shared__ uint64_t groups[TPG_MAX_BLOCK_DIM];
    __shared__ uint64_t aggregates[TPG_MAX_BLOCK_DIM];
    __shared__ bool row_handled[TPG_MAX_BLOCK_DIM];
    __shared__ int handout_ids[TPG_MAX_BLOCK_DIM];
    __shared__ int handout_counter;

    size_t rowcount = input.row_count;
    int prev_handout_counter = 0;
    if (threadIdx.x == 0) {
        handout_counter = 0;
    }
    handout_ids[threadIdx.x] = -1;
    __syncthreads();

    // group aquiring phase
    // some threads in the warp have no unassigned group slots,
    // must handle the handout
    while (base_idx < rowcount) {
        // read input into shared memory, 1 element per thread
        if (idx < rowcount) {
            groups[threadIdx.x] = input.group_col[idx];
            aggregates[threadIdx.x] = input.aggregate_col[idx];
            row_handled[threadIdx.x] = false;
        }
        else {
            row_handled[threadIdx.x] = true;
        }

        // each thread checks each input to see if it's responsible
        __syncthreads();
        for (int i = 0; i < blockDim.x; i++) {
            for (int j = 0; j < thread_group_count; j++) {
                if (groups[i] == thread_groups[j]) {
                    if (base_idx + i < rowcount) {
                        row_handled[i] = true;
                        thread_aggregates[j] += aggregates[i];
                    }
                }
            }
        }
        idx += stride;
        base_idx += stride;

        // if no thread was responsible for the input this thread read in,
        // hand out this group to one of the unassigned threads
        __syncthreads();
        if (!row_handled[threadIdx.x]) {
            int hand_out_to_tid = atomicAdd(&handout_counter, 1) % blockDim.x;
            handout_ids[hand_out_to_tid] = threadIdx.x;
        }

        // check if we have to deal with handouts
        __syncthreads();
        if (handout_counter == prev_handout_counter) continue;

        // if our thread got handed out a group responsibility,
        // check if nobody else before us got assigned the same group
        // if not, take on responsibility, if yes, yield resp. to the lower tid
        __syncthreads();
        int handout_id = handout_ids[threadIdx.x];
        bool handout_assigned = false;
        if (handout_id != -1) {
            handout_assigned = true;
            uint64_t handed_out_group = groups[handout_id];
            int i = prev_handout_counter % blockDim.x;
            while (i != threadIdx.x) {
                int prev_handout_id = handout_ids[i];
                if (groups[prev_handout_id] == handed_out_group) {
                    atomicSub(&handout_counter, 1);
                    atomicAdd(
                        (cudaUInt64_t*)&aggregates[prev_handout_id],
                        aggregates[handout_id]);
                    handout_assigned = false;
                    break;
                }
                i++;
                if (i == blockDim.x) i = 0;
            }
        }

        // if we yielded, reset up our handout id
        // we have to do this after the actual yield since
        // other threads may still index based on it during yield detection
        __syncthreads();
        if (handout_id != -1 && handout_assigned == false) {
            handout_ids[threadIdx.x] = -1;
        }

        // if we yielded our responsibility to a lower tid, but a higher tid
        // didn't yield, steal its responsibility
        // the i'th thread gaining responsibility steals from the i'th non
        // yielding one
        __syncthreads();
        int steal_from_tid = -1;
        if (handout_id != -1) {
            int curr_handout_counter = handout_counter;
            int hci = curr_handout_counter % blockDim.x;
            if (!handout_assigned &&
                curr_handout_counter >
                    thread_group_count * blockDim.x + threadIdx.x &&
                (thread_group_count < RT_MAX_GROUPS_PER_THREAD)) {
                handout_assigned = true;
                int handout_priority = 0;
                // since the lowest tid can't yield, start from the next
                int i = (prev_handout_counter + 1) % blockDim.x;
                while (i != threadIdx.x) {
                    if (handout_ids[i] == -1) handout_priority++;
                    i++;
                    if (i == blockDim.x) i = 0;
                }
                steal_from_tid = hci;
                while (true) {
                    if (steal_from_tid == blockDim.x) {
                        steal_from_tid = 0;
                    }
                    handout_id = handout_ids[steal_from_tid];
                    if (handout_id != -1) {
                        if (handout_priority == 0) break;
                        handout_priority--;
                    }
                    steal_from_tid++;
                }
            }
        }

        // inform the tid that we stole from that it's no longer responsible
        __syncthreads();
        if (steal_from_tid != -1) {
            handout_ids[steal_from_tid] = -1;
            handout_ids[threadIdx.x] = handout_id;
        }

        // finally assign group if we weren't stolen from,
        __syncthreads();
        if (handout_assigned && handout_id != -1 &&
            handout_ids[threadIdx.x] != -1) {
            handout_ids[threadIdx.x] = -1;
            thread_groups[thread_group_count] = groups[handout_id];
            thread_aggregates[thread_group_count] = aggregates[handout_id];
            thread_group_count++;
        }

        // if the last thread in the warp was assigned a group, advance
        // to the satisfied warp phase
        prev_handout_counter = handout_counter;
        if (prev_handout_counter == MAX_GROUPS) {
            break;
        }
    }

    // satisfied warp phase
    // every thread has been assigned a group, no need to check anymore
    while (base_idx < rowcount) {
        // read input into shared memory, 1 element per thread
        if (idx < rowcount) {
            groups[threadIdx.x] = input.group_col[idx];
            aggregates[threadIdx.x] = input.aggregate_col[idx];
        }

        __syncthreads();
        for (int i = 0; i < blockDim.x; i++) {
            for (int j = 0; j < thread_group_count; j++) {
                if (groups[i] == thread_groups[j]) {
                    if (base_idx + i < rowcount) {
                        thread_aggregates[j] += aggregates[i];
                    }
                }
            }
        }

        idx += stride;
        base_idx += stride;
    }

    // the input is processed, proceed to writeout
    if (NAIVE_WRITEOUT) {
        int empty_group_slot = -1;
        for (int i = 0; i < thread_group_count; i++) {
            if (thread_groups[i] == TPG_EMPTY_GROUP_VALUE) {
                empty_group_slot = i;
            }
        }
        thread_per_group_naive_write_out<MAX_GROUP_BITS, false>(
            output, &handout_counter, thread_group_count, thread_groups,
            thread_aggregates, empty_group_slot);
    }
    else {
        for (int i = 0; i < thread_group_count; i++) {
            group_ht_insert<MAX_GROUP_BITS, true>(
                hashtable, thread_groups[i], thread_aggregates[i]);
        }
    }
}

template <int MAX_GROUP_BITS, bool NAIVE_WRITEOUT>
__global__ void kernel_thread_per_group_more_groups_old(
    db_table input, db_table output, group_ht_entry<true>* hashtable,
    int stream_count, int stream_idx)
{
    __shared__ uint64_t groups[TPG_MAX_BLOCK_DIM];
    __shared__ uint64_t aggregates[TPG_MAX_BLOCK_DIM];

    __shared__ int groups_found;
    __shared__ int rows_read;
    __shared__ bool row_consumed;

    size_t idx = threadIdx.x + (size_t)blockIdx.x * blockDim.x +
                 (size_t)stream_idx * blockDim.x * gridDim.x;

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
        thread_per_group_naive_write_out<MAX_GROUP_BITS, true>(
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
