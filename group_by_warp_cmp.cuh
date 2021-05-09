#pragma once
#include "cuda_group_by.cuh"
#include <type_traits>
#include "group_by_hashtable.cuh"

#define WCMP_MAX_GROUPS WCMP_CUDA_WARP_SIZE
// the minimum number of groups where we use the
// kernel_thread_per_group_more_groups kernel instead of the
// more_threads version

#define WCMP_MIN_BLOCK_DIM 32
#define WCMP_MAX_BLOCK_DIM 1024

GROUP_BY_HASHTABLE_FORWARD_REQUIREMENTS(group_by_warp_cmp)

static inline bool approach_warp_cmp_available(
    int group_bits, size_t row_count, int grid_dim, int block_dim,
    int stream_count)
{
    const size_t group_count = (1 << group_bits);
    if (!grid_dim || !block_dim) return false;
    if (group_count > CUDA_WARP_SIZE) return false;
    if (block_dim < CUDA_WARP_SIZE) return false;
    if (block_dim > CUDA_MAX_BLOCK_SIZE) return false;
    return true;
}

template <int MAX_GROUP_BITS>
__global__ void kernel_warp_cmp(
    db_table input, db_table output, group_ht_entry<>* hashtable,
    int stream_count, int stream_idx)
{
    size_t base_idx = (size_t)blockIdx.x * blockDim.x +
                      (size_t)stream_idx * blockDim.x * gridDim.x;
    size_t idx = threadIdx.x + base_idx;
    size_t stride = (size_t)blockDim.x * gridDim.x * stream_count;
    constexpr int BATCH_SIZE_UNCAPPED = (1 << MAX_GROUP_BITS);
    // this number is one of {2,4,8,16,32}
    constexpr int BATCH_SIZE = (BATCH_SIZE_UNCAPPED <= CUDA_WARP_SIZE)
                                   ? BATCH_SIZE_UNCAPPED
                                   : CUDA_WARP_SIZE;
    int batch_idx = threadIdx.x / BATCH_SIZE;
    int batch_base = batch_idx * BATCH_SIZE;
    int next_batch_base = batch_base + BATCH_SIZE;
    bool group_assigned = false;
    uint64_t assigned_group;
    uint64_t assigned_aggregate;
    __shared__ uint64_t groups[CUDA_MAX_BLOCK_SIZE];
    __shared__ uint64_t aggregates[CUDA_MAX_BLOCK_SIZE];
    __shared__ bool row_handled[CUDA_MAX_BLOCK_SIZE];
    __shared__ int handout_ids[CUDA_MAX_BLOCK_SIZE];
    __shared__ int handout_counters[CUDA_MAX_BLOCK_SIZE / BATCH_SIZE];

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
    if (group_assigned) {
        group_ht_insert<MAX_GROUP_BITS>(
            hashtable, assigned_group, assigned_aggregate);
    }
}

template <int MAX_GROUP_BITS>
void group_by_warp_cmp(
    gpu_data* gd, int grid_dim, int block_dim, int stream_count,
    cudaStream_t* streams, cudaEvent_t* events, cudaEvent_t start_event,
    cudaEvent_t end_event)
{
    int actual_stream_count = stream_count ? stream_count : 1;
    uint64_t zero = 0;

    CUDA_TRY(cudaEventRecord(start_event));
    cudaMemcpyToSymbol(
        group_ht_groups_found, &zero, sizeof(zero), 0, cudaMemcpyHostToDevice);
    for (int i = 0; i < actual_stream_count; i++) {
        cudaStream_t stream = stream_count ? streams[i] : 0;
        kernel_warp_cmp<MAX_GROUP_BITS><<<grid_dim, block_dim, 0, stream>>>(
            gd->input, gd->output, group_ht_entry<>::table, actual_stream_count,
            i);

        if (stream_count > 1) cudaEventRecord(events[i], stream);
    }
    group_by_hashtable_writeout<MAX_GROUP_BITS>(
        gd, grid_dim, block_dim, stream_count, streams, events, start_event,
        end_event);
}
