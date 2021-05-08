#pragma once
#include "group_by_hashtable.cuh"

#define SHARED_MEM_HT_ENTRY_BITS 4
#define SHARED_MEM_HT_EMPTY_GROUP_VAL ((uint64_t)-3)
#define SHARED_MEM_HT_OVERSIZE_BITS 1
#define SHARED_MEM_HT_MAX_GROUP_BITS                                           \
    (CUDA_SHARED_MEM_BITS_PER_BLOCK - SHARED_MEM_HT_OVERSIZE_BITS -            \
     SHARED_MEM_HT_ENTRY_BITS)

struct shared_mem_ht_entry {
    uint64_t group;
    uint64_t aggregate;
};

GROUP_BY_HASHTABLE_FORWARD_REQUIREMENTS(group_by_shared_mem_hashtable)

static inline bool approach_shared_mem_hashtable_available(
    int group_bits, int row_count, int grid_dim, int block_dim,
    int stream_count)
{
    if (!grid_dim || !block_dim) return false;
    return group_bits <= SHARED_MEM_HT_MAX_GROUP_BITS;
}

template <int MAX_GROUP_BITS>
__global__ void kernel_shared_mem_ht(
    db_table input, group_ht_entry<>* hashtable, int stream_count,
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
__global__ void kernel_shared_mem_ht_optimistic(
    db_table input, group_ht_entry<>* hashtable, int stream_count,
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
    constexpr size_t MAX_GROUPS = ((size_t)1 << MAX_GROUP_BITS);

    __shared__ int empty_group_used;
    __shared__ uint64_t empty_group_aggregate;
    __shared__ shared_mem_ht_entry shared_mem_ht[SHARED_MEM_HT_CAPACITY];
    __shared__ size_t groups_found;
    __shared__ size_t collisions;

    int tid = threadIdx.x + blockIdx.x * blockDim.x +
              stream_idx * blockDim.x * gridDim.x;
    int stride = blockDim.x * gridDim.x * stream_count;

    if (threadIdx.x == 0) {
        groups_found = 0;
        collisions = 0;
        empty_group_used = 0;
        empty_group_aggregate = 0;
    }

    for (int i = threadIdx.x; i < SHARED_MEM_HT_CAPACITY; i += blockDim.x) {
        shared_mem_ht[i].group = SHARED_MEM_HT_EMPTY_GROUP_VAL;
        shared_mem_ht[i].aggregate = 0;
    }
    __syncthreads();
    size_t i = tid;
    size_t last_check = 0;
    for (; i < input.row_count; i += stride) {
        uint64_t group = input.group_col[i];
        uint64_t aggregate = input.aggregate_col[i];
        if (group == SHARED_MEM_HT_EMPTY_GROUP_VAL) {
            if (empty_group_used == 0) {
                if (atomicCAS(&empty_group_used, 0, 1) == 0) {
                    atomicAdd((cudaUInt64_t*)&groups_found, 1);
                }
            }
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
                if (found == group) break;
                if (found == SHARED_MEM_HT_EMPTY_GROUP_VAL) {
                    atomicAdd((cudaUInt64_t*)&groups_found, 1);
                    if (hte != &shared_mem_ht[group & SHARED_MEM_HT_MASK]) {
                        atomicAdd((cudaUInt64_t*)&collisions, 1);
                    }
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
        if (i - last_check > MAX_GROUPS) {
            last_check = i;
            if (groups_found == MAX_GROUPS) {
                i += stride;
                break;
            }
        }
    }
    // all existing groups found. use optimized codepath
    if (empty_group_used) {
        for (; i < input.row_count; i += stride) {
            uint64_t group = input.group_col[i];
            uint64_t aggregate = input.aggregate_col[i];
            if (group == SHARED_MEM_HT_EMPTY_GROUP_VAL) {
                atomicAdd((cudaUInt64_t*)&empty_group_aggregate, aggregate);
                continue;
            }
            shared_mem_ht_entry* hte =
                &shared_mem_ht[group & SHARED_MEM_HT_MASK];
            while (true) {
                if (hte->group == group) break;
                if (hte != &shared_mem_ht[SHARED_MEM_HT_CAPACITY - 1]) {
                    hte++;
                }
                else {
                    hte = &shared_mem_ht[0];
                }
            }
            atomicAdd((cudaUInt64_t*)&hte->aggregate, aggregate);
        }
    }
    else if (collisions) {
        for (; i < input.row_count; i += stride) {
            uint64_t group = input.group_col[i];
            shared_mem_ht_entry* hte =
                &shared_mem_ht[group & SHARED_MEM_HT_MASK];
            while (true) {
                if (hte->group == group) break;
                if (hte != &shared_mem_ht[SHARED_MEM_HT_CAPACITY - 1]) {
                    hte++;
                }
                else {
                    hte = &shared_mem_ht[0];
                }
            }
            atomicAdd((cudaUInt64_t*)&hte->aggregate, input.aggregate_col[i]);
        }
    }
    else {
        for (; i < input.row_count; i += stride) {
            uint64_t group = input.group_col[i];
            shared_mem_ht_entry* hte =
                &shared_mem_ht[group & SHARED_MEM_HT_MASK];
            atomicAdd((cudaUInt64_t*)&hte->aggregate, input.aggregate_col[i]);
        }
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

template <int MAX_GROUP_BITS, bool OPTIMISTIC>
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
        if (OPTIMISTIC) {
            kernel_shared_mem_ht_optimistic<MAX_GROUP_BITS>
                <<<grid_dim, block_dim, 0, stream>>>(
                    gd->input, group_ht_entry<>::table, actual_stream_count, i);
        }
        else {
            kernel_shared_mem_ht<MAX_GROUP_BITS>
                <<<grid_dim, block_dim, 0, stream>>>(
                    gd->input, group_ht_entry<>::table, actual_stream_count, i);
        }
        // if we have only one stream there is no need for waiting events
        if (stream_count > 1) cudaEventRecord(events[i], stream);
    }
    group_by_hashtable_writeout<MAX_GROUP_BITS>(
        gd, grid_dim, block_dim, stream_count, streams, events, start_event,
        end_event);
}
