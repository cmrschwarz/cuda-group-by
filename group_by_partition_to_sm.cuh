#pragma once
#include "cuda_group_by.cuh"
#include "group_by_global_array.cuh"

// uint64 for sm_array has 8 bytes -> take 3 bits off of shared mem bits per
#define PTSM_SM_ARRAY_OVERSIZE_SUFFICIENT                                      \
    (CUDA_SHARED_MEM_PER_BLOCK >=                                              \
     ((CUDA_MAX_BLOCK_SIZE / CUDA_WARP_SIZE) * sizeof(bucket_entry*) +         \
      (1 << (CUDA_SHARED_MEM_BITS_PER_BLOCK - 3)) *                            \
          (sizeof(uint64_t) + sizeof(bool))))

#define PTSM_SM_ARRAY_MAX_GROUP_BITS                                           \
    (PTSM_SM_ARRAY_OVERSIZE_SUFFICIENT ? CUDA_SHARED_MEM_BITS_PER_BLOCK - 3    \
                                       : CUDA_SHARED_MEM_BITS_PER_BLOCK - 4)

#define PTSM_PER_PARTITION_GROUP_BITS PTSM_SM_ARRAY_MAX_GROUP_BITS
#define PTSM_PER_PARTITION_GROUP_COUNT                                         \
    ((size_t)1 << PTSM_SM_ARRAY_MAX_GROUP_BITS)
// uint64 for bucket_pointers 8 bytes -> take 3 bits off of shared mem bits per
#define PTSM_MAX_PARTITION_COUNT_BITS (CUDA_SHARED_MEM_BITS_PER_BLOCK - 3)
#define PTSM_MAX_PARTITION_COUNT ((size_t)1 << PTSM_MAX_PARTITION_COUNT_BITS)
// 128 elements per bucket. we don't want to go too big to keep the potential
// oversize down, but we don't want to go too small to keep decent cache usage
#define PTSM_BUCKET_CAP_BITS 7
#define PTSM_BUCKET_CAP (1 << PTSM_BUCKET_CAP_BITS)
#define PTSM_BUCKET_OVERFILL_BITS CUDA_MAX_BLOCK_SIZE_BITS
#define PTSM_BUCKET_IDX_BITS (PTSM_BUCKET_CAP_BITS + PTSM_BUCKET_OVERFILL_BITS)
#define PTSM_MAX_OVERFILL (CUDA_MAX_BLOCK_SIZE_BITS - PTSM_BUCKET_CAP_BITS + 1)
#define PTSM_MAX_STREAMED_GRID_SIZE 512
union bucket_entry {
    struct bucket_entry_head {
        union bucket_entry* prev;
        size_t element_count;
    } head;
    struct bucket_entry_value {
        uint64_t group;
        uint64_t aggregate;
    } value;
};

__device__ void* ptsm_bucket_mem_pos;
bucket_entry** ptsm_bucket_heads;
void* ptsm_bucket_mem;

static inline void group_by_partition_to_sm_get_mem_requirements(
    size_t max_groups, size_t max_rows, size_t* zeroed, size_t* uninitialized)
{
    group_by_global_array_get_mem_requirements(
        max_groups, max_rows, zeroed, uninitialized);
    size_t max_group_bits = log2(max_groups);
    size_t partition_count_max = max_groups / PTSM_PER_PARTITION_GROUP_COUNT;
    if (partition_count_max > PTSM_MAX_PARTITION_COUNT) {
        partition_count_max = PTSM_MAX_PARTITION_COUNT;
    }
    size_t buckets_cap_max_oversize =
        PTSM_MAX_STREAMED_GRID_SIZE * partition_count_max * PTSM_BUCKET_CAP;
    size_t max_buckets_size_total =
        (max_rows + buckets_cap_max_oversize) * sizeof(bucket_entry);
    size_t bucket_ptrs_size = PTSM_MAX_PARTITION_COUNT * sizeof(bucket_entry*);
    *uninitialized = ceil_to_mult(*uninitialized, CUDA_MAX_CACHE_LINE_SIZE) +
                     max_buckets_size_total;
    *zeroed =
        ceil_to_mult(*zeroed, CUDA_MAX_CACHE_LINE_SIZE) + bucket_ptrs_size;
}

void group_by_partition_to_sm_init(
    size_t max_groups, size_t max_rows, void* zeroed_mem,
    void* uninitialized_mem)
{
    size_t gm_zeroed_offset;
    size_t gm_uninit_offset;
    group_by_global_array_get_mem_requirements(
        max_groups, max_rows, &gm_zeroed_offset, &gm_uninit_offset);
    group_by_global_array_init(
        max_groups, max_rows, zeroed_mem, uninitialized_mem);
    ptsm_bucket_heads = (bucket_entry**)ptradd(
        zeroed_mem, ceil_to_mult(gm_zeroed_offset, CUDA_MAX_CACHE_LINE_SIZE));
    ptsm_bucket_mem = ptradd(
        uninitialized_mem,
        ceil_to_mult(gm_uninit_offset, CUDA_MAX_CACHE_LINE_SIZE));
}

void group_by_partition_to_sm_fin()
{
}

static inline bool approach_partition_to_sm_available(
    int group_bits, size_t row_count, int grid_dim, int block_dim,
    int stream_count)
{
    if (!grid_dim || !block_dim) return false;
    if (grid_dim * (stream_count ? stream_count : 1) >
        PTSM_MAX_STREAMED_GRID_SIZE) {
        return false;
    }
    if (group_bits <= PTSM_PER_PARTITION_GROUP_BITS) return false;
    size_t max_group_bits =
        PTSM_PER_PARTITION_GROUP_BITS + PTSM_MAX_PARTITION_COUNT_BITS;
    if (group_bits > max_group_bits) return false;
    return true;
}

template <int MAX_GROUP_BITS>
__global__ void kernel_partition_to_sm(
    db_table input, void* buckets_mem, bucket_entry** bucket_ptrs,
    int stream_count, int stream_idx)
{
    constexpr int PARTITION_COUNT_UNCAPPED =
        (MAX_GROUP_BITS <= PTSM_PER_PARTITION_GROUP_BITS)
            ? 1
            : 1 << (MAX_GROUP_BITS - PTSM_PER_PARTITION_GROUP_BITS);
    constexpr int PARTITION_COUNT =
        PARTITION_COUNT_UNCAPPED > PTSM_MAX_PARTITION_COUNT
            ? 1
            : PARTITION_COUNT_UNCAPPED;
    constexpr int BUCKET_MASK = (1 << PTSM_BUCKET_IDX_BITS) - 1;
    size_t row_count = input.row_count;
    size_t base_idx = (size_t)blockIdx.x * blockDim.x +
                      (size_t)stream_idx * blockDim.x * gridDim.x;
    size_t idx = base_idx + threadIdx.x;
    size_t stride = (size_t)blockDim.x * gridDim.x * stream_count;
    __shared__ uint64_t bucket_pointers[PARTITION_COUNT];
    __shared__ bool conflicts;
    for (int i = threadIdx.x; i < PARTITION_COUNT; i += blockDim.x) {
        bucket_entry* be = (bucket_entry*)atomicAdd(
            (cudaVoidPtr_t*)&ptsm_bucket_mem_pos,
            PTSM_BUCKET_CAP * sizeof(bucket_entry));
        uint64_t diff = ptrsub(be, buckets_mem);
        // assert(diff < ((uint64_t)1 << (64 - PTSM_BUCKET_IDX_BITS)));
        bucket_pointers[i] = (diff << PTSM_BUCKET_IDX_BITS) + 1;
    }
    __syncthreads();
    // since we have syncthreads" in this loop we have to make sure
    // that all threads traverse it the same number of times
    while (base_idx < row_count) {
        bool has_conflict = false;
        int bucket_idx = 0;
        uint64_t group, aggregate;
        uint64_t ptr_idx;
        int part_idx;
        bucket_entry* bucket;
        if (idx < row_count) {
            group = input.group_col[idx];
            aggregate = input.aggregate_col[idx];
            part_idx = (int)(group >> PTSM_PER_PARTITION_GROUP_BITS);
            ptr_idx = atomicAdd((cudaUInt64_t*)&bucket_pointers[part_idx], 1);
            bucket_idx = (int)ptr_idx & BUCKET_MASK;
            bucket = (bucket_entry*)ptradd(
                buckets_mem, (ptr_idx >> PTSM_BUCKET_IDX_BITS));
            if (bucket_idx < PTSM_BUCKET_CAP) {
                bucket[bucket_idx].value.group = group;
                bucket[bucket_idx].value.aggregate = aggregate;
                /*printf(
                    "%i %i %i: filling slot %i in partid %i, bucket %llx: "
                    "%llu\n",
                    threadIdx.x, blockIdx.x, stream_idx, bucket_idx, part_idx,
                    bucket, group);*/
            }
            else {
                has_conflict = true;
                conflicts = true;
            }
        }
        __syncthreads();
        while (conflicts) {
            if (bucket_idx == PTSM_BUCKET_CAP) {
                bucket->head.element_count = PTSM_BUCKET_CAP;
                bucket->head.prev = (bucket_entry*)atomicExch(
                    (cudaVoidPtr_t*)&bucket_ptrs[part_idx],
                    (cudaVoidPtr_t)bucket);

                bucket_entry* be = (bucket_entry*)atomicAdd(
                    (cudaVoidPtr_t*)&ptsm_bucket_mem_pos,
                    PTSM_BUCKET_CAP * sizeof(bucket_entry));
                /*printf(
                    "%i %i %i, it %i: (row %llu/%llu): adding full bucket %llx "
                    "in partid %i, prev %llx; new bucket %llx\n",
                    threadIdx.x, blockIdx.x, stream_idx, it, idx, base_idx,
                    bucket, part_idx, bucket->head.prev, be);*/
                be[1].value.group = group;
                be[1].value.aggregate = aggregate;
                has_conflict = false;
                bucket_idx = 0;
                bucket_pointers[part_idx] = ((uint64_t)ptrsub(be, buckets_mem)
                                             << PTSM_BUCKET_IDX_BITS) +
                                            2;
            }
            __syncthreads();
            conflicts = false;
            __syncthreads();
            if (has_conflict) {
                ptr_idx =
                    atomicAdd((cudaUInt64_t*)&bucket_pointers[part_idx], 1);
                bucket_idx = (int)ptr_idx & BUCKET_MASK;
                bucket = (bucket_entry*)ptradd(
                    buckets_mem, (ptr_idx >> PTSM_BUCKET_IDX_BITS));
                if (bucket_idx < PTSM_BUCKET_CAP) {
                    bucket[bucket_idx].value.group = group;
                    bucket[bucket_idx].value.aggregate = aggregate;
                    has_conflict = false;
                }
                else {
                    conflicts = true;
                }
            }
            __syncthreads();
        }
        __syncthreads();
        base_idx += stride;
        idx += stride;
    }
    for (int i = threadIdx.x; i < PARTITION_COUNT; i += blockDim.x) {
        uint64_t ptr_idx = bucket_pointers[i];
        int bucket_idx = (int)ptr_idx & BUCKET_MASK;
        if (bucket_idx > 1) {
            bucket_entry* bucket = (bucket_entry*)ptradd(
                buckets_mem, (ptr_idx >> PTSM_BUCKET_IDX_BITS));
            bucket->head.element_count = bucket_idx;
            bucket->head.prev = (bucket_entry*)atomicExch(
                (cudaVoidPtr_t*)&bucket_ptrs[i], (cudaVoidPtr_t)bucket);
            /*printf(
                "%i %i %i (row %llu/%llu): adding bucket %llx in partid %i, "
                "prev %llx\n",
                threadIdx.x, blockIdx.x, stream_idx, idx, base_idx, bucket, i,
                bucket->head.prev);*/
        }
    }
}

template <int MAX_GROUP_BITS>
__global__ void kernel_partitions_into_sm_array(
    db_table output, bucket_entry** bucket_ptrs, uint64_t* global_array,
    bool* global_occurance_array, int stream_count, int stream_idx)
{
    // the ternary guards against template instantiations that would
    // cause ptxas error during compilations by requiring
    // too much shared memory even if these instantiations are never used
    constexpr int PARTITION_COUNT_UNCAPPED =
        (MAX_GROUP_BITS <= PTSM_PER_PARTITION_GROUP_BITS)
            ? 1
            : 1 << (MAX_GROUP_BITS - PTSM_PER_PARTITION_GROUP_BITS);
    constexpr int PARTITION_COUNT =
        PARTITION_COUNT_UNCAPPED > PTSM_MAX_PARTITION_COUNT
            ? 1
            : PARTITION_COUNT_UNCAPPED;
    constexpr uint64_t PARITION_MASK = PTSM_PER_PARTITION_GROUP_COUNT - 1;
    constexpr int MAX_WARP_COUNT = CUDA_MAX_BLOCK_SIZE / CUDA_WARP_SIZE;
    __shared__ uint64_t shared_mem_array[PTSM_PER_PARTITION_GROUP_COUNT];
    __shared__ bool shared_mem_array_occurance[PTSM_PER_PARTITION_GROUP_COUNT];
    __shared__ bucket_entry* warp_buckets[MAX_WARP_COUNT];

    __shared__ bool bucket_aquired;

    int warp_id = threadIdx.x / CUDA_WARP_SIZE;
    int warp_offset = threadIdx.x % CUDA_WARP_SIZE;
    bool warp_leader = warp_offset == 0;
    int partition_stride = gridDim.x * stream_count;
    for (int part_id = (blockIdx.x + stream_idx * gridDim.x) % PARTITION_COUNT;
         part_id < PARTITION_COUNT; part_id += partition_stride) {
        if (threadIdx.x == 0) bucket_aquired = false;
        __syncthreads();
        if (warp_leader) {
            bucket_entry* be = bucket_ptrs[part_id];
            while (true) {
                if (!be) break;
                bucket_entry* be_prev = (bucket_entry*)atomicCAS(
                    (cudaVoidPtr_t*)&bucket_ptrs[part_id], (cudaVoidPtr_t)be,
                    (cudaVoidPtr_t)be->head.prev);
                if (be == be_prev) break;
                be = be_prev;
            }
            warp_buckets[warp_id] = be;
            if (be) bucket_aquired = true;
        }
        __syncthreads();
        if (!bucket_aquired) continue;
        for (int i = threadIdx.x; i < PTSM_PER_PARTITION_GROUP_COUNT;
             i += blockDim.x) {
            shared_mem_array[i] = 0;
            shared_mem_array_occurance[i] = false;
        }
        __syncthreads();
        while (true) {
            bucket_entry* be = warp_buckets[warp_id];
            if (!be) break;
            assert(be->head.element_count <= PTSM_BUCKET_CAP);
            /*   printf(
                "%i %i %i: err: partid %i, bucket %llx has cap %llu\n",
                   threadIdx.x, blockIdx.x, stream_idx, part_id, be,
                   be->head.element_count);
                   */
            //+1 since the first element is the head
            for (int i = warp_offset + 1; i < be->head.element_count;
                 i += CUDA_WARP_SIZE) {
                uint64_t group = be[i].value.group & PARITION_MASK;
                // assert(be[i].value.group >> PTSM_PARTITION_GROUP_BITS ==
                //   part_id);
                /* printf(
                     "%i %i %i: err: slot %i in partid %i, bucket %llx: "
                     "%llu\n",
                     threadIdx.x, blockIdx.x, stream_idx, i, part_id, be,
                     be[i].value.group);*/

                atomicAdd(
                    (cudaUInt64_t*)&shared_mem_array[group],
                    be[i].value.aggregate);
                shared_mem_array_occurance[group] = true;
            }
            __syncwarp();
            if (warp_leader) {
                be = bucket_ptrs[part_id];
                while (true) {
                    if (!be) break;
                    bucket_entry* be_prev = (bucket_entry*)atomicCAS(
                        (cudaVoidPtr_t*)&bucket_ptrs[part_id],
                        (cudaVoidPtr_t)be, (cudaVoidPtr_t)be->head.prev);
                    if (be == be_prev) break;
                    be = be_prev;
                }
                /*if (be) {
                  printf(
                       "%i %i %i: aquiring bucket %llx with partid %i, prev:
                "
                       "%llx"
                       "\n",
                       threadIdx.x, blockIdx.x, stream_idx, be, part_id,
                       be->head.prev);
                }*/
                warp_buckets[warp_id] = be;
            }
            __syncwarp();
        }
        __syncthreads();
        for (uint64_t i = threadIdx.x; i < PTSM_PER_PARTITION_GROUP_COUNT;
             i += blockDim.x) {
            if (!shared_mem_array_occurance[i]) continue;
            global_array_insert<true>(
                global_array, global_occurance_array,
                ((uint64_t)part_id << PTSM_PER_PARTITION_GROUP_BITS) + i,
                shared_mem_array[i]);
        }
    }
}

template <int MAX_GROUP_BITS>
void group_by_partition_to_sm(
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
    // reset buffer memory
    cudaMemcpyToSymbol(
        ptsm_bucket_mem_pos, &ptsm_bucket_mem, sizeof(void*), 0,
        cudaMemcpyHostToDevice);
    // for stream_count 0 we use the default stream,
    // but thats actually still one stream not zero
    int actual_stream_count = stream_count ? stream_count : 1;
    for (int i = 0; i < actual_stream_count; i++) {
        cudaStream_t stream = stream_count ? streams[i] : 0;
        kernel_partition_to_sm<MAX_GROUP_BITS>
            <<<grid_dim, block_dim, 0, stream>>>(
                gd->input, ptsm_bucket_mem, ptsm_bucket_heads,
                actual_stream_count, i);

        // if we have only one stream there is no need for waiting events
        if (stream_count > 1) cudaEventRecord(events[stream_count + i], stream);
    }
    for (int i = 0; i < actual_stream_count; i++) {
        cudaStream_t stream = stream_count ? streams[i] : 0;
        if (stream_count > 1) {
            // every write out kernel needs to wait on every fill kernel
            for (int j = 0; j < stream_count; j++) {
                // the stream doesn't need to wait on itself
                if (j == i) continue;
                cudaStreamWaitEvent(stream, events[stream_count + j], 0);
            }
        }
        kernel_partitions_into_sm_array<MAX_GROUP_BITS>
            <<<grid_dim, block_dim, 0, stream>>>(
                gd->output, ptsm_bucket_heads, global_array,
                global_array_occurance_flags, actual_stream_count, i);
        if (stream_count > 1) cudaEventRecord(events[i], stream);
    }
    group_by_global_array_writeout<MAX_GROUP_BITS>(
        gd, grid_dim, block_dim, stream_count, streams, events, start_event,
        end_event);
}
