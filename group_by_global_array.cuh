#pragma once
#include "cuda_group_by.cuh"

#if !GROUP_COUNT_EQUALS_GROUP_MAX_VAL
#    error "global_array requires GROUP_COUNT_EQUALS_GROUP_MAX_VAL to be true"
#endif

#ifdef CUDA_GROUP_BY_CMAKE_BUILD
#    include <cub/cub.cuh>
#else
// always use the submodule version when we are not building with cmake
// and don't have a proper include path setup
#    include "./deps/cub/cub/cub.cuh"
#endif

uint64_t* global_array = nullptr;
void* cub_flagged_temp_storage = nullptr;
void* cub_flagged_temp_storage_2 = nullptr;
size_t cub_flagged_temp_storage_size;
bool* global_array_occurance_flags;
__device__ cudaUInt64_t global_array_groups_found;
cudaUInt64_t* global_array_groups_found_dev_ptr;

static inline void group_by_global_array_init(size_t max_groups)
{
    if (global_array) return;
    CUDA_TRY(cudaMalloc(&global_array, max_groups * sizeof(uint64_t)));
    CUDA_TRY(
        cudaMalloc(&global_array_occurance_flags, max_groups * sizeof(bool)));
    CUDA_TRY(cudaMemset(global_array, 0, max_groups * sizeof(uint64_t)));
    CUDA_TRY(cudaMemset(
        global_array_occurance_flags, false, max_groups * sizeof(bool)));
    cub::DeviceSelect::Flagged(
        NULL, cub_flagged_temp_storage_size, (uint64_t*)NULL, (bool*)NULL,
        (uint64_t*)NULL, (size_t*)NULL, max_groups, 0, false);
    CUDA_TRY(
        cudaMalloc(&cub_flagged_temp_storage, cub_flagged_temp_storage_size));
    CUDA_TRY(
        cudaMalloc(&cub_flagged_temp_storage_2, cub_flagged_temp_storage_size));
    CUDA_TRY(cudaGetSymbolAddress(
        (void**)&global_array_groups_found_dev_ptr, global_array_groups_found));
}
static inline void group_by_global_array_fin()
{
    if (!global_array) return;
    CUDA_TRY(cudaFree(cub_flagged_temp_storage_2));
    CUDA_TRY(cudaFree(cub_flagged_temp_storage));
    CUDA_TRY(cudaFree(global_array));
    CUDA_TRY(cudaFree(global_array_occurance_flags));
    global_array = NULL;
}

template <bool OPTIMISTIC>
__device__ void global_array_insert(
    uint64_t* array, bool* occurrance_array, uint64_t group, uint64_t aggregate)
{
    atomicAdd((cudaUInt64_t*)(array + group), aggregate);
    if (OPTIMISTIC) {
        if (!occurrance_array[group]) occurrance_array[group] = true;
    }
    else {
        occurrance_array[group] = true;
    }
}

template <int MAX_GROUP_BITS, bool OPTIMISTIC>
__global__ void kernel_fill_global_array(
    db_table input, uint64_t* array, bool* occurrance_array, int stream_count,
    int stream_idx)
{
    int stride = blockDim.x * gridDim.x * stream_count;
    int tid = threadIdx.x + blockIdx.x * blockDim.x +
              stream_idx * blockDim.x * gridDim.x;
    for (size_t i = tid; i < input.row_count; i += stride) {
        uint64_t group = input.group_col[i];
        uint64_t agg = input.aggregate_col[i];
        global_array_insert<OPTIMISTIC>(array, occurrance_array, group, agg);
    }
}

template <int MAX_GROUP_BITS>
__global__ void kernel_write_out_global_array(
    db_table output, uint64_t* array, bool* occurance_array, int stream_count,
    int stream_idx)
{
    constexpr size_t ARRAY_SIZE = ((size_t)1) << MAX_GROUP_BITS;
    int tid = threadIdx.x + blockIdx.x * blockDim.x +
              stream_idx * blockDim.x * gridDim.x;
    int stride = blockDim.x * gridDim.x * stream_count;
    for (size_t i = tid; i < ARRAY_SIZE; i += stride) {
        if (!occurance_array[i]) continue;
        size_t out_idx = atomicAdd(&global_array_groups_found, 1);
        output.group_col[out_idx] = i;
        output.aggregate_col[out_idx] = array[i];
        occurance_array[i] = false;
        array[i] = 0;
    }
}

static inline bool approach_global_array_available(
    int group_bits, int row_count, int grid_dim, int block_dim,
    int stream_count)
{
    if (!grid_dim || !block_dim) return false;
    return true;
}

struct cub_group_iterator {
    size_t i = 0;
    __device__ size_t operator[](size_t idx)
    {
        return i + idx;
    }
    __device__ size_t operator==(cub_group_iterator rhs)
    {
        return i == rhs.i;
    }
    __device__ cub_group_iterator operator+(size_t idx)
    {
        return cub_group_iterator{i + idx};
    }
    __device__ cub_group_iterator& operator++()
    {
        i++;
        return *this;
    }
    __device__ size_t operator*()
    {
        return i;
    }
};
template <> struct std::iterator_traits<cub_group_iterator> {
    typedef size_t value_type;
    typedef int64_t difference_type;
    typedef size_t* pointer;
    typedef size_t& reference;
    typedef std::input_iterator_tag iterator_category;
};

template <int MAX_GROUP_BITS, bool COMPRESSTORE = true>
void group_by_global_array_writeout(
    gpu_data* gd, int grid_dim, int block_dim, int stream_count,
    cudaStream_t* streams, cudaEvent_t* events, cudaEvent_t start_event,
    cudaEvent_t end_event)
{
    size_t group_count = (size_t)1 << MAX_GROUP_BITS;
    if (COMPRESSTORE) {
        for (int i = 0; i < 2; i++) {
            cudaStream_t stream = stream_count ? streams[i] : 0;
            if (stream_count > 1) {
                // every write out kernel needs to wait on every fill kernel
                for (int j = 0; j < stream_count; j++) {
                    // the stream doesn't need to wait on itself
                    if (j == i) continue;
                    cudaStreamWaitEvent(stream, events[j], 0);
                }
            }
        }
        size_t mem;
        cub::DeviceSelect::Flagged(
            NULL, mem, (uint64_t*)NULL, (bool*)NULL, (uint64_t*)NULL,
            (size_t*)NULL, group_count, 0, false);
        RELASE_ASSERT(mem <= cub_flagged_temp_storage_size);
        cub::DeviceSelect::Flagged(
            cub_flagged_temp_storage, cub_flagged_temp_storage_size,
            cub_group_iterator{}, global_array_occurance_flags,
            gd->output.group_col, global_array_groups_found_dev_ptr,
            group_count, (stream_count > 1) ? streams[0] : 0, false);
        cub::DeviceSelect::Flagged(
            cub_flagged_temp_storage_2, cub_flagged_temp_storage_size,
            global_array, global_array_occurance_flags,
            gd->output.aggregate_col, global_array_groups_found_dev_ptr,
            group_count, (stream_count > 1) ? streams[1] : 0, false);
        cudaMemsetAsync(
            global_array, 0, group_count * sizeof(uint64_t),
            (stream_count > 1) ? streams[1] : 0);
        cudaMemset(global_array_occurance_flags, 0, group_count * sizeof(bool));
    }
    else {
        int actual_stream_count = stream_count ? stream_count : 1;
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
            kernel_write_out_global_array<MAX_GROUP_BITS>
                <<<grid_dim, block_dim, 0, stream>>>(
                    gd->output, global_array, global_array_occurance_flags,
                    actual_stream_count, i);
        }
    }
    CUDA_TRY(cudaEventRecord(end_event));
    CUDA_TRY(cudaGetLastError());
    cudaMemcpyFromSymbol(
        &gd->output.row_count, global_array_groups_found, sizeof(size_t), 0,
        cudaMemcpyDeviceToHost);
}

template <int MAX_GROUP_BITS, bool OPTIMISTIC, bool COMPRESSTORE>
void group_by_global_array(
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
    // for stream_count 0 we use the default stream,
    // but thats actually still one stream not zero
    int actual_stream_count = stream_count ? stream_count : 1;
    for (int i = 0; i < actual_stream_count; i++) {
        cudaStream_t stream = stream_count ? streams[i] : 0;
        kernel_fill_global_array<MAX_GROUP_BITS, OPTIMISTIC>
            <<<grid_dim, block_dim, 0, stream>>>(
                gd->input, global_array, global_array_occurance_flags,
                actual_stream_count, i);
        // if we have only one stream there is no need for waiting events
        if (stream_count > 1) cudaEventRecord(events[i], stream);
    }
    group_by_global_array_writeout<MAX_GROUP_BITS, COMPRESSTORE>(
        gd, grid_dim, block_dim, stream_count, streams, events, start_event,
        end_event);
}
