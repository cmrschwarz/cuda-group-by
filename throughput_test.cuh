#pragma once

#include "cuda_group_by.cuh"

__device__ uint64_t throughput_test_value_sink = 0;

static inline bool approach_throughput_test_available(
    int group_bits, size_t row_count, int grid_dim, int block_dim,
    int stream_count)
{
    if (!grid_dim || !block_dim) return false;
    return true;
}

__global__ void
kernel_throughput_test(db_table input, int stream_count, int stream_idx)
{
    int stride = blockDim.x * gridDim.x * stream_count;
    int tid = threadIdx.x + blockIdx.x * blockDim.x +
              stream_idx * blockDim.x * gridDim.x;
    uint64_t group_sum = 0;
    uint64_t aggregate_sum = 0;
    for (size_t i = tid; i < input.row_count; i += stride) {
        group_sum += input.group_col[i];
        aggregate_sum += input.aggregate_col[i];
    }
    // make sure the sums are used so the optimizer doesn't get trigger happy
    atomicAdd(
        (cudaUInt64_t*)&throughput_test_value_sink, group_sum - aggregate_sum);
}

template <int MAX_GROUP_BITS>
void throughput_test(
    gpu_data* gd, int grid_dim, int block_dim, int stream_count,
    cudaStream_t* streams, cudaEvent_t* events, cudaEvent_t start_event,
    cudaEvent_t end_event)
{

    CUDA_TRY(cudaEventRecord(start_event));
    int actual_stream_count = stream_count ? stream_count : 1;
    for (int i = 0; i < actual_stream_count; i++) {
        cudaStream_t stream = stream_count ? streams[i] : 0;
        kernel_throughput_test<<<grid_dim, block_dim, 0, stream>>>(
            gd->input, actual_stream_count, i);
        if (stream_count > 1) {
            cudaEventRecord(events[i], stream);
        }
    }
    if (stream_count > 1) {
        for (int i = 0; i < stream_count; i++) {
            cudaStreamWaitEvent(0, events[i], 0);
        }
    }
    CUDA_TRY(cudaEventRecord(end_event));
    CUDA_TRY(cudaGetLastError());
}
