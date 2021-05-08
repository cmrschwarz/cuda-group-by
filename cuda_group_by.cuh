#pragma once
#define __STDC_FORMAT_MACROS
#include <inttypes.h>
#include <cstddef>
#include <cstdint>
#include <algorithm>
#include <cuda.h>
#include "utils.cuh"

// we need this stupid type since c++ doesn't see
// uint64_t and unsigned long long int as the same type
// (even though they are on literally any plattform)
// this causes e.g. atomicAdd to complain about parameter types
typedef unsigned long long int cudaUInt64_t;
#define CUDA_WARP_SIZE 32
#define CUDA_WARP_SIZE_BITS 5
#define CUDA_MAX_BLOCK_SIZE 1024
#define CUDA_MAX_BLOCK_SIZE_BITS 10
#define CUDA_L1_CACHE_LINE_SIZE 128
#define CUDA_L2_CACHE_LINE_SIZE 32
#define CUDA_MAX_CACHE_LINE_SIZE CUDA_L1_CACHE_LINE_SIZE

// 48 Kilobytes of shared memory per block
#define CUDA_SHARED_MEM_PER_BLOCK 0xC000

// -> (rounded down) 15 bits of shared memory
#define CUDA_SHARED_MEM_BITS_PER_BLOCK 15

struct db_table {
    uint64_t* group_col;
    uint64_t* aggregate_col;
    size_t row_count;
};

struct gpu_data {
    db_table input;
    db_table output;
};

static inline void alloc_db_table_gpu(db_table* t, uint64_t row_count)
{
    CUDA_TRY(cudaMalloc(&t->group_col, row_count * sizeof(uint64_t)));
    CUDA_TRY(cudaMalloc(&t->aggregate_col, row_count * sizeof(uint64_t)));
    t->row_count = row_count;
}

static inline void free_db_table_gpu(db_table* t)
{
    CUDA_TRY(cudaFree(t->group_col));
    CUDA_TRY(cudaFree(t->aggregate_col));
}

static inline void
gpu_data_alloc(gpu_data* gd, size_t max_groups, size_t max_rows)
{
    alloc_db_table_gpu(&gd->input, max_rows);
    alloc_db_table_gpu(&gd->output, max_groups);
}

static inline void gpu_data_free(gpu_data* gd)
{
    free_db_table_gpu(&gd->output);
    free_db_table_gpu(&gd->input);
}

static inline size_t ceil_to_pow_two(size_t v)
{
    return v == 1 ? 1 : 1 << (8 * sizeof(size_t) - __builtin_clzl(v - 1));
}

static inline size_t ceil_to_mult(size_t v, size_t mult)
{
    if (v % mult != 0) v += mult - (v % mult);
    return v;
}

static inline void* ptradd(void* ptr, size_t val)
{
    return (void*)(((char*)ptr) + val);
}

static inline int log2(size_t v)
{
    return 8 * sizeof(size_t) - __builtin_clzll((v)) - 1;
}
