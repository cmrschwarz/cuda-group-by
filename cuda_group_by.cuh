#pragma once
#include <cstddef>
#include <cstdint>
#include "utils.cuh"

// we need this stupid type since c++ doesn't see
// uint64_t and unsigned long long int as the same type
// (even though they are on literally any plattform)
// this causes e.g. atomicAdd to complain about parameter types
typedef unsigned long long int cudaUInt64_t;

struct db_table {
    uint64_t* group_col;
    uint64_t* aggregate_col;
    size_t row_count;
};

struct group_ht_entry {
    uint64_t group;
    uint64_t aggregate;
    size_t output_idx;
};
static inline void group_ht_alloc(group_ht_entry** group_ht, size_t max_groups);
static inline void group_ht_free(group_ht_entry* group_ht);

struct gpu_data {
    db_table input;
    db_table output;
    group_ht_entry* group_ht;
};

#define GPU_MAX_THREADS_PER_BLOCK 1024

__device__ size_t groups_found = 0;

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
gpu_data_alloc(gpu_data* gd, size_t max_rows, size_t max_groups)
{
    alloc_db_table_gpu(&gd->input, max_rows);
    alloc_db_table_gpu(&gd->output, max_groups);
    group_ht_alloc(&gd->group_ht, max_groups);
}

static inline void gpu_data_free(gpu_data* gd)
{
    group_ht_free(gd->group_ht);
    free_db_table_gpu(&gd->output);
    free_db_table_gpu(&gd->input);
}
