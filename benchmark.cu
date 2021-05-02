#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <random>
#include <limits>
#include <iostream>
#include <fstream>
#include <iomanip>

// to disable openmp even if available
#define DONT_WANT_OPENMP false
// to disable pinning of the output buffer
#define DONT_WANT_PINNED_MEM false
// set to false to reduce data size for debugging
#define BIG_DATA false
// use small group values to ease debugging
#define SMALL_GROUP_VALS true
// enforce this assumption so we can use the insert_by family of approaches
#define GROUP_COUNT_EQUALS_GROUP_MAX_VAL true
// use small aggregate values to ease debugging
#define SMALL_AGGREGATE_VALS true
// continue in case of a validation failiure
#define ALLOW_FAILIURE false
// disable actual validation and just say "PASS"
#define VALIDATION_OFF false

#if defined(_OPENMP) && !(DONT_WANT_OPENMP)
#    include <omp.h>
#    define USE_OPENMP true
#else
#    define USE_OPENMP false
#endif

#define ENABLE_APPROACH_HASHTABLE false
#define ENABLE_APPROACH_SHARED_MEM_HASHTABLE false
#define ENABLE_APPROACH_PER_THREAD_HASHTABLE false
#define ENABLE_APPROACH_WARP_CMP false
#define ENABLE_APPROACH_BLOCK_CMP false
#define ENABLE_APPROACH_CUB_RADIX_SORT false
#define ENABLE_APPROACH_THROUGHPUT_TEST true
#define ENABLE_APPROACH_SHARED_MEM_PERFECT_HASHTABLE false

#define ENABLE_APPROACH_GLOBAL_ARRAY true
#define ENABLE_APPROACH_SHARED_MEM_ARRAY false
#define ENABLE_APPROACH_PER_THREAD_ARRAY false

#define ENABLE_HASHTABLE_EAGER_OUT_IDX false
#define ENABLE_BLOCK_CMP_NAIVE_WRITEOUT false
#define ENABLE_BLOCK_CMP_OLD false

#if ENABLE_APPROACH_HASHTABLE
#    include "group_by_hashtable.cuh"
#endif

#if ENABLE_APPROACH_WARP_CMP
#    include "group_by_warp_cmp.cuh"
#endif

#if ENABLE_APPROACH_BLOCK_CMP
#    include "group_by_block_cmp.cuh"
#endif

#if ENABLE_APPROACH_SHARED_MEM_HASHTABLE
#    include "group_by_shared_mem_hashtable.cuh"
#endif

#if ENABLE_APPROACH_SHARED_MEM_PERFECT_HASHTABLE
#    include "group_by_shared_mem_perfect_hashtable.cuh"
#endif

#if ENABLE_APPROACH_PER_THREAD_HASHTABLE
#    include "group_by_per_thread_hashtable.cuh"
#endif

#if ENABLE_APPROACH_CUB_RADIX_SORT
#    include "group_by_cub_radix_sort.cuh"
#endif

#if ENABLE_APPROACH_THROUGHPUT_TEST
#    include "throughput_test.cuh"
#endif

#if ENABLE_APPROACH_GLOBAL_ARRAY
#    include "group_by_global_array.cuh"
#endif

#if ENABLE_APPROACH_SHARED_MEM_ARRAY
#    include "group_by_shared_mem_array.cuh"
#endif

#if ENABLE_APPROACH_PER_THREAD_ARRAY
#    include "group_by_per_thread_array.cuh"
#endif

#if BIG_DATA
#    define ITERATION_COUNT 5
#else
#    define ITERATION_COUNT 3
#endif
#if BIG_DATA
#    define BENCHMARK_STREAMS_MAX 8
const size_t benchmark_stream_count_variants[] = {0, 4, BENCHMARK_STREAMS_MAX};
#else
#    define BENCHMARK_STREAMS_MAX 4
const size_t benchmark_stream_count_variants[] = {0, BENCHMARK_STREAMS_MAX};
#endif

#if BIG_DATA
// 2^26, 16 Byte per entry -> 1 GiB data
#    define BENCHMARK_ROWS_BITS_MAX 26
#    define BENCHMARK_ROWS_MAX ((size_t)1 << BENCHMARK_ROWS_BITS_MAX)
const size_t benchmark_row_count_variants[] = {
    1024, 131072, BENCHMARK_ROWS_MAX / 2, BENCHMARK_ROWS_MAX};
#else
#    define BENCHMARK_ROWS_BITS_MAX 26
#    define BENCHMARK_ROWS_MAX ((size_t)1 << BENCHMARK_ROWS_BITS_MAX)
const size_t benchmark_row_count_variants[] = {
    32, 128, 1024, 16384, 131072, BENCHMARK_ROWS_MAX / 2, BENCHMARK_ROWS_MAX};
// const size_t benchmark_row_count_variants[] = {BENCHMARK_ROWS_MAX};
#endif

#if BIG_DATA
const int benchmark_gpu_block_dim_variants[] = {0, 32, 64, 128, 256, 512, 1024};
#else
const int benchmark_gpu_block_dim_variants[] = {0, 32, 128, 512};
#endif

#if BIG_DATA
const int benchmark_gpu_grid_dim_variants[] = {0,   32,   64,   128,  256,
                                               512, 1024, 2048, 4096, 8192};
#else
const int benchmark_gpu_grid_dim_variants[] = {0, 128, 512};
#endif

#if BIG_DATA
#    define BENCHMARK_GROUP_BITS_MAX BENCHMARK_ROWS_BITS_MAX
#else
#    define BENCHMARK_GROUP_BITS_MAX 26
#endif

#if SMALL_GROUP_VALS
#    define BENCHMARK_GROUP_VALS_MIN 0
#    define BENCHMARK_GROUP_VALS_MAX                                           \
        (((size_t)1 << BENCHMARK_GROUP_BITS_MAX) - 1)
#else
#    define BENCHMARK_GROUP_VALS_MIN std::numeric_limits<uint64_t>::min()
#    define BENCHMARK_GROUP_VALS_MAX std::numeric_limits<uint64_t>::max()
#endif

#if SMALL_AGGREGATE_VALS
#    define BENCHMARK_AGGREGATE_VALS_MIN 0
#    define BENCHMARK_AGGREGATE_VALS_MAX 100
#else
#    define BENCHMARK_AGGREGATE_VALS_MIN std::numeric_limits<uint64_t>::min()
#    define BENCHMARK_AGGREGATE_VALS_MAX std::numeric_limits<uint64_t>::max()
#endif

#define BENCHMARK_GROUPS_MAX ((size_t)1 << BENCHMARK_GROUP_BITS_MAX)

#define BENCHMARK_GPU_GRID_DIM_VARIANT_COUNT                                   \
    ARRAY_SIZE(benchmark_gpu_grid_dim_variants)
#define BENCHMARK_GPU_BLOCK_DIM_VARIANT_COUNT                                  \
    ARRAY_SIZE(benchmark_gpu_block_dim_variants)
#define BENCHMARK_ROW_COUNT_VARIANT_COUNT                                      \
    ARRAY_SIZE(benchmark_row_count_variants)
#define BENCHMARK_STREAM_COUNT_VARIANT_COUNT                                   \
    ARRAY_SIZE(benchmark_stream_count_variants)

int OMP_THREAD_COUNT = 0;

struct bench_data {
    union { // anonymous unions to disable RAII
        std::unordered_map<uint64_t, uint64_t>
            expected_output[BENCHMARK_ROW_COUNT_VARIANT_COUNT];
    };
    union {
        std::ofstream output_csv;
    };

    cudaDeviceProp device_properties;

    db_table input_cpu;
    db_table output_cpu;

    cudaStream_t streams[BENCHMARK_STREAMS_MAX];
    cudaEvent_t events[BENCHMARK_STREAMS_MAX];

    cudaEvent_t start_event;
    cudaEvent_t end_event;

    std::uniform_int_distribution<uint64_t> group_idx_dist;
#if !GROUP_COUNT_EQUALS_GROUP_MAX_VAL
    std::uniform_int_distribution<uint64_t> group_vals_dist;
#endif
    std::uniform_int_distribution<uint64_t> aggregates_dist;

    gpu_data data_gpu;

    bench_data()
    {
    }
    ~bench_data()
    {
    }
};

void alloc_pinned_db_table_cpu(db_table* t, uint64_t row_count)
{
    CUDA_TRY(cudaMallocHost(&t->group_col, row_count * sizeof(uint64_t)));
    CUDA_TRY(cudaMallocHost(&t->aggregate_col, row_count * sizeof(uint64_t)));
}

void free_pinned_db_table_cpu(db_table* t)
{
    CUDA_TRY(cudaFreeHost(t->aggregate_col));
    CUDA_TRY(cudaFreeHost(t->group_col));
}

void alloc_db_table_cpu(db_table* t, uint64_t row_count)
{
    t->group_col = (uint64_t*)malloc(row_count * sizeof(uint64_t));
    RELASE_ASSERT(t->group_col);
    t->aggregate_col = (uint64_t*)malloc(row_count * sizeof(uint64_t));
    RELASE_ASSERT(t->aggregate_col);
    t->row_count = row_count;
}

void free_db_table_cpu(db_table* t)
{
    free(t->group_col);
    free(t->aggregate_col);
}

void alloc_bench_data(bench_data* bd)
{
    int dc;
    cudaGetDeviceCount(&dc);
    // RELASE_ASSERT(dc == 1);
    cudaGetDeviceProperties(&bd->device_properties, 0);

    for (int rcv = 0; rcv < BENCHMARK_ROW_COUNT_VARIANT_COUNT; rcv++) {
        RELASE_ASSERT((new (&bd->expected_output[rcv])
                           std::unordered_map<uint64_t, uint64_t>()));
    }

#if DONT_WANT_PINNED_MEM
    alloc_db_table_cpu(&bd->input_cpu, BENCHMARK_ROWS_MAX);
    alloc_db_table_cpu(&bd->output_cpu, BENCHMARK_GROUPS_MAX);
#else
    alloc_pinned_db_table_cpu(&bd->input_cpu, BENCHMARK_ROWS_MAX);
    alloc_pinned_db_table_cpu(&bd->output_cpu, BENCHMARK_GROUPS_MAX);
#endif

    for (int i = 0; i < BENCHMARK_STREAMS_MAX; i++) {
        CUDA_TRY(cudaStreamCreate(&bd->streams[i]));
        CUDA_TRY(cudaEventCreate(&bd->events[i]));
    }

    CUDA_TRY(cudaEventCreate(&bd->start_event));
    CUDA_TRY(cudaEventCreate(&bd->end_event));

    gpu_data_alloc(&bd->data_gpu, BENCHMARK_ROWS_MAX, BENCHMARK_GROUPS_MAX);

#if ENABLE_APPROACH_HASHTABLE
    group_by_hashtable_init(BENCHMARK_GROUPS_MAX);
#endif
#if ENABLE_APPROACH_WARP_CMP
    group_by_warp_cmp_init(BENCHMARK_GROUPS_MAX);
#endif
#if ENABLE_APPROACH_BLOCK_CMP
    group_by_block_cmp_init(BENCHMARK_GROUPS_MAX);
#endif
#if ENABLE_APPROACH_SHARED_MEM_HASHTABLE
    group_by_shared_mem_hashtable_init(BENCHMARK_GROUPS_MAX);
#endif
#if ENABLE_APPROACH_SHARED_MEM_PERFECT_HASHTABLE
    group_by_shared_mem_perfect_hashtable_init(BENCHMARK_GROUPS_MAX);
#endif
#if ENABLE_APPROACH_PER_THREAD_HASHTABLE
    group_by_per_thread_hashtable_init(BENCHMARK_GROUPS_MAX);
#endif
#if ENABLE_APPROACH_CUB_RADIX_SORT
    group_by_cub_radix_sort_init(BENCHMARK_ROWS_MAX);
#endif
#if ENABLE_APPROACH_THROUGHPUT_TEST
    throughput_test_init();
#endif
#if ENABLE_APPROACH_GLOBAL_ARRAY
    group_by_global_array_init(BENCHMARK_GROUPS_MAX);
#endif
#if ENABLE_APPROACH_SHARED_MEM_ARRAY
    group_by_shared_mem_array_init(BENCHMARK_GROUPS_MAX);
#endif
#if ENABLE_APPROACH_PER_THREAD_ARRAY
    group_by_per_thread_array_init(BENCHMARK_GROUPS_MAX);
#endif
}

void free_bench_data(bench_data* bd)
{
#if ENABLE_APPROACH_PER_THREAD_ARRAY
    group_by_per_thread_array_fin();
#endif
#if ENABLE_APPROACH_SHARED_MEM_ARRAY
    group_by_shared_mem_array_fin();
#endif
#if ENABLE_APPROACH_GLOBAL_ARRAY
    group_by_global_array_fin();
#endif
#if ENABLE_APPROACH_THROUGHPUT_TEST
    throughput_test_fin();
#endif
#if ENABLE_APPROACH_CUB_RADIX_SORT
    group_by_cub_radix_sort_fin();
#endif
#if ENABLE_APPROACH_PER_THREAD_HASHTABLE
    group_by_per_thread_hashtable_fin();
#endif
#if ENABLE_APPROACH_SHARED_MEM_PERFECT_HASHTABLE
    group_by_shared_mem_perfect_hashtable_fin();
#endif
#if ENABLE_APPROACH_SHARED_MEM_HASHTABLE
    group_by_shared_mem_hashtable_fin();
#endif
#if ENABLE_APPROACH_BLOCK_CMP
    group_by_block_cmp_fin();
#endif
#if ENABLE_APPROACH_WARP_CMP
    group_by_warp_cmp_fin();
#endif
#if ENABLE_APPROACH_HASHTABLE
    group_by_hashtable_fin();
#endif

    gpu_data_free(&bd->data_gpu);

    CUDA_TRY(cudaEventDestroy(bd->end_event));
    CUDA_TRY(cudaEventDestroy(bd->start_event));

    for (int i = 0; i < BENCHMARK_STREAMS_MAX; i++) {
        CUDA_TRY(cudaEventDestroy(bd->events[i]));
        CUDA_TRY(cudaStreamDestroy(bd->streams[i]));
    }
#if DONT_WANT_PINNED_MEM
    free_db_table_cpu(&bd->output_cpu);
    free_db_table_cpu(&bd->input_cpu);
#else
    free_pinned_db_table_cpu(&bd->output_cpu);
    free_pinned_db_table_cpu(&bd->input_cpu);
#endif

    for (int rcv = 0; rcv < BENCHMARK_ROW_COUNT_VARIANT_COUNT; rcv++) {
        bd->expected_output[rcv].~unordered_map();
    }
}

template <size_t GENERATOR_STRIDE>
void write_bench_data(
    bench_data* bd, size_t group_count, size_t generator_base_seed)
{
    std::mt19937_64 generator{};

    // generate group_count different group values
    // (duplicates just mean less groups, no big deal)
#if !GROUP_COUNT_EQUALS_GROUP_MAX_VAL
    std::vector<uint64_t> groups{};
    groups.reserve(group_count);
    for (uint64_t i = 0; i < group_count; i++) {
        if (i % GENERATOR_STRIDE == 0) {
            generator = std::mt19937_64(generator_base_seed + i);
        }
        groups.push_back(bd->group_vals_dist(generator));
    }
#endif

    // initialize input table with random group and aggregate values
    // and increase the ag
    bd->expected_output[0].clear();
    size_t last_row_count = 0;
    for (int rcv = 0; rcv < BENCHMARK_ROW_COUNT_VARIANT_COUNT; rcv++) {
        size_t row_count = benchmark_row_count_variants[rcv];
        for (uint64_t i = last_row_count; i < row_count; i++) {
            if (i % GENERATOR_STRIDE == 0) {
                generator = std::mt19937_64(generator_base_seed + i);
            }
            uint64_t group;
#if GROUP_COUNT_EQUALS_GROUP_MAX_VAL
            group = bd->group_idx_dist(generator);
#else
            group = groups[bd->group_idx_dist(generator)];
#endif
            uint64_t val = bd->aggregates_dist(generator);
            bd->input_cpu.group_col[i] = group;
            bd->input_cpu.aggregate_col[i] = val;
            auto idx = bd->expected_output[rcv].find(group);
            if (idx != bd->expected_output[rcv].end()) {
                idx->second += val;
            }
            else {
                bd->expected_output[rcv][group] = val;
            }
        }
        if (rcv + 1 < BENCHMARK_ROW_COUNT_VARIANT_COUNT) {
            // for higher row count variants we can reuse the
            // expected_output accumulated so far
            bd->expected_output[rcv + 1] = bd->expected_output[rcv];
        }
        last_row_count = row_count;
    }
}

template <size_t GENERATOR_STRIDE>
void write_bench_data_omp(
    bench_data* bd, size_t group_count, size_t generator_base_seed)
{
    size_t max_row_count =
        benchmark_row_count_variants[BENCHMARK_ROW_COUNT_VARIANT_COUNT - 1];

    // since these generator types are huge (2504 bytes on my machine)
    // im not to worried about false sharing
    std::vector<std::mt19937_64> generators;
    generators.resize(OMP_THREAD_COUNT);

    // going back to the 90's to get a dynamic array that does't zero
    // initialize. *insert Thorvalds quote here*
    uint64_t* groups = (uint64_t*)malloc(group_count * sizeof(uint64_t));
    RELASE_ASSERT(groups);

    size_t stride = group_count / OMP_THREAD_COUNT;
    if (!stride) stride = 1;
    if (stride % GENERATOR_STRIDE != 0) {
        stride += GENERATOR_STRIDE - (stride % GENERATOR_STRIDE);
    }
#if !GROUP_COUNT_EQUALS_GROUP_MAX_VAL
#    pragma omp parallel for
    for (int t = 0; t < OMP_THREAD_COUNT; t++) {
        size_t start = t * stride;
        if (start < group_count) {
            size_t end = (t + 1) * stride;
            if (end > group_count || t + 1 == OMP_THREAD_COUNT) {
                end = group_count;
            }
            for (size_t i = start; i < end; i++) {
                if (i % GENERATOR_STRIDE == 0) {
                    generators[t] = std::mt19937_64(generator_base_seed + i);
                }
                groups[i] = bd->group_vals_dist(generators[t]);
            }
        }
    }
#endif
    typedef std::tuple<
        std::unordered_map<uint64_t, uint64_t>, size_t, size_t, int, int>
        section;
    constexpr int map_idx = 0;
    constexpr int start_idx = 1;
    constexpr int end_idx = 2;
    constexpr int rcv_idx = 3;
    constexpr int thrd_idx = 4;
    std::vector<section> sections;

    for (int rcv = 0; rcv < BENCHMARK_ROW_COUNT_VARIANT_COUNT; rcv++) {
        section sec = section{};
        std::get<start_idx>(sec) =
            rcv ? benchmark_row_count_variants[rcv - 1] : 0;
        std::get<end_idx>(sec) = benchmark_row_count_variants[rcv];
        std::get<rcv_idx>(sec) = rcv;
        sections.push_back(std::move(sec));
    }
    size_t thread_work = max_row_count / OMP_THREAD_COUNT;
    int i = 0;
    int t = 0;
    size_t t_work = 0;
    const float slack = 0.1;
    while (true) {
        if (t == OMP_THREAD_COUNT) break;
        size_t work =
            std::get<end_idx>(sections[i]) - std::get<start_idx>(sections[i]);
        if (t_work + work < thread_work * (1 + slack)) {
            std::get<thrd_idx>(sections[i]) = t;
            t_work += work;
            i++;
            if (t_work > (1 - slack) * thread_work) t++;
            continue;
        }

        if (thread_work > t_work) {
            section sec_split = sections[i];
            std::get<end_idx>(sec_split) = std::get<end_idx>(sections[i]);
            std::get<start_idx>(sec_split) =
                std::get<start_idx>(sections[i]) + (thread_work - t_work);
            std::get<end_idx>(sections[i]) = std::get<start_idx>(sec_split);
            std::get<thrd_idx>(sections[i]) = t;
            i++;
            sections.insert(sections.begin() + i, std::move(sec_split));
        }
        t++;
        t_work = 0;
    }
    while (i < sections.size()) {
        std::get<thrd_idx>(sections[i]) = OMP_THREAD_COUNT - 1;
        i++;
    }

#pragma omp parallel for
    for (int t = 0; t < OMP_THREAD_COUNT; t++) {
        int sid = -1;
        for (int i = 0; i < sections.size(); i++) {
            int tid = std::get<thrd_idx>(sections[i]);
            if (tid >= t) {
                if (tid == t) sid = i;
                break;
            }
        }
        if (sid != -1) {
            size_t start = std::get<start_idx>(sections[sid]);
            size_t gen_base = (start / GENERATOR_STRIDE) * GENERATOR_STRIDE;
            if (gen_base != start) {
                generators[t] = std::mt19937_64(generator_base_seed + gen_base);
                // discard twice since we use the generator for group and value
                generators[t].discard((start - gen_base) * 2);
            }
            while (sid < sections.size() &&
                   std::get<thrd_idx>(sections[sid]) == t) {
                size_t start = std::get<start_idx>(sections[sid]);
                size_t end = std::get<end_idx>(sections[sid]);
                for (uint64_t i = start; i < end; i++) {
                    if (i % GENERATOR_STRIDE == 0) {
                        generators[t] =
                            std::mt19937_64(generator_base_seed + i);
                    }
                    uint64_t group_val;
#if GROUP_COUNT_EQUALS_GROUP_MAX_VAL
                    group_val = bd->group_idx_dist(generators[t]);
#else
                    group_val = groups[bd->group_idx_dist(generators[t])];
#endif
                    bd->input_cpu.group_col[i] = group_val;

                    bd->input_cpu.aggregate_col[i] =
                        bd->aggregates_dist(generators[t]);
                }
                sid++;
            }
        }
    }

    if (group_count < 0.01 * max_row_count) {
        // number of groups is comparatively small, merging sectios is cheap
#pragma omp parallel for
        for (int t = 0; t < OMP_THREAD_COUNT; t++) {
            int sid = -1;
            for (int i = 0; i < sections.size(); i++) {
                int tid = std::get<thrd_idx>(sections[i]);
                if (tid >= t) {
                    if (tid == t) sid = i;
                    break;
                }
            }
            if (sid != -1) {
                while (sid < sections.size() &&
                       std::get<thrd_idx>(sections[sid]) == t) {
                    size_t start = std::get<start_idx>(sections[sid]);
                    size_t end = std::get<end_idx>(sections[sid]);
                    auto& map = std::get<map_idx>(sections[sid]);
                    for (uint64_t i = start; i < end; i++) {
                        uint64_t group = bd->input_cpu.group_col[i];
                        uint64_t val = bd->input_cpu.aggregate_col[i];
                        auto idx = map.find(group);
                        if (idx != map.end()) {
                            idx->second += val;
                        }
                        else {
                            map[group] = val;
                        }
                    }
                    sid++;
                }
            }
        }
#pragma omp parallel for
        for (int t = 0; t < OMP_THREAD_COUNT; t++) {
            for (int rcv_id = t; rcv_id < BENCHMARK_ROW_COUNT_VARIANT_COUNT;
                 rcv_id += OMP_THREAD_COUNT) {
                bd->expected_output[rcv_id].clear();
                int s_end = 0;
                bool found = false;
                while (s_end < sections.size()) {
                    int rcv = std::get<rcv_idx>(sections[s_end]);
                    if (found && rcv != rcv_id) break;
                    if (rcv == rcv_id) found = true;
                    s_end++;
                }
                auto& map = bd->expected_output[rcv_id];
                for (int i = 0; i != s_end; i++) {
                    auto& sec_map = std::get<map_idx>(sections[i]);
                    for (auto kv : sec_map) {
                        auto idx = map.find(kv.first);
                        if (idx != map.end()) {
                            idx->second += kv.second;
                        }
                        else {
                            map[kv.first] = kv.second;
                        }
                    }
                }
            }
        }
    }
    else {
#pragma omp parallel for
        for (int rcv = 0; rcv < BENCHMARK_ROW_COUNT_VARIANT_COUNT; rcv++) {
            size_t start = rcv ? benchmark_row_count_variants[rcv - 1] : 0;
            size_t end = benchmark_row_count_variants[rcv];
            auto& map = bd->expected_output[rcv];
            map.clear();
            for (uint64_t i = start; i < end; i++) {
                uint64_t group = bd->input_cpu.group_col[i];
                uint64_t val = bd->input_cpu.aggregate_col[i];
                auto idx = map.find(group);
                if (idx != map.end()) {
                    idx->second += val;
                }
                else {
                    map[group] = val;
                }
            }
        }
        for (int rcv = 1; rcv < BENCHMARK_ROW_COUNT_VARIANT_COUNT; rcv++) {
            auto& prev = bd->expected_output[rcv - 1];
            auto& map = bd->expected_output[rcv];
            for (auto kv : prev) {
                auto idx = map.find(kv.first);
                if (idx != map.end()) {
                    idx->second += kv.second;
                }
                else {
                    map[kv.first] = kv.second;
                }
            }
        }
    }
}

void setup_bench_data(bench_data* bd, size_t group_bits)
{
    // use static seeds for the generators to improve reproducability
    // special care was also taken to make sure that OMP_THREAD_COUNT
    // does not influence the results
    size_t group_count = (size_t)1 << group_bits;
    constexpr size_t generator_base_seed = 1337;
    constexpr size_t generator_stride = 1 << 15;

#if !GROUP_COUNT_EQUALS_GROUP_MAX_VAL
    bd->group_vals_dist = std::uniform_int_distribution<uint64_t>{
        BENCHMARK_GROUP_VALS_MIN, BENCHMARK_GROUP_VALS_MAX};
#endif

    bd->aggregates_dist = std::uniform_int_distribution<uint64_t>{
        BENCHMARK_AGGREGATE_VALS_MIN, BENCHMARK_AGGREGATE_VALS_MAX};
    bd->group_idx_dist =
        std::uniform_int_distribution<uint64_t>{0, group_count - 1};

    // completely separate the cases to make it more readable
#if !USE_OPENMP
    write_bench_data<generator_stride>(bd, group_count, generator_base_seed);
#else
    write_bench_data_omp<generator_stride>(
        bd, group_count, generator_base_seed);
#endif

    // store the final row count
    bd->input_cpu.row_count =
        benchmark_row_count_variants[BENCHMARK_ROW_COUNT_VARIANT_COUNT - 1];
    // copy the input to the gpu
    CUDA_TRY(cudaMemcpy(
        bd->data_gpu.input.group_col, bd->input_cpu.group_col,
        BENCHMARK_ROWS_MAX * sizeof(uint64_t), cudaMemcpyHostToDevice));
    CUDA_TRY(cudaMemcpy(
        bd->data_gpu.input.aggregate_col, bd->input_cpu.aggregate_col,
        BENCHMARK_ROWS_MAX * sizeof(uint64_t), cudaMemcpyHostToDevice));

#if ENABLE_APPROACH_SHARED_MEM_PERFECT_HASHTABLE
    build_perfect_hashtable(
        bd->expected_output, benchmark_row_count_variants,
        BENCHMARK_ROW_COUNT_VARIANT_COUNT, group_bits);
#endif
}
bool validate(bench_data* bd, int row_count_variant)
{
#if (!VALIDATION_OFF)
    bd->output_cpu.row_count = bd->data_gpu.output.row_count;
    std::vector<size_t> faults;
    faults.resize(OMP_THREAD_COUNT, 0);
    size_t row_count = bd->output_cpu.row_count;
    bool fault_occured = false;
    if (row_count > (1 << 13)) {
        size_t stride = row_count / OMP_THREAD_COUNT;
        if (!stride) stride = 1;
#    pragma omp parallel for
        for (int t = 0; t < OMP_THREAD_COUNT; t++) {
            size_t start = t * stride;
            if (start < row_count) {
                size_t end = (t + 1) * stride;
                if (end > row_count || t + 1 == OMP_THREAD_COUNT) {
                    end = row_count;
                }
                if (start < end) {
                    size_t byte_count = (end - start) * sizeof(uint64_t);
                    cudaMemcpy(
                        bd->output_cpu.group_col + start,
                        bd->data_gpu.output.group_col + start, byte_count,
                        cudaMemcpyDeviceToHost);
                    cudaMemcpy(
                        bd->output_cpu.aggregate_col + start,
                        bd->data_gpu.output.aggregate_col + start, byte_count,
                        cudaMemcpyDeviceToHost);

                    for (size_t i = start; i < end; i++) {
                        uint64_t group = bd->output_cpu.group_col[i];
                        auto expected =
                            bd->expected_output[row_count_variant].find(group);
                        uint64_t got = bd->output_cpu.aggregate_col[i];
                        if (expected ==
                                bd->expected_output[row_count_variant].end() ||
                            expected->second != got) {
                            faults[t] = i + 1;
                            i = end;
                            fault_occured = true;
                        }
                    }
                }
            }
        }
    }
    else {
        cudaMemcpy(
            bd->output_cpu.group_col, bd->data_gpu.output.group_col,
            row_count * sizeof(uint64_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(
            bd->output_cpu.aggregate_col, bd->data_gpu.output.aggregate_col,
            row_count * sizeof(uint64_t), cudaMemcpyDeviceToHost);
        for (size_t i = 0; i < row_count; i++) {
            uint64_t group = bd->output_cpu.group_col[i];
            auto expected = bd->expected_output[row_count_variant].find(group);
            uint64_t got = bd->output_cpu.aggregate_col[i];
            if (expected == bd->expected_output[row_count_variant].end() ||
                expected->second != got) {
                faults[0] = i + 1;
                fault_occured = true;
                break;
            }
        }
    }
    if (fault_occured) {
#    if ALLOW_FAILIURE
        return false;
#    else
        for (size_t i : faults) {
            if (i == 0) continue;

            i--;
            uint64_t group = bd->output_cpu.group_col[i];
            auto expected = bd->expected_output[row_count_variant].find(group);
            uint64_t got = bd->output_cpu.aggregate_col[i];
            if (expected == bd->expected_output[row_count_variant].end()) {
                fprintf(
                    stderr,
                    "validation failiure: found unexpected group %" PRIu64 "in "
                    "output index %" PRIu64 "\n",
                    group, i);
                __builtin_trap();
                return false;
            }
            else if (expected->second != got) {
                fprintf(
                    stderr,
                    "validation failiure for group %" PRIu64
                    ": expected %" PRIu64 ", got %" PRIu64 "\n",
                    group, expected->second, got);
                __builtin_trap();
                return false;
            }
        }
#    endif
    }
    const size_t expected_output_row_count =
        bd->expected_output[row_count_variant].size();
    if (bd->output_cpu.row_count != expected_output_row_count) {
#    if (!ALLOW_FAILIURE)
        fprintf(
            stderr,
            "validation failiure: expected %" PRIu64
            " different groups, got %" PRIu64 "\n",
            expected_output_row_count, bd->output_cpu.row_count);
        if (bd->output_cpu.row_count < expected_output_row_count) {
            std::unordered_set<uint64_t> occured_groups;
            for (size_t i = 0; i < bd->output_cpu.row_count; i++) {
                occured_groups.insert(bd->output_cpu.group_col[i]);
            }
            for (auto it = bd->expected_output[row_count_variant].begin();
                 it != bd->expected_output[row_count_variant].end(); ++it) {
                if (occured_groups.find(it->first) == occured_groups.end()) {
                    fprintf(
                        stderr, "missing expected group %" PRIu64 "\n",
                        it->first);
                }
            }
        }
        __builtin_trap();
#    endif
        return false;
    }
#endif
    return true;
}

void record_time_and_validate(
    bench_data* bd, int group_bit_count, int row_count_variant, int grid_dim,
    int block_dim, int stream_count, int iteration, const char* approach_name,
    bool no_validate = false)
{
    CUDA_TRY(cudaEventSynchronize(bd->end_event));
    CUDA_TRY(cudaDeviceSynchronize());
    float time;
    CUDA_TRY(cudaEventElapsedTime(&time, bd->start_event, bd->end_event));
    bool success = true;
    if (!no_validate) {
        success = validate(bd, row_count_variant);
    }

#if (!ALLOW_FAILIURE)
    RELASE_ASSERT(success);
#endif
    // the flush on std::endl is intentional to allow for tail -f style
    // status inspections
    bd->output_csv << approach_name << ";" << ((size_t)1 << group_bit_count)
                   << ";" << benchmark_row_count_variants[row_count_variant]
                   << ";" << grid_dim << ";" << block_dim << ";" << stream_count
                   << ";" << iteration << ";" << (success ? "PASS" : "FAIL")
                   << ";" << time << std::endl;
}

template <int GROUP_BIT_COUNT>
void run_approach_hashtable(
    bench_data* bd, bool eager, int row_count_variant, size_t row_count,
    int grid_dim, int block_dim, int stream_count, int iteration)
{
#if ENABLE_APPROACH_HASHTABLE

    if (!approach_hashtable_available(
            GROUP_BIT_COUNT, row_count, grid_dim, block_dim, stream_count)) {
        return;
    }

#    if ENABLE_HASHTABLE_EAGER_OUT_IDX
    if (eager) {
        group_by_hashtable<GROUP_BIT_COUNT, true>(
            &bd->data_gpu, grid_dim, block_dim, stream_count, bd->streams,
            bd->events, bd->start_event, bd->end_event);
        record_time_and_validate(
            bd, GROUP_BIT_COUNT, row_count_variant, grid_dim, block_dim,
            stream_count, iteration, "hashtable_eager_out_idx");
    }
#    endif
    if (!eager) {
        group_by_hashtable<GROUP_BIT_COUNT, false>(
            &bd->data_gpu, grid_dim, block_dim, stream_count, bd->streams,
            bd->events, bd->start_event, bd->end_event);
        record_time_and_validate(
            bd, GROUP_BIT_COUNT, row_count_variant, grid_dim, block_dim,
            stream_count, iteration, "hashtable");
    }
#endif
}

template <int GROUP_BIT_COUNT>
void run_approach_warp_cmp(
    bench_data* bd, int row_count_variant, size_t row_count, int grid_dim,
    int block_dim, int stream_count, int iteration)
{
#if ENABLE_APPROACH_WARP_CMP
    if (!approach_warp_cmp_available(
            GROUP_BIT_COUNT, row_count, grid_dim, block_dim, stream_count)) {
        return;
    }
    group_by_warp_cmp<GROUP_BIT_COUNT>(
        &bd->data_gpu, grid_dim, block_dim, stream_count, bd->streams,
        bd->events, bd->start_event, bd->end_event);
    record_time_and_validate(
        bd, GROUP_BIT_COUNT, row_count_variant, grid_dim, block_dim,
        stream_count, iteration, "warp_cmp");
#endif
}

template <int GROUP_BIT_COUNT>
void run_approach_block_cmp(
    bench_data* bd, bool naive, bool old, int row_count_variant,
    size_t row_count, int grid_dim, int block_dim, int stream_count,
    int iteration)
{
#if ENABLE_APPROACH_BLOCK_CMP
    if (!approach_block_cmp_available(
            GROUP_BIT_COUNT, row_count, grid_dim, block_dim, stream_count)) {
        return;
    }
    if (!naive && !old) {
        group_by_block_cmp<GROUP_BIT_COUNT, false, false>(
            &bd->data_gpu, grid_dim, block_dim, stream_count, bd->streams,
            bd->events, bd->start_event, bd->end_event);
        record_time_and_validate(
            bd, GROUP_BIT_COUNT, row_count_variant, grid_dim, block_dim,
            stream_count, iteration, "block_cmp");
    }

#    if ENABLE_BLOCK_CMP_NAIVE_WRITEOUT
    if (naive && !old) {
        group_by_block_cmp<GROUP_BIT_COUNT, true, false>(
            &bd->data_gpu, grid_dim, block_dim, stream_count, bd->streams,
            bd->events, bd->start_event, bd->end_event);
        record_time_and_validate(
            bd, GROUP_BIT_COUNT, row_count_variant, grid_dim, block_dim,
            stream_count, iteration, "block_cmp_naive_writeout");
    }
#    endif

#    if ENABLE_BLOCK_CMP_OLD
    if (!naive && old) {
        group_by_block_cmp<GROUP_BIT_COUNT, false, true>(
            &bd->data_gpu, grid_dim, block_dim, stream_count, bd->streams,
            bd->events, bd->start_event, bd->end_event);
        record_time_and_validate(
            bd, GROUP_BIT_COUNT, row_count_variant, grid_dim, block_dim,
            stream_count, iteration, "block_cmp_old");
    }
#        if ENABLE_BLOCK_CMP_NAIVE_WRITEOUT
    if (naive && old) {
        group_by_block_cmp<GROUP_BIT_COUNT, true, true>(
            &bd->data_gpu, grid_dim, block_dim, stream_count, bd->streams,
            bd->events, bd->start_event, bd->end_event);
        record_time_and_validate(
            bd, GROUP_BIT_COUNT, row_count_variant, grid_dim, block_dim,
            stream_count, iteration, "block_cmp_old_naive_writeout");
    }
#        endif
#    endif
#endif
}

template <int GROUP_BIT_COUNT>
void run_approach_shared_mem_hashtable(
    bench_data* bd, bool optimistic, int row_count_variant, size_t row_count,
    int grid_dim, int block_dim, int stream_count, int iteration)
{
#if ENABLE_APPROACH_SHARED_MEM_HASHTABLE
    if (!approach_shared_mem_hashtable_available(
            GROUP_BIT_COUNT, row_count, grid_dim, block_dim, stream_count)) {
        return;
    }
    if (optimistic) {
        group_by_shared_mem_hashtable<GROUP_BIT_COUNT, true>(
            &bd->data_gpu, grid_dim, block_dim, stream_count, bd->streams,
            bd->events, bd->start_event, bd->end_event);
        record_time_and_validate(
            bd, GROUP_BIT_COUNT, row_count_variant, grid_dim, block_dim,
            stream_count, iteration, "shared_mem_hashtable_optimistic");
    }
    else {
        group_by_shared_mem_hashtable<GROUP_BIT_COUNT, false>(
            &bd->data_gpu, grid_dim, block_dim, stream_count, bd->streams,
            bd->events, bd->start_event, bd->end_event);
        record_time_and_validate(
            bd, GROUP_BIT_COUNT, row_count_variant, grid_dim, block_dim,
            stream_count, iteration, "shared_mem_hashtable");
    }

#endif
}

template <int GROUP_BIT_COUNT>
void run_approach_shared_mem_perfect_hashtable(
    bench_data* bd, int row_count_variant, size_t row_count, int grid_dim,
    int block_dim, int stream_count, int iteration)
{
#if ENABLE_APPROACH_SHARED_MEM_PERFECT_HASHTABLE
    if (!approach_shared_mem_perfect_hashtable_available(
            GROUP_BIT_COUNT, row_count, grid_dim, block_dim, stream_count)) {
        return;
    }

    group_by_shared_mem_perfect_hashtable<GROUP_BIT_COUNT>(
        &bd->data_gpu, grid_dim, block_dim, stream_count, bd->streams,
        bd->events, bd->start_event, bd->end_event);
    record_time_and_validate(
        bd, GROUP_BIT_COUNT, row_count_variant, grid_dim, block_dim,
        stream_count, iteration, "shared_mem_perfect_hashtable");
#endif
}
template <int GROUP_BIT_COUNT>
void run_approach_per_thread_hashtable(
    bench_data* bd, bool bank_optimized, int row_count_variant,
    size_t row_count, int grid_dim, int block_dim, int stream_count,
    int iteration)
{
#if ENABLE_APPROACH_PER_THREAD_HASHTABLE
    if (!approach_per_thread_hashtable_available(
            GROUP_BIT_COUNT, row_count, grid_dim, block_dim, stream_count)) {
        return;
    }
    if (!bank_optimized) {
        group_by_per_thread_hashtable<GROUP_BIT_COUNT>(
            &bd->data_gpu, grid_dim, block_dim, stream_count, bd->streams,
            bd->events, bd->start_event, bd->end_event);
        record_time_and_validate(
            bd, GROUP_BIT_COUNT, row_count_variant, grid_dim, block_dim,
            stream_count, iteration, "per_thread_hashtable");
    }
    else {
        group_by_per_thread_hashtable<GROUP_BIT_COUNT, true>(
            &bd->data_gpu, grid_dim, block_dim, stream_count, bd->streams,
            bd->events, bd->start_event, bd->end_event);
        record_time_and_validate(
            bd, GROUP_BIT_COUNT, row_count_variant, grid_dim, block_dim,
            stream_count, iteration, "per_thread_hashtable_bank_optimized");
    }
#endif
}

template <int GROUP_BIT_COUNT>
void run_approach_cub_radix_sort(
    bench_data* bd, int row_count_variant, size_t row_count, int grid_dim,
    int block_dim, int stream_count, int iteration)
{
#if ENABLE_APPROACH_CUB_RADIX_SORT
    if (!approach_cub_radix_sort_available(
            GROUP_BIT_COUNT, row_count, grid_dim, block_dim, stream_count)) {
        return;
    }
    group_by_cub_radix_sort<GROUP_BIT_COUNT>(
        &bd->data_gpu, grid_dim, block_dim, stream_count, bd->streams,
        bd->events, bd->start_event, bd->end_event);
    record_time_and_validate(
        bd, GROUP_BIT_COUNT, row_count_variant, grid_dim, block_dim,
        stream_count, iteration, "cub_radix_sort");
#endif
}

template <int GROUP_BIT_COUNT>
void run_approach_throughput_test(
    bench_data* bd, int row_count_variant, size_t row_count, int grid_dim,
    int block_dim, int stream_count, int iteration)
{
#if ENABLE_APPROACH_THROUGHPUT_TEST
    if (!approach_throughput_test_available(
            GROUP_BIT_COUNT, row_count, grid_dim, block_dim, stream_count)) {
        return;
    }
    throughput_test<GROUP_BIT_COUNT>(
        &bd->data_gpu, grid_dim, block_dim, stream_count, bd->streams,
        bd->events, bd->start_event, bd->end_event);
    record_time_and_validate(
        bd, GROUP_BIT_COUNT, row_count_variant, grid_dim, block_dim,
        stream_count, iteration, "throughput_test", true);
#endif
}

template <int GROUP_BIT_COUNT, bool OPTIMISTIC, bool COMPRESSTORE>
void run_approach_global_array(
    bench_data* bd, int row_count_variant, size_t row_count, int grid_dim,
    int block_dim, int stream_count, int iteration)
{
#if ENABLE_APPROACH_GLOBAL_ARRAY
    if (!approach_global_array_available(
            GROUP_BIT_COUNT, row_count, grid_dim, block_dim, stream_count)) {
        return;
    }

    group_by_global_array<GROUP_BIT_COUNT, OPTIMISTIC, COMPRESSTORE>(
        &bd->data_gpu, grid_dim, block_dim, stream_count, bd->streams,
        bd->events, bd->start_event, bd->end_event);
    const char* ap_name;
    if (OPTIMISTIC && COMPRESSTORE) {
        ap_name = "global_array_optimistic_compresstore";
    }
    if (OPTIMISTIC && !COMPRESSTORE) ap_name = "global_array_optimistic";
    if (!OPTIMISTIC && COMPRESSTORE) ap_name = "global_array_compresstore";
    if (!OPTIMISTIC && !COMPRESSTORE) ap_name = "global_array";
    record_time_and_validate(
        bd, GROUP_BIT_COUNT, row_count_variant, grid_dim, block_dim,
        stream_count, iteration, ap_name);
#endif
}

template <int GROUP_BIT_COUNT>
void run_approach_shared_mem_array(
    bench_data* bd, bool optimistic, int row_count_variant, size_t row_count,
    int grid_dim, int block_dim, int stream_count, int iteration)
{
#if ENABLE_APPROACH_SHARED_MEM_ARRAY
    if (!approach_shared_mem_array_available(
            GROUP_BIT_COUNT, row_count, grid_dim, block_dim, stream_count)) {
        return;
    }
    if (optimistic) {
        group_by_shared_mem_array<GROUP_BIT_COUNT, true>(
            &bd->data_gpu, grid_dim, block_dim, stream_count, bd->streams,
            bd->events, bd->start_event, bd->end_event);
        record_time_and_validate(
            bd, GROUP_BIT_COUNT, row_count_variant, grid_dim, block_dim,
            stream_count, iteration, "shared_mem_array_optimistic");
    }
    else {
        group_by_shared_mem_array<GROUP_BIT_COUNT, false>(
            &bd->data_gpu, grid_dim, block_dim, stream_count, bd->streams,
            bd->events, bd->start_event, bd->end_event);
        record_time_and_validate(
            bd, GROUP_BIT_COUNT, row_count_variant, grid_dim, block_dim,
            stream_count, iteration, "shared_mem_array");
    }

#endif
}

template <int GROUP_BIT_COUNT>
void run_approach_per_thread_array(
    bench_data* bd, bool bank_optimized, int row_count_variant,
    size_t row_count, int grid_dim, int block_dim, int stream_count,
    int iteration)
{
#if ENABLE_APPROACH_PER_THREAD_ARRAY
    if (!approach_per_thread_array_available(
            GROUP_BIT_COUNT, row_count, grid_dim, block_dim, stream_count)) {
        return;
    }
    if (bank_optimized) {
        group_by_per_thread_array<GROUP_BIT_COUNT, true>(
            &bd->data_gpu, grid_dim, block_dim, stream_count, bd->streams,
            bd->events, bd->start_event, bd->end_event);
        record_time_and_validate(
            bd, GROUP_BIT_COUNT, row_count_variant, grid_dim, block_dim,
            stream_count, iteration, "per_thread_array_bank_optimized");
    }
    else {
        group_by_per_thread_array<GROUP_BIT_COUNT, false>(
            &bd->data_gpu, grid_dim, block_dim, stream_count, bd->streams,
            bd->events, bd->start_event, bd->end_event);
        record_time_and_validate(
            bd, GROUP_BIT_COUNT, row_count_variant, grid_dim, block_dim,
            stream_count, iteration, "per_thread_array");
    }

#endif
}

template <int GROUP_BIT_COUNT>
int run_approach(
    bench_data* bd, int approach_id, int row_count_variant, size_t row_count,
    int grid_dim, int block_dim, int stream_count, int iteration)
{
    switch (approach_id) {
        case 0: {
            run_approach_hashtable<GROUP_BIT_COUNT>(
                bd, false, row_count_variant, row_count, grid_dim, block_dim,
                stream_count, iteration);
        } break;
        case 1: {
            run_approach_hashtable<GROUP_BIT_COUNT>(
                bd, true, row_count_variant, row_count, grid_dim, block_dim,
                stream_count, iteration);
        } break;
        case 2: {
            run_approach_shared_mem_hashtable<GROUP_BIT_COUNT>(
                bd, false, row_count_variant, row_count, grid_dim, block_dim,
                stream_count, iteration);
        } break;
        case 3: {
            run_approach_shared_mem_hashtable<GROUP_BIT_COUNT>(
                bd, true, row_count_variant, row_count, grid_dim, block_dim,
                stream_count, iteration);
        } break;
        case 4: {
            run_approach_per_thread_hashtable<GROUP_BIT_COUNT>(
                bd, false, row_count_variant, row_count, grid_dim, block_dim,
                stream_count, iteration);
        } break;
        case 5: {
            run_approach_per_thread_hashtable<GROUP_BIT_COUNT>(
                bd, true, row_count_variant, row_count, grid_dim, block_dim,
                stream_count, iteration);
        } break;
        case 6: {
            run_approach_warp_cmp<GROUP_BIT_COUNT>(
                bd, row_count_variant, row_count, grid_dim, block_dim,
                stream_count, iteration);
        } break;
        case 7: {
            run_approach_block_cmp<GROUP_BIT_COUNT>(
                bd, false, false, row_count_variant, row_count, grid_dim,
                block_dim, stream_count, iteration);
        } break;
        case 8: {
            run_approach_block_cmp<GROUP_BIT_COUNT>(
                bd, false, true, row_count_variant, row_count, grid_dim,
                block_dim, stream_count, iteration);
        } break;
        case 9: {
            run_approach_block_cmp<GROUP_BIT_COUNT>(
                bd, true, false, row_count_variant, row_count, grid_dim,
                block_dim, stream_count, iteration);
        } break;
        case 10: {
            run_approach_block_cmp<GROUP_BIT_COUNT>(
                bd, true, true, row_count_variant, row_count, grid_dim,
                block_dim, stream_count, iteration);
        } break;
        case 11: {
            run_approach_cub_radix_sort<GROUP_BIT_COUNT>(
                bd, row_count_variant, row_count, grid_dim, block_dim,
                stream_count, iteration);
        } break;
        case 12: {
            run_approach_throughput_test<GROUP_BIT_COUNT>(
                bd, row_count_variant, row_count, grid_dim, block_dim,
                stream_count, iteration);
        } break;
        case 13: {
            run_approach_shared_mem_perfect_hashtable<GROUP_BIT_COUNT>(
                bd, row_count_variant, row_count, grid_dim, block_dim,
                stream_count, iteration);
        } break;
        case 14: {
            run_approach_global_array<GROUP_BIT_COUNT, false, false>(
                bd, row_count_variant, row_count, grid_dim, block_dim,
                stream_count, iteration);
        } break;
        case 15: {
            run_approach_global_array<GROUP_BIT_COUNT, false, true>(
                bd, row_count_variant, row_count, grid_dim, block_dim,
                stream_count, iteration);
        } break;
        case 16: {
            run_approach_global_array<GROUP_BIT_COUNT, true, false>(
                bd, row_count_variant, row_count, grid_dim, block_dim,
                stream_count, iteration);
        } break;
        case 17: {
            run_approach_global_array<GROUP_BIT_COUNT, true, true>(
                bd, row_count_variant, row_count, grid_dim, block_dim,
                stream_count, iteration);
        } break;
        case 18: {
            run_approach_shared_mem_array<GROUP_BIT_COUNT>(
                bd, true, row_count_variant, row_count, grid_dim, block_dim,
                stream_count, iteration);
        } break;
        case 19: {
            run_approach_per_thread_array<GROUP_BIT_COUNT>(
                bd, false, row_count_variant, row_count, grid_dim, block_dim,
                stream_count, iteration);
        } break;
        case 20: {
            run_approach_per_thread_array<GROUP_BIT_COUNT>(
                bd, true, row_count_variant, row_count, grid_dim, block_dim,
                stream_count, iteration);
        } break;
        case 21: return -1;
        default: assert(false);
    }
    return approach_id + 1;
}
template <int GROUP_BIT_COUNT>
void run_benchmarks_for_group_bit_count(bench_data* bd)
{
    static_assert(GROUP_BIT_COUNT <= BENCHMARK_GROUP_BITS_MAX);
    setup_bench_data(bd, GROUP_BIT_COUNT);
    for (int rcv = 0; rcv < BENCHMARK_ROW_COUNT_VARIANT_COUNT; rcv++) {
        size_t row_count = benchmark_row_count_variants[rcv];
        // artificially reduce the row counts
        bd->data_gpu.input.row_count = benchmark_row_count_variants[rcv];
        for (int gdv = 0; gdv < BENCHMARK_GPU_GRID_DIM_VARIANT_COUNT; gdv++) {
            int grid_dim = benchmark_gpu_grid_dim_variants[gdv];
            for (int bdv = 0; bdv < BENCHMARK_GPU_BLOCK_DIM_VARIANT_COUNT;
                 bdv++) {
                int block_dim = benchmark_gpu_block_dim_variants[bdv];
                for (int scv = 0; scv < BENCHMARK_STREAM_COUNT_VARIANT_COUNT;
                     scv++) {
                    int stream_count = benchmark_stream_count_variants[scv];
                    int approach_id = 0;
                    while (approach_id != -1) {
                        int approach_id_next;
                        for (int it = 0; it < ITERATION_COUNT; it++) {
                            approach_id_next = run_approach<GROUP_BIT_COUNT>(
                                bd, approach_id, rcv, row_count, grid_dim,
                                block_dim, stream_count, it);
                        }
                        approach_id = approach_id_next;
                    }
                }
            }
        }
    }
}

int main()
{
#if USE_OPENMP
    OMP_THREAD_COUNT = omp_get_max_threads();
#else
    OMP_THREAD_COUNT = 1;
#endif
    bench_data bench_data;
    alloc_bench_data(&bench_data);
    new (&bench_data.output_csv) std::ofstream{"bench.csv"};
    bench_data.output_csv << "approach;groups;rows;grid dim;block dim;stream "
                             "count; run index; validation; time in ms"
                          << std::endl;
    bench_data.output_csv << std::fixed << std::setprecision(20);

    run_benchmarks_for_group_bit_count<1>(&bench_data);
    run_benchmarks_for_group_bit_count<3>(&bench_data);
    run_benchmarks_for_group_bit_count<5>(&bench_data);
    run_benchmarks_for_group_bit_count<7>(&bench_data);
    run_benchmarks_for_group_bit_count<9>(&bench_data);
    run_benchmarks_for_group_bit_count<10>(&bench_data);
    run_benchmarks_for_group_bit_count<11>(&bench_data);

#if BIG_DATA
    run_benchmarks_for_group_bit_count<2>(&bench_data);
    run_benchmarks_for_group_bit_count<4>(&bench_data);
    run_benchmarks_for_group_bit_count<11>(&bench_data);
    run_benchmarks_for_group_bit_count<15>(&bench_data);
    run_benchmarks_for_group_bit_count<20>(&bench_data);
#endif

    run_benchmarks_for_group_bit_count<BENCHMARK_GROUP_BITS_MAX>(&bench_data);

    bench_data.output_csv.flush();
    bench_data.output_csv.~basic_ofstream();
    free_bench_data(&bench_data);
    return 0;
}
