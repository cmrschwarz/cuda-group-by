#include <unordered_map>
#include <vector>
#include <random>
#include <limits>
#include <iostream>
#include <fstream>
#include <iomanip>

#include "group_by_hashtable.cuh"
#include "group_by_thread_per_group.cuh"

// set to false to reduce data size for debugging
#define BIG_DATA false

#if BIG_DATA
#define BENCHMARK_STREAMS_MAX 16
const size_t benchmark_stream_count_variants[] = {0, 1, 2,
                                                  4, 8, BENCHMARK_STREAMS_MAX};
#else
#define BENCHMARK_STREAMS_MAX 4
const size_t benchmark_stream_count_variants[] = {0, 1, BENCHMARK_STREAMS_MAX};
#endif

#if BIG_DATA
// 2^27, 8 Byte per entry -> 1 GiB per stored column
#define BENCHMARK_ROWS_MAX ((size_t)1 << 27)
const size_t benchmark_row_count_variants[] = {
    2, 4, 16, 32, 128, 1024, 16384, BENCHMARK_ROWS_MAX};
#else
#define BENCHMARK_ROWS_MAX ((size_t)1 << 16)
const size_t benchmark_row_count_variants[] = {2, 32, 128, BENCHMARK_ROWS_MAX};
#endif

#if BIG_DATA
const int benchmark_gpu_block_dim_variants[] = {32, 64, 128, 256, 512, 1024};
#else
const int benchmark_gpu_block_dim_variants[] = {32, 128, 1024};
#endif

#if BIG_DATA
const int benchmark_gpu_grid_dim_variants[] = {1, 8, 64, 128, 256, 512, 1024};
#else
const int benchmark_gpu_grid_dim_variants[] = {1, 64, 1024};
#endif

#if BIG_DATA
#define BENCHMARK_GROUP_BITS_MAX 20
#else
#define BENCHMARK_GROUP_BITS_MAX 10
#endif

#if BIG_DATA
#define BENCHMARK_GROUP_VALS_MIN std::numeric_limits<uint64_t>::min()
#define BENCHMARK_GROUP_VALS_MAX std::numeric_limits<uint64_t>::max()
#else
#define BENCHMARK_GROUP_VALS_MIN 0
#define BENCHMARK_GROUP_VALS_MAX (((size_t)1 << BENCHMARK_GROUP_BITS_MAX) - 1)
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

struct bench_data {
    union { // anonymous unions to disable RAII
        std::unordered_map<uint64_t, uint64_t>
            expected_output[BENCHMARK_ROW_COUNT_VARIANT_COUNT];
    };
    union {
        std::ofstream output_csv;
    };

    db_table input_cpu;
    db_table output_cpu;

    cudaStream_t streams[BENCHMARK_STREAMS_MAX];
    cudaEvent_t events[BENCHMARK_STREAMS_MAX];

    cudaEvent_t start_event;
    cudaEvent_t end_event;

    gpu_data data_gpu;

    bench_data()
    {
    }
    ~bench_data()
    {
    }
};

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
    for (int rcv = 0; rcv < BENCHMARK_ROW_COUNT_VARIANT_COUNT; rcv++) {
        RELASE_ASSERT((new (&bd->expected_output[rcv])
                           std::unordered_map<uint64_t, uint64_t>()));
    }

    alloc_db_table_cpu(&bd->input_cpu, BENCHMARK_ROWS_MAX);
    alloc_db_table_cpu(&bd->output_cpu, BENCHMARK_GROUPS_MAX);

    for (int i = 0; i < BENCHMARK_STREAMS_MAX; i++) {
        CUDA_TRY(cudaStreamCreate(&bd->streams[i]));
        CUDA_TRY(cudaEventCreate(&bd->events[i]));
    }

    CUDA_TRY(cudaEventCreate(&bd->start_event));
    CUDA_TRY(cudaEventCreate(&bd->end_event));

    gpu_data_alloc(&bd->data_gpu, BENCHMARK_ROWS_MAX, BENCHMARK_GROUPS_MAX);
    group_by_hashtable_init(BENCHMARK_ROWS_MAX);
    group_by_thread_per_group_init();
}

void free_bench_data(bench_data* bd)
{
    group_by_thread_per_group_fin();
    group_by_hashtable_fin();

    gpu_data_free(&bd->data_gpu);

    CUDA_TRY(cudaEventDestroy(bd->end_event));
    CUDA_TRY(cudaEventDestroy(bd->start_event));

    for (int i = 0; i < BENCHMARK_STREAMS_MAX; i++) {
        CUDA_TRY(cudaEventDestroy(bd->events[i]));
        CUDA_TRY(cudaStreamDestroy(bd->streams[i]));
    }

    free_db_table_cpu(&bd->output_cpu);
    free_db_table_cpu(&bd->input_cpu);

    for (int rcv = 0; rcv < BENCHMARK_ROW_COUNT_VARIANT_COUNT; rcv++) {
        bd->expected_output[rcv].~unordered_map();
    }
}

void setup_bench_data(bench_data* bd, size_t group_count)
{
    std::mt19937_64 generator{1337};

    std::uniform_int_distribution<uint64_t> uint_rng{BENCHMARK_GROUP_VALS_MIN,
                                                     BENCHMARK_GROUP_VALS_MAX};

    std::uniform_int_distribution<uint64_t> group_rng{0, group_count - 1};

    std::vector<uint64_t> groups{};
    groups.reserve(group_count);

    // generate group_count different group values
    // (duplicates just mean less groups, no big deal)
    for (uint64_t i = 0; i < group_count; i++) {
        groups.push_back(uint_rng(generator));
    }

    // initialize input table with random group and aggregate values
    // and increase the aggregate values in expected_output accordingly
    bd->expected_output[0].clear();
    size_t last_row_count = 0;
    for (int rcv = 0; rcv < BENCHMARK_ROW_COUNT_VARIANT_COUNT; rcv++) {
        size_t row_count = benchmark_row_count_variants[rcv];
        for (uint64_t i = last_row_count; i < row_count; i++) {
            uint64_t group = groups[group_rng(generator)];
            uint64_t val = uint_rng(generator);
            bd->input_cpu.group_col[i] = group;
            bd->input_cpu.aggregate_col[i] = val;
            if (bd->expected_output[rcv].find(group) !=
                bd->expected_output[rcv].end()) {
                bd->expected_output[rcv][group] += val;
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
    // store the final row count
    bd->input_cpu.row_count = last_row_count;

    // copy the input to the gpu
    CUDA_TRY(cudaMemcpy(
        bd->data_gpu.input.group_col, bd->input_cpu.group_col,
        BENCHMARK_ROWS_MAX * sizeof(uint64_t), cudaMemcpyHostToDevice));
    CUDA_TRY(cudaMemcpy(
        bd->data_gpu.input.aggregate_col, bd->input_cpu.aggregate_col,
        BENCHMARK_ROWS_MAX * sizeof(uint64_t), cudaMemcpyHostToDevice));
}
bool validate(bench_data* bd, int row_count_variant)
{
    bd->output_cpu.row_count = bd->data_gpu.output.row_count;
    cudaMemcpy(
        bd->output_cpu.group_col, bd->data_gpu.output.group_col,
        bd->output_cpu.row_count * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(
        bd->output_cpu.aggregate_col, bd->data_gpu.output.aggregate_col,
        bd->output_cpu.row_count * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < bd->output_cpu.row_count; i++) {
        uint64_t group = bd->output_cpu.group_col[i];
        auto expected = bd->expected_output[row_count_variant].find(group);
        uint64_t got = bd->output_cpu.aggregate_col[i];
        if (expected == bd->expected_output[row_count_variant].end()) {
            fprintf(
                stderr,
                "validation failiure: found unexpected group %llu in output "
                "index %llu\n",
                group, i);
            __builtin_trap();
            return false;
        }
        else if (expected->second != got) {
            fprintf(
                stderr,
                "validation failiure for group %llu: expected %llu, got "
                "%llu\n",
                group, expected->second, got);
            __builtin_trap();
            return false;
        }
    }
    const size_t expected_output_row_count =
        bd->expected_output[row_count_variant].size();
    if (bd->output_cpu.row_count != expected_output_row_count) {
        fprintf(
            stderr,
            "validation failiure: expected %llu different groups, got %llu\n",
            expected_output_row_count, bd->output_cpu.row_count);
        __builtin_trap();
        return false;
    }
    return true;
}

void record_time_and_validate(
    bench_data* bd, int group_count, int row_count_variant, int grid_dim,
    int block_dim, int stream_count, const char* approach_name)
{
    float time;
    CUDA_TRY(cudaEventElapsedTime(&time, bd->start_event, bd->end_event));
    RELASE_ASSERT(validate(bd, row_count_variant));
    // the flush on std::endl is intentional to allow for tail -f style
    // status inspections
    bd->output_csv << approach_name << ";" << group_count << ";"
                   << benchmark_row_count_variants[row_count_variant] << ";"
                   << grid_dim << ";" << block_dim << ";" << stream_count << ";"
                   << time << std::endl;
}

template <int GROUP_BIT_COUNT>
void run_benchmarks_for_group_bit_count(bench_data* bd)
{
    static_assert(GROUP_BIT_COUNT <= BENCHMARK_GROUP_BITS_MAX);
    setup_bench_data(bd, 1 << GROUP_BIT_COUNT);
    for (int rcv = 0; rcv < BENCHMARK_ROW_COUNT_VARIANT_COUNT; rcv++) {
        size_t row_count = benchmark_row_count_variants[rcv];
        // artificially reduce the row counts
        bd->data_gpu.input.row_count = row_count;
        for (int gdv = 0; gdv < BENCHMARK_GPU_GRID_DIM_VARIANT_COUNT; gdv++) {
            int grid_dim = benchmark_gpu_grid_dim_variants[gdv];
            for (int bdv = 0; bdv < BENCHMARK_GPU_BLOCK_DIM_VARIANT_COUNT;
                 bdv++) {
                int block_dim = benchmark_gpu_block_dim_variants[gdv];
                for (int scv = 0; scv < BENCHMARK_STREAM_COUNT_VARIANT_COUNT;
                     scv++) {
                    int stream_count = benchmark_stream_count_variants[scv];
                    if (approach_hashtable_available(
                            GROUP_BIT_COUNT, row_count, grid_dim, block_dim,
                            stream_count)) {
                        group_by_hashtable<GROUP_BIT_COUNT>(
                            &bd->data_gpu, grid_dim, block_dim, stream_count,
                            bd->streams, bd->events, bd->start_event,
                            bd->end_event);
                        record_time_and_validate(
                            bd, 1 << GROUP_BIT_COUNT, rcv, grid_dim, block_dim,
                            stream_count, "hashtable");
                    }
                    if (approach_thread_per_group_available(
                            GROUP_BIT_COUNT, row_count, grid_dim, block_dim,
                            stream_count)) {

                        group_by_thread_per_group<GROUP_BIT_COUNT>(
                            &bd->data_gpu, grid_dim, block_dim, stream_count,
                            bd->streams, bd->events, bd->start_event,
                            bd->end_event);
                        record_time_and_validate(
                            bd, 1 << GROUP_BIT_COUNT, rcv, grid_dim, block_dim,
                            stream_count, "thread_per_group");
                    }
                }
            }
        }
    }
}

int main()
{
    bench_data bench_data;
    alloc_bench_data(&bench_data);
    new (&bench_data.output_csv) std::ofstream{"bench.csv"};
    bench_data.output_csv
        << "approach;groups;rows;grid dim;block dim;stream count;time in ms\n";
    bench_data.output_csv << std::fixed << std::setprecision(20);
    run_benchmarks_for_group_bit_count<1>(&bench_data);
    run_benchmarks_for_group_bit_count<5>(&bench_data);
    run_benchmarks_for_group_bit_count<BENCHMARK_GROUP_BITS_MAX>(&bench_data);
#if BIG_DATA
    run_benchmarks_for_group_bit_count<2>(&bench_data);
    run_benchmarks_for_group_bit_count<8>(&bench_data);
    run_benchmarks_for_group_bit_count<16>(&bench_data);
#endif
    bench_data.output_csv.flush();
    bench_data.output_csv.~basic_ofstream();
    free_bench_data(&bench_data);
    return 0;
}
