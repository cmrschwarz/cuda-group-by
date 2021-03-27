#include <unordered_map>
#include <vector>
#include <random>
#include <limits>

#include "utils.h"
#include "cuda_group_by.cuh"

#define MAX_GROUPS 128

#define BENCHMARK_GROUP_VAL_MIN std::numeric_limits<uint64_t>::min()
#define BENCHMARK_GROUP_VAL_MAX std::numeric_limits<uint64_t>::max()

#define BENCHMARK_MIN_STREAMS 1
#define BENCHMARK_MAX_STREAMS 8
#define BENCHMARK_ROWS_MIN 1
// 10**27 rows, 8 Byte per entry -> 1 GB per column
#define BENCHMARK_ROWS_MAX (1 << 27)
#define BENCHMARK_GROUPS_BITS_MIN 2
#define BENCHMARK_GROUPS_BITS_MAX 7

#define BENCHMARK_GROUPS_MAX (((size_t)1) << BENCHMARK_GROUPS_BITS_MAX)

struct bench_data {
    union { // screw RAII
        std::unordered_map<uint64_t, uint64_t> expected_output;
    };

    db_table input_cpu;
    db_table output_cpu;

    cudaStream_t streams[BENCHMARK_MAX_STREAMS];
    cudaEvent_t events[BENCHMARK_MAX_STREAMS];

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
    RELASE_ASSERT(
        (new (&bd->expected_output) std::unordered_map<uint64_t, uint64_t>()));

    alloc_db_table_cpu(&bd->input_cpu, BENCHMARK_ROWS_MAX);
    alloc_db_table_cpu(&bd->output_cpu, BENCHMARK_GROUPS_MAX);

    for (int i = 0; i < BENCHMARK_MAX_STREAMS; i++) {
        CUDA_TRY(cudaStreamCreate(&bd->streams[i]));
        CUDA_TRY(cudaEventCreate(&bd->events[i]));
    }

    CUDA_TRY(cudaEventCreate(&bd->start_event));
    CUDA_TRY(cudaEventCreate(&bd->end_event));

    gpu_data_alloc(&bd->data_gpu, BENCHMARK_ROWS_MAX, BENCHMARK_GROUPS_MAX);
}

void free_bench_data(bench_data* bd)
{
    gpu_data_free(&bd->data_gpu);

    CUDA_TRY(cudaEventDestroy(bd->end_event));
    CUDA_TRY(cudaEventDestroy(bd->start_event));

    for (int i = 0; i < BENCHMARK_MAX_STREAMS; i++) {
        CUDA_TRY(cudaEventDestroy(bd->events[i]));
        CUDA_TRY(cudaStreamDestroy(bd->streams[i]));
    }

    free_db_table_cpu(&bd->output_cpu);
    free_db_table_cpu(&bd->input_cpu);

    bd->expected_output.~unordered_map();
}

void setup_bench_data(bench_data* bd, size_t group_count, size_t row_count)
{
    std::mt19937_64 generator{1337};

    std::uniform_int_distribution<uint64_t> uint_rng{BENCHMARK_GROUP_VAL_MIN,
                                                     BENCHMARK_GROUP_VAL_MAX};

    std::uniform_int_distribution<uint64_t> group_rng{0, group_count - 1};

    std::vector<uint64_t> groups{};
    groups.reserve(group_count);

    // generate group_count different group values
    // (duplicates just mean less groups, no big deal)
    bd->expected_output.clear();
    for (uint64_t i = 0; i < group_count; i++) {
        groups.push_back(uint_rng(generator));
    }

    // initialize input table with random group and aggregate values
    for (uint64_t i = 0; i < row_count; i++) {
        uint64_t group = groups[group_rng(generator)];
        uint64_t val = uint_rng(generator);
        bd->input_cpu.group_col[i] = group;
        bd->input_cpu.aggregate_col[i] = val;
        if (bd->expected_output.find(group) != bd->expected_output.end()) {
            bd->expected_output[group] += val;
        }
        else {
            bd->expected_output[group] = val;
        }
    }
    // copy the input to the gpu
    CUDA_TRY(cudaMemcpy(
        bd->data_gpu.input.group_col, bd->input_cpu.group_col,
        row_count * sizeof(uint64_t), cudaMemcpyHostToDevice));
    CUDA_TRY(cudaMemcpy(
        bd->data_gpu.input.aggregate_col, bd->input_cpu.aggregate_col,
        row_count * sizeof(uint64_t), cudaMemcpyHostToDevice));

    // fix up the row counts
    bd->input_cpu.row_count = row_count;
    bd->data_gpu.input.row_count = row_count;
}
bool validate(bench_data* bd)
{
    for (size_t i = 0; i < bd->output_cpu.row_count; i++) {
        uint64_t group = bd->output_cpu.group_col[i];
        auto expected = bd->expected_output.find(group);
        uint64_t got = bd->output_cpu.aggregate_col[i];
        if (expected == bd->expected_output.end()) {
            fprintf(
                stderr,
                "validation failiure: found unexpected group %llu in output "
                "index %llu\n",
                group, i);
            return false;
        }
        else if (expected->second != got) {
            fprintf(
                stderr,
                "validation failiure for group %llu: expected %llu, got "
                "%llu\n",
                group, expected->second, got);
            return false;
        }
    }
    if (bd->output_cpu.row_count != bd->expected_output.size()) {
        fprintf(
            stderr, "validation failiure: expected %llu groups, got %llu\n",
            bd->expected_output.size(), bd->output_cpu.row_count);
    }
    return true;
}

void run_benchmarks(bench_data* bd)
{
    setup_bench_data(bd, BENCHMARK_GROUPS_MAX, BENCHMARK_ROWS_MAX);
    group_by_hashtable<BENCHMARK_GROUPS_BITS_MAX>(
        &bd->data_gpu, 128, 1, 1, bd->streams, bd->events, bd->start_event,
        bd->end_event);
    float time;
    CUDA_TRY(cudaEventElapsedTime(&time, bd->start_event, bd->end_event));

    bd->output_cpu.row_count = bd->data_gpu.output.row_count;
    cudaMemcpy(
        bd->output_cpu.group_col, bd->data_gpu.output.group_col,
        bd->output_cpu.row_count * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(
        bd->output_cpu.aggregate_col, bd->data_gpu.output.aggregate_col,
        bd->output_cpu.row_count * sizeof(uint64_t), cudaMemcpyDeviceToHost);

    validate(bd);

    printf(
        "elapsed: %f, creating %llu groups out of %llu rows\n", time,
        bd->output_cpu.row_count, bd->input_cpu.row_count);
}

int main()
{
    bench_data bench_data;
    alloc_bench_data(&bench_data);
    run_benchmarks(&bench_data);
    free_bench_data(&bench_data);
    return 0;
}
