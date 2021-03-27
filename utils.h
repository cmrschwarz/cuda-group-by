#pragma once
#define CUDA_TRY(expr)                                                         \
    do {                                                                       \
        cudaError_t _err_ = (expr);                                            \
        if (_err_ != cudaSuccess) {                                            \
            report_cuda_error(_err_, #expr, __FILE__, __LINE__, true);         \
        }                                                                      \
    } while (0)

#define RELASE_ASSERT(expr)                                                    \
    if (!(expr)) do {                                                          \
            fprintf(                                                           \
                stderr, "%s:%s: assertion failed: '%s'\n", __FILE__, __LINE__, \
                #expr);                                                        \
            exit(1);                                                           \
    } while (0)

static inline void report_cuda_error(
    cudaError_t err, const char* cmd, const char* file, int line, bool die)
{
    fprintf(
        stderr, "CUDA Error at %s:%i%s%s\n%s\n", file, line, cmd ? ": " : "",
        cmd ? cmd : "", cudaGetErrorString(err));
    if (die) exit(EXIT_FAILURE);
}
