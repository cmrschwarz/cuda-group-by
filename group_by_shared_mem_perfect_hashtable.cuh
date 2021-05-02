#pragma once
#include <unordered_map>
#include "group_by_hashtable.cuh"

#define SHARED_MEM_PHT_L2_ENTRY_BITS 3
#define SHARED_MEM_PHT_EMPTY_GROUP_VAL ((uint64_t)-3)
#define SHARED_MEM_PHT_L1_OVERSIZE_BITS 1
#define SHARED_MEM_PHT_L2_OVERSIZE_BITS 1
#define SHARED_MEM_PHT_L2_TOTAL_MEM_OVERSIZE_BITS 3
#define SHARED_MEM_PHT_MAX_GROUP_BITS                                          \
    (CUDA_SHARED_MEM_BITS_PER_BLOCK -                                          \
     SHARED_MEM_PHT_L2_TOTAL_MEM_OVERSIZE_BITS - SHARED_MEM_PHT_L2_ENTRY_BITS)

// for the algorithms used here, see
// https://www.cs.cmu.edu/~avrim/451f11/lectures/lect1004.pdf
/*
// stores the first prime larger than any power of two from 2^0 through 2^31
// (is always smaller than the next power of two thanks to the
// Bertrand–Chebyshev theorem)
// generated by gen_perfect_hash_primes.hs
static const uint64_t PERFECT_HASH_PRIMES = {
    2,          3,          5,        11,        17,         37,
    67,         -131,       257,      521,       1031,       2053,
    4099,       8209,       -16411,   32771,     65537,      131101,
    262147,     524309,     1048583,  -2097169,  4194319,    8388617,
    16777259,   33554467,   67108879, 134217757, -268435459, 536870923,
    1073741827, 2147483659,
};

#define PH_SM_HT_PRIMES_MAX_BITS                                               \
    (sizeof(PH_SM_HT_PRIMES) / sizeof(uint64_t) - 1)

static_assert(PH_SM_HT_MAX_GROUP_BITS <= PH_SM_HT_PRIMES_MAX_BITS);
*/

// https://en.wikipedia.org/wiki/Universal_hashing#avoiding_modular_arithmetic
static inline __device__ __host__ size_t
almost_universal_hash(uint64_t x, uint64_t a, int bits)
{
    return (size_t)(x * a) >> (64 - bits);
}

struct shared_mem_pht_l1_entry {
    uint32_t offset;
    uint32_t capacity_bits;
    union {
        uint64_t hash_const;
        // used during table construction
        // different name for code readability
        size_t element_count;
    };
};

struct shared_mem_pht_l2_occurance_map_entry {
    uint64_t group;
    union {
        size_t min_rowcount_occuring;
        // used during table construction
        // different name for code readability
        bool occuring;
    };
};

static shared_mem_pht_l1_entry* pht_l1;
static shared_mem_pht_l1_entry* pht_l1_dev;
static uint64_t pht_l1_hash_const;

static shared_mem_pht_l2_occurance_map_entry* pht_l2_occurance_map;
static shared_mem_pht_l2_occurance_map_entry* pht_l2_occurance_map_dev;
// since this is only applicable for small group counts,
// no time was spent on optimizing this
static inline void build_perfect_hashtable(
    const std::unordered_map<uint64_t, uint64_t>* expected_groups,
    const size_t* row_count_variants, int row_count_variant_count,
    int group_bits)
{
    std::uniform_int_distribution<uint64_t> dist{
        std::numeric_limits<uint64_t>::min(),
        std::numeric_limits<uint64_t>::max()};
    std::mt19937_64 rng{12};
    if (group_bits > SHARED_MEM_PHT_MAX_GROUP_BITS) return;
    int l1_table_bits = group_bits + SHARED_MEM_PHT_L1_OVERSIZE_BITS;
    size_t l1_table_capacity = ((size_t)1 << l1_table_bits);
    size_t l1_table_size = l1_table_capacity * sizeof(shared_mem_pht_l1_entry);
    size_t l2_combined_table_capacity =
        (size_t)1 << (group_bits + SHARED_MEM_PHT_L2_TOTAL_MEM_OVERSIZE_BITS);
    size_t l2_occurance_map_size =
        l2_combined_table_capacity *
        sizeof(shared_mem_pht_l2_occurance_map_entry);
    for (uint64_t l1_attemps = l1_table_capacity * l1_table_capacity;
         l1_attemps > 0; l1_attemps--) {
        uint64_t a = dist(rng);
        a += (1 - a % 2); // odd number wanted
        memset(pht_l1, 0, l1_table_size);
        for (auto kv : expected_groups[row_count_variant_count - 1]) {
            size_t hash = almost_universal_hash(kv.first, a, l1_table_bits);
            pht_l1[hash].element_count++;
        }
        for (size_t i = 0; i < l1_table_capacity - 1; i++) {
            pht_l1[i + 1].offset = pht_l1[i].offset + pht_l1[i].element_count;
            pht_l1[i].element_count = 0; // reset for next stage
        }
        for (auto kv : expected_groups[row_count_variant_count - 1]) {
            size_t hash = almost_universal_hash(kv.first, a, l1_table_bits);
            shared_mem_pht_l1_entry* l1e = &pht_l1[hash];
            pht_l2_occurance_map[l1e->offset + l1e->element_count].group =
                kv.first;
            l1e->element_count++;
        }
        uint64_t combined_capacity = 0;

        bool fail = false;
        for (size_t i = 0; i < l1_table_capacity; i++) {
            shared_mem_pht_l1_entry* l1e = &pht_l1[i];
            uint64_t group_vals_offset = l1e->offset;
            l1e->offset = combined_capacity;
            if (l1e->element_count == 0) continue;
            if (l1e->element_count == 1) {
                l1e->hash_const = 0; // will always produce hash 0
                l1e->capacity_bits = 1; // doesn't really matter
                combined_capacity += 1;
                continue;
            }
            int cap_bits_min = log2(ceil_to_pow_two(l1e->element_count));
            bool success = false;
            // some extra leeway with the capacity
            for (int c = cap_bits_min; c <= 3 * cap_bits_min; c++) {
                size_t capacity = (size_t)1 << c;
                if (combined_capacity + capacity > l2_combined_table_capacity) {
                    break;
                }
                for (uint64_t l2_attemps = capacity * capacity; l2_attemps > 0;
                     l2_attemps--) {
                    uint64_t l2a = dist(rng);
                    l2a += (1 - l2a % 2); // odd number wanted
                    for (size_t i = 0; i < capacity; i++) {
                        pht_l2_occurance_map[l1e->offset + i].occuring = false;
                    }
                    bool collision = false;
                    for (size_t i = 0; i < l1e->element_count; i++) {
                        size_t hash = almost_universal_hash(
                            pht_l2_occurance_map[group_vals_offset + i].group,
                            l2a, c);
                        if (pht_l2_occurance_map[l1e->offset + hash].occuring) {
                            collision = true;
                            break;
                        }
                        pht_l2_occurance_map[l1e->offset + hash].occuring =
                            true;
                    }
                    if (collision) continue;
                    success = true;
                    l1e->hash_const = l2a;
                    break;
                }
                if (success) {
                    l1e->capacity_bits = c;
                    combined_capacity += capacity;
                    break;
                }
            }
            if (!success) {
                fail = true;
                break;
            }
        }
        if (fail) continue;
        memset(pht_l2_occurance_map, 0, l2_occurance_map_size);
        for (auto kv : expected_groups[row_count_variant_count - 1]) {
            size_t l1_hash = almost_universal_hash(kv.first, a, l1_table_bits);
            shared_mem_pht_l1_entry* l1e = &pht_l1[l1_hash];
            size_t l2_hash = almost_universal_hash(
                kv.first, l1e->hash_const, l1e->capacity_bits);
            for (int i = 0; i < row_count_variant_count; i++) {
                if (expected_groups[i].find(kv.first) !=
                    expected_groups[i].end()) {
                    pht_l2_occurance_map[l1e->offset + l2_hash]
                        .min_rowcount_occuring = row_count_variants[i];
                    pht_l2_occurance_map[l1e->offset + l2_hash].group =
                        kv.first;
                    break;
                }
            }
        }
        cudaMemcpy(
            pht_l2_occurance_map_dev, pht_l2_occurance_map,
            l2_occurance_map_size, cudaMemcpyHostToDevice);
        cudaMemcpy(pht_l1_dev, pht_l1, l1_table_size, cudaMemcpyHostToDevice);
        pht_l1_hash_const = a;
        return;
    }
    // we failed to build a perfect hashtable :( [should be very unlikely]
    RELASE_ASSERT(false);
}

static inline void group_by_shared_mem_perfect_hashtable_init(size_t max_groups)
{
    group_by_hashtable_init(max_groups);
    max_groups = ceil_to_pow_two(max_groups);
    int group_bits = log2(max_groups);
    if (group_bits > SHARED_MEM_PHT_MAX_GROUP_BITS) {
        group_bits = SHARED_MEM_PHT_MAX_GROUP_BITS;
        max_groups = (size_t)1 << SHARED_MEM_PHT_MAX_GROUP_BITS;
    }
    size_t l1_table_size = (max_groups << SHARED_MEM_PHT_L1_OVERSIZE_BITS) *
                           sizeof(shared_mem_pht_l1_entry);

    pht_l1 = (shared_mem_pht_l1_entry*)malloc(l1_table_size);
    RELASE_ASSERT(pht_l1);
    CUDA_TRY(cudaMalloc(&pht_l1_dev, l1_table_size));

    size_t l2_occurance_map_size =
        (max_groups << SHARED_MEM_PHT_L2_TOTAL_MEM_OVERSIZE_BITS) *
        sizeof(shared_mem_pht_l2_occurance_map_entry);
    pht_l2_occurance_map =
        (shared_mem_pht_l2_occurance_map_entry*)malloc(l2_occurance_map_size);
    RELASE_ASSERT(pht_l2_occurance_map);
    CUDA_TRY(cudaMalloc(&pht_l2_occurance_map_dev, l2_occurance_map_size));
}

static inline void group_by_shared_mem_perfect_hashtable_fin()
{
    CUDA_TRY(cudaFree(pht_l2_occurance_map_dev));
    free(pht_l2_occurance_map);
    CUDA_TRY(cudaFree(pht_l1_dev));
    free(pht_l1);
    group_by_hashtable_fin();
}

static inline bool approach_shared_mem_perfect_hashtable_available(
    int group_bits, int row_count, int grid_dim, int block_dim,
    int stream_count)
{
    if (!grid_dim || !block_dim) return false;
    return group_bits <= SHARED_MEM_PHT_MAX_GROUP_BITS;
}

template <int MAX_GROUP_BITS>
__global__ void kernel_shared_mem_pht(
    db_table input, shared_mem_pht_l1_entry* l1_ht, uint64_t l1_hash_const,
    shared_mem_pht_l2_occurance_map_entry* pht_l2_occurance_map,
    group_ht_entry<>* hashtable, int stream_count, int stream_idx)
{
    // the ternary guards against template instantiations that would
    // cause ptxas error during compilations by requiring
    // too much shared memory even if these instantiations are never used
    constexpr size_t L2_CAPACITY =
        MAX_GROUP_BITS <= SHARED_MEM_PHT_MAX_GROUP_BITS
            ? (size_t)1
                  << (MAX_GROUP_BITS +
                      SHARED_MEM_PHT_L2_TOTAL_MEM_OVERSIZE_BITS)
            : 1;

    constexpr size_t L1_BITS = MAX_GROUP_BITS + SHARED_MEM_PHT_L1_OVERSIZE_BITS;

    __shared__ uint64_t shared_aggregates[L2_CAPACITY];

    int tid = threadIdx.x + blockIdx.x * blockDim.x +
              stream_idx * blockDim.x * gridDim.x;
    int stride = blockDim.x * gridDim.x * stream_count;

    for (int i = threadIdx.x; i < L2_CAPACITY; i += blockDim.x) {
        shared_aggregates[i] = 0;
    }
    __syncthreads();

    for (size_t i = tid; i < input.row_count; i += stride) {
        uint64_t group = input.group_col[i];
        size_t l1_hash = almost_universal_hash(group, l1_hash_const, L1_BITS);

        shared_mem_pht_l1_entry* l1e = &l1_ht[l1_hash];
        size_t l2_hash =
            almost_universal_hash(group, l1e->hash_const, l1e->capacity_bits);
        atomicAdd(
            (cudaUInt64_t*)&shared_aggregates[l2_hash + l1e->offset],
            input.aggregate_col[i]);
    }

    __syncthreads();
    for (int i = threadIdx.x; i < L2_CAPACITY; i += blockDim.x) {
        // check if the group value occurs in the perfect hashtable
        // at or above our row count
        size_t min_rc = pht_l2_occurance_map[i].min_rowcount_occuring;
        if (min_rc != 0 && min_rc <= input.row_count) {
            group_ht_insert<MAX_GROUP_BITS, false>(
                hashtable, pht_l2_occurance_map[i].group, shared_aggregates[i]);
        }
    }
}

template <int MAX_GROUP_BITS>
void group_by_shared_mem_perfect_hashtable(
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
        kernel_shared_mem_pht<MAX_GROUP_BITS>
            <<<grid_dim, block_dim, 0, stream>>>(
                gd->input, pht_l1_dev, pht_l1_hash_const,
                pht_l2_occurance_map_dev, group_ht_entry<>::table,
                actual_stream_count, i);
        // if we have only one stream there is no need for waiting events
        if (stream_count > 1) cudaEventRecord(events[i], stream);
    }
    group_by_hashtable_writeout<MAX_GROUP_BITS>(
        gd, grid_dim, block_dim, stream_count, streams, events, start_event,
        end_event);
}
