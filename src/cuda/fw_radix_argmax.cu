#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

constexpr int WARP_SIZE  = 32;

template<int RADIX_BITS>
__device__ __forceinline__
int get_radix_bin(uint32_t key, int shift)
{
    constexpr int NUM_BINS = 1 << RADIX_BITS;
    return (key >> shift) & (NUM_BINS - 1);
}

__device__ __forceinline__
float warp_max(float val)
{
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

template<int THREADS>
__global__
void fw_init_kernel(const float* logits, float* H, int K)
{
    constexpr int NUM_WARPS = THREADS / WARP_SIZE;
    int row_id = blockIdx.x;
    int tid = threadIdx.x;
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;

    float local_max = -INFINITY;
    
    __shared__ float warp_maxes[NUM_WARPS];

    for (int i = tid; i < K; i += blockDim.x) {
        float z = logits[row_id * K + i];
        local_max = fmaxf(local_max, z);
    }

    local_max = warp_max(local_max);
    if (lane == 0) {
        warp_maxes[warp] = local_max;
    }

    __syncthreads();

    if (warp == 0) {
        float val = (lane < NUM_WARPS) ? warp_maxes[lane] : -INFINITY;
        val = warp_max(val);
        if (lane == 0) {
            H[row_id] = val - 1.0f;
        }
    }

}

__device__ __forceinline__
uint32_t float_to_radix_bits(float z)
{
    uint32_t bits = __float_as_uint(z);
    if (bits & 0x80000000u) {
        return ~bits;
    } else {
        return bits ^ 0x80000000u;
    }
}

__device__ __forceinline__
float radix_bits_to_float(uint32_t key)
{
    uint32_t bits;
    if (key & 0x80000000u) {
        bits = key ^ 0x80000000u;
    }
    else {
        bits = ~key;
    }
    return __uint_as_float(bits);
}

template<int NUM_WARPS, int NUM_BINS>
__device__ __forceinline__
void reset_warp_histograms(int warp_counts[NUM_WARPS][NUM_BINS], float warp_sums[NUM_WARPS][NUM_BINS],  uint32_t warp_min_keys[NUM_WARPS][NUM_BINS], uint32_t warp_max_keys[NUM_WARPS][NUM_BINS],int tid)
{
    for (int i = tid; i < NUM_WARPS * NUM_BINS; i += blockDim.x) {
        int w = i / NUM_BINS;
        int b = i % NUM_BINS;
        warp_counts[w][b] = 0;
        warp_sums[w][b]   = 0.0f;
        warp_min_keys[w][b] = 0xFFFFFFFFu;
        warp_max_keys[w][b] = 0u;
    }
}

template<int NUM_PARTITIONS, int NUM_BINS>
__device__ __forceinline__
void reset_partition_histograms(
    int part_counts[NUM_PARTITIONS][NUM_BINS],
    float part_sums[NUM_PARTITIONS][NUM_BINS],
    uint32_t part_min_keys[NUM_PARTITIONS][NUM_BINS],
    uint32_t part_max_keys[NUM_PARTITIONS][NUM_BINS],
    int tid
)
{
    for (int i = tid;
         i < NUM_PARTITIONS * NUM_BINS;
         i += blockDim.x)
    {
        int p = i / NUM_BINS;
        int b = i % NUM_BINS;

        part_counts[p][b] = 0;
        part_sums[p][b] = 0.0f;
        part_min_keys[p][b] = 0xFFFFFFFFu;
        part_max_keys[p][b] = 0u;
    }
}

template<int RADIX_BITS, int NUM_WARPS, int NUM_BINS>
__device__ __forceinline__
void accumulate_warp_histograms(const float* logits, int row_id, int K, int num_values, float threshold, int shift, int warp_counts[NUM_WARPS][NUM_BINS], float warp_sums[NUM_WARPS][NUM_BINS],
    uint32_t warp_min_keys[NUM_WARPS][NUM_BINS], uint32_t warp_max_keys[NUM_WARPS][NUM_BINS],
    uint32_t prefix, uint32_t prefix_mask )
{
    int tid  = threadIdx.x;
    int warp = threadIdx.x >> 5;

    for (int i = tid; i < num_values; i += blockDim.x) {
        float z = logits[row_id * K + i];

        if (z > threshold) {
            uint32_t key = float_to_radix_bits(z);
            if ((key & prefix_mask) == prefix) {
                int bin = get_radix_bin<RADIX_BITS>(key, shift);

                atomicAdd(&warp_counts[warp][bin], 1);
                atomicAdd(&warp_sums[warp][bin], z);
                atomicMin(&warp_min_keys[warp][bin], key);
                atomicMax(&warp_max_keys[warp][bin], key);
            }
        }
    }
}


template<int NUM_WARPS, int NUM_BINS>
__device__ __forceinline__
void reduce_warp_histograms(int warp_counts[NUM_WARPS][NUM_BINS], float warp_sums[NUM_WARPS][NUM_BINS], uint32_t warp_min_keys[NUM_WARPS][NUM_BINS],
    uint32_t warp_max_keys[NUM_WARPS][NUM_BINS], int bin_counts[NUM_BINS], float bin_sums[NUM_BINS], uint32_t bin_min_keys[NUM_BINS],uint32_t bin_max_keys[NUM_BINS])
{
    int tid = threadIdx.x;

    for (int b = tid; b < NUM_BINS; b += blockDim.x) {
        int count = 0;
        float sum = 0.0f;
        uint32_t min_key = 0xFFFFFFFFu;
        uint32_t max_key = 0u;

        for (int w = 0; w < NUM_WARPS; ++w) {
            count += warp_counts[w][b];
            sum   += warp_sums[w][b];
            min_key = min(min_key, warp_min_keys[w][b]);
            max_key = max(max_key, warp_max_keys[w][b]);
        }

        bin_counts[b] = count;
        bin_sums[b]   = sum;
        bin_min_keys[b] = min_key;
        bin_max_keys[b] = max_key;
    }
}

template<int NUM_PARTITIONS, int NUM_BINS>
__device__ __forceinline__
void reduce_partition_histograms(
    int part_counts[NUM_PARTITIONS][NUM_BINS],
    float part_sums[NUM_PARTITIONS][NUM_BINS],
    uint32_t part_min_keys[NUM_PARTITIONS][NUM_BINS],
    uint32_t part_max_keys[NUM_PARTITIONS][NUM_BINS],

    int bin_counts[NUM_BINS],
    float bin_sums[NUM_BINS],
    uint32_t bin_min_keys[NUM_BINS],
    uint32_t bin_max_keys[NUM_BINS]
)
{
    int tid = threadIdx.x;

    for (int b = tid; b < NUM_BINS; b += blockDim.x) {

        int count = 0;
        float sum = 0.0f;
        uint32_t min_key = 0xFFFFFFFFu;
        uint32_t max_key = 0u;

        #pragma unroll
        for (int p = 0; p < NUM_PARTITIONS; ++p) {

            count += part_counts[p][b];
            sum   += part_sums[p][b];

            min_key = min(
                min_key,
                part_min_keys[p][b]
            );

            max_key = max(
                max_key,
                part_max_keys[p][b]
            );
        }

        bin_counts[b]   = count;
        bin_sums[b]     = sum;
        bin_min_keys[b] = min_key;
        bin_max_keys[b] = max_key;
    }
}

template<
    int RADIX_BITS,
    int NUM_WARPS,
    int NUM_PARTITIONS,
    int NUM_BINS
>
__device__ __forceinline__
void accumulate_partition_histograms(
    const float* logits,
    int row_id,
    int K,
    int num_values,
    float threshold,
    int shift,

    int part_counts[NUM_PARTITIONS][NUM_BINS],
    float part_sums[NUM_PARTITIONS][NUM_BINS],
    uint32_t part_min_keys[NUM_PARTITIONS][NUM_BINS],
    uint32_t part_max_keys[NUM_PARTITIONS][NUM_BINS],

    uint32_t prefix,
    uint32_t prefix_mask
)
{
    int tid  = threadIdx.x;
    int warp = tid >> 5;

    // Map warps to histogram partitions.
    //
    // Example:
    // NUM_WARPS = 8, NUM_PARTITIONS = 4
    //
    // warp 0,1 -> partition 0
    // warp 2,3 -> partition 1
    // warp 4,5 -> partition 2
    // warp 6,7 -> partition 3
    //
    int partition =
        (warp * NUM_PARTITIONS) / NUM_WARPS;

    for (int i = tid; i < num_values; i += blockDim.x) {

        float z = logits[row_id * K + i];

        if (z > threshold) {

            uint32_t key = float_to_radix_bits(z);

            if ((key & prefix_mask) == prefix) {

                int bin =
                    get_radix_bin<RADIX_BITS>(
                        key,
                        shift
                    );

                atomicAdd(
                    &part_counts[partition][bin],
                    1
                );

                atomicAdd(
                    &part_sums[partition][bin],
                    z
                );

                atomicMin(
                    &part_min_keys[partition][bin],
                    key
                );

                atomicMax(
                    &part_max_keys[partition][bin],
                    key
                );
            }
        }
    }
}

template<int RADIX_BITS, int THREADS, int BLOCKS_PER_ROW>
__global__ void fw_precompact_histogram_kernel(
    const float* __restrict__ logits,
    int K,
    int* __restrict__ partial_counts,
    float* __restrict__ partial_sums,
    uint32_t* __restrict__ partial_min_keys,
    uint32_t* __restrict__ partial_max_keys)
{
    constexpr int NUM_BINS = 1 << RADIX_BITS;
    constexpr int SHIFT = 32 - RADIX_BITS;

    constexpr int NUM_WARPS = THREADS / WARP_SIZE;
    constexpr int NUM_PARTITIONS = 8;

    int row = blockIdx.x / BLOCKS_PER_ROW;
    int block_in_row = blockIdx.x % BLOCKS_PER_ROW;
    int tid = threadIdx.x;
    int warp = tid >> 5;

    int partition =
        (warp * NUM_PARTITIONS) / NUM_WARPS;

    __shared__ int part_counts[NUM_PARTITIONS][NUM_BINS];
    __shared__ float part_sums[NUM_PARTITIONS][NUM_BINS];
    __shared__ uint32_t part_min_keys[NUM_PARTITIONS][NUM_BINS];
    __shared__ uint32_t part_max_keys[NUM_PARTITIONS][NUM_BINS];

    //
    // Initialize all partition histograms
    //
    for (int i = tid;
         i < NUM_PARTITIONS * NUM_BINS;
         i += blockDim.x)
    {
        int p = i / NUM_BINS;
        int b = i % NUM_BINS;

        part_counts[p][b] = 0;
        part_sums[p][b] = 0.0f;
        part_min_keys[p][b] = 0xFFFFFFFFu;
        part_max_keys[p][b] = 0u;
    }

    __syncthreads();

    int scan_tid =
        block_in_row * THREADS + tid;

    int scan_stride =
        BLOCKS_PER_ROW * THREADS;

    //
    // Accumulate into private partition
    //
    for (int i = scan_tid; i < K; i += scan_stride) {

        float z = logits[row * K + i];

        uint32_t key =
            float_to_radix_bits(z);

        int bin =
            get_radix_bin<RADIX_BITS>(
                key,
                SHIFT
            );

        atomicAdd(
            &part_counts[partition][bin],
            1
        );

        atomicAdd(
            &part_sums[partition][bin],
            z
        );

        atomicMin(
            &part_min_keys[partition][bin],
            key
        );

        atomicMax(
            &part_max_keys[partition][bin],
            key
        );
    }

    __syncthreads();

    //
    // Reduce the partitions and write output
    //
    for (int b = tid; b < NUM_BINS; b += blockDim.x) {

        int count = 0;
        float sum = 0.0f;
        uint32_t min_key = 0xFFFFFFFFu;
        uint32_t max_key = 0u;

        #pragma unroll
        for (int p = 0; p < NUM_PARTITIONS; ++p) {

            count += part_counts[p][b];
            sum += part_sums[p][b];

            min_key =
                min(min_key,
                    part_min_keys[p][b]);

            max_key =
                max(max_key,
                    part_max_keys[p][b]);
        }

        size_t idx =
            ((size_t)row * BLOCKS_PER_ROW
             + block_in_row)
            * NUM_BINS + b;

        partial_counts[idx] = count;
        partial_sums[idx] = sum;
        partial_min_keys[idx] = min_key;
        partial_max_keys[idx] = max_key;
    }
}


// template<int RADIX_BITS, int THREADS, int BLOCKS_PER_ROW>
// __global__ void fw_precompact_histogram_kernel(
//     const float* __restrict__ logits,
//     int K,
//     int* __restrict__ partial_counts,
//     float* __restrict__ partial_sums,
//     uint32_t* __restrict__ partial_min_keys,
//     uint32_t* __restrict__ partial_max_keys)
// {
//     constexpr int NUM_BINS = 1 << RADIX_BITS;
//     constexpr int SHIFT = 32 - RADIX_BITS;

//     int row = blockIdx.x / BLOCKS_PER_ROW;
//     int block_in_row = blockIdx.x % BLOCKS_PER_ROW;
//     int tid = threadIdx.x;

//     __shared__ int bin_counts[NUM_BINS];
//     __shared__ float bin_sums[NUM_BINS];
//     __shared__ uint32_t bin_min_keys[NUM_BINS];
//     __shared__ uint32_t bin_max_keys[NUM_BINS];

//     for (int b = tid; b < NUM_BINS; b += blockDim.x) {
//         bin_counts[b] = 0;
//         bin_sums[b] = 0.0f;
//         bin_min_keys[b] = 0xFFFFFFFFu;
//         bin_max_keys[b] = 0u;
//     }

//     __syncthreads();

//     int scan_tid =
//         block_in_row * THREADS + tid;

//     int scan_stride =
//         BLOCKS_PER_ROW * THREADS;

//     for (int i = scan_tid; i < K; i += scan_stride) {

//         float z = logits[row * K + i];

//         uint32_t key = float_to_radix_bits(z);

//         int bin = get_radix_bin<RADIX_BITS>(key, SHIFT);

//         atomicAdd(&bin_counts[bin], 1);
//         atomicAdd(&bin_sums[bin], z);
//         atomicMin(&bin_min_keys[bin], key);
//         atomicMax(&bin_max_keys[bin], key);
//     }

//     __syncthreads();

//     for (int b = tid; b < NUM_BINS; b += blockDim.x) {

//         size_t idx =
//             ((size_t)row * BLOCKS_PER_ROW + block_in_row)
//             * NUM_BINS + b;

//         partial_counts[idx] = bin_counts[b];
//         partial_sums[idx] = bin_sums[b];
//         partial_min_keys[idx] = bin_min_keys[b];
//         partial_max_keys[idx] = bin_max_keys[b];
//     }
// }

template<int RADIX_BITS, int THREADS, int BLOCKS_PER_ROW>
__global__ void fw_precompact_reduce_kernel(
    const int* __restrict__ partial_counts,
    const float* __restrict__ partial_sums,
    const uint32_t* __restrict__ partial_min_keys,
    const uint32_t* __restrict__ partial_max_keys,
    float* __restrict__ H,
    int* __restrict__ pre_C,
    float* __restrict__ pre_S,
    uint32_t* pre_prefix,
    uint32_t* pre_prefix_mask,
    float* __restrict__ pre_threshold)
{
    constexpr int NUM_BINS = 1 << RADIX_BITS;

    int row = blockIdx.x;
    int tid = threadIdx.x;

    __shared__ int bin_counts[NUM_BINS];
    __shared__ float bin_sums[NUM_BINS];
    __shared__ uint32_t bin_min_keys[NUM_BINS];
    __shared__ uint32_t bin_max_keys[NUM_BINS];

   
    // Combine partial histograms from all blocks of this row
    for (int b = tid; b < NUM_BINS; b += blockDim.x) {

        int count = 0;
        float sum = 0.0f;

        uint32_t min_key = 0xFFFFFFFFu;
        uint32_t max_key = 0u;

        for (int p = 0; p < BLOCKS_PER_ROW; ++p) {

            size_t idx =
                ((size_t)row * BLOCKS_PER_ROW + p)
                * NUM_BINS + b;

            int c = partial_counts[idx];

            if (c > 0) {
                count += c;
                sum += partial_sums[idx];

                min_key =
                    min(min_key, partial_min_keys[idx]);

                max_key =
                    max(max_key, partial_max_keys[idx]);
            }
        }

        bin_counts[b] = count;
        bin_sums[b] = sum;
        bin_min_keys[b] = min_key;
        bin_max_keys[b] = max_key;
    }

    __syncthreads();

    // One thread walks the bins
    if (tid == 0) {

        float h = 0.0f; 
        float refine_threshold = 0.0f;
        int C = 0;
        float S = 0.0f;

        uint32_t prefix = 0u;
        uint32_t prefix_mask = 0u;

        for (int b = NUM_BINS - 1; b >= 0; --b) {

            if (bin_counts[b] == 0)
                continue;

            int candidate_C =
                C + bin_counts[b];

            float candidate_S =
                S + bin_sums[b];

            float new_h =
                (candidate_S - 1.0f)
                / candidate_C;

            uint32_t new_h_key =
                float_to_radix_bits(new_h);

            uint32_t min_key =
                bin_min_keys[b];

            uint32_t max_key =
                bin_max_keys[b];

            // ACCEPT
            if (new_h_key < min_key) {
                C = candidate_C;
                S = candidate_S;
                h = new_h;
                continue;
            }

            // REFINE
            else if (new_h_key < max_key) {
                // if (row == 0) {
                //     printf("REFINE: bin=%d C_before=%d bin_count=%d " "h_before=%f new_h=%f\n",  b, C, bin_counts[b], h, new_h);
                // }  
                if (C == 0) {
                    float Z1 = radix_bits_to_float(max_key);
                    h = Z1 - 1.0f;
                } 
                constexpr int SHIFT = 32 - RADIX_BITS; 
                prefix = ((uint32_t)b << SHIFT); 
                prefix_mask = ((uint32_t)(NUM_BINS - 1) << SHIFT);
                refine_threshold = fmaxf(new_h, h);

                break;
            }

            // STOP
            else {
                if (C == 0) {
                    float Z1 = radix_bits_to_float(max_key);
                    h = Z1 - 1.0f;
                    refine_threshold = h;
                }
                break;
            }
        }

        H[row] = h;
        pre_C[row] = C;
        pre_S[row] = S;
        pre_prefix[row] = prefix;
        pre_prefix_mask[row] = prefix_mask;
        pre_threshold[row] = refine_threshold;
    }
}

template<int THREADS, int BLOCKS_PER_ROW>
__global__ void fw_compact_multiblock_kernel(
    const float* logits,
    const float* H,
    const float* pre_threshold,
    int K,
    float* candidate_buffer,
    int* candidate_indices,
    int* candidate_counts)
{
    constexpr int NUM_WARPS = THREADS / 32;

    int row = blockIdx.x / BLOCKS_PER_ROW;
    int block_in_row = blockIdx.x % BLOCKS_PER_ROW;

    int tid  = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;

    float threshold = pre_threshold[row];

    __shared__ int warp_counts[NUM_WARPS];
    __shared__ int warp_offsets[NUM_WARPS];

    __shared__ int block_count;
    __shared__ int block_base;

    int scan_stride = BLOCKS_PER_ROW * THREADS;

    //
    // IMPORTANT:
    // base is common to the whole block.
    //
    for (int base = block_in_row * THREADS;
         base < K;
         base += scan_stride)
    {
        int i = base + tid;

        bool keep = false;
        float z = 0.0f;

        if (i < K) {
            z = logits[row * K + i];
            keep = (z > threshold);
        }

        //
        // Find surviving threads within each warp.
        //
        unsigned mask =
            __ballot_sync(0xffffffffu, keep);

        int warp_count = __popc(mask);

        //
        // Number of surviving lanes BEFORE this lane.
        //
        unsigned lower_mask =
            (lane == 0)
            ? 0u
            : ((1u << lane) - 1u);

        int lane_offset =
            __popc(mask & lower_mask);

        //
        // One thread per warp stores warp survivor count.
        //
        if (lane == 0) {
            warp_counts[warp] = warp_count;
        }

        __syncthreads();

        //
        // Only thread 0 computes offsets across the 8 warps.
        //
        if (tid == 0) {

            int total = 0;

            #pragma unroll
            for (int w = 0; w < NUM_WARPS; ++w) {

                warp_offsets[w] = total;
                total += warp_counts[w];
            }

            block_count = total;

            //
            // ONE global atomic for this whole block chunk.
            //
            if (total > 0) {
                block_base =
                    atomicAdd(
                        &candidate_counts[row],
                        total
                    );
            }
        }

        __syncthreads();

        //
        // Surviving threads now know unique output positions.
        //
        if (keep) {

            int pos =
                block_base
                + warp_offsets[warp]
                + lane_offset;

            candidate_buffer[row * K + pos] = z;

            candidate_indices[row * K + pos] = i;
        }

        __syncthreads();
    }
}

// template<int THREADS, int BLOCKS_PER_ROW>
// __global__ void fw_compact_multiblock_kernel(
//     const float* logits,
//     const float* H,
//     const float* pre_threshold,
//     int K,
//     float* candidate_buffer,
//     int* candidate_indices,
//     int* candidate_counts)
// {
//     int row = blockIdx.x / BLOCKS_PER_ROW;
//     int block_in_row = blockIdx.x % BLOCKS_PER_ROW;
//     int tid = threadIdx.x;
    
//     float threshold = pre_threshold[row];

//     int scan_tid =
//         block_in_row * THREADS + tid;

//     int scan_stride =
//         BLOCKS_PER_ROW * THREADS;

//     for (int i = scan_tid; i < K; i += scan_stride) {

//         float z = logits[row * K + i];

//         if (z > threshold) {

//             int pos = atomicAdd(
//                 &candidate_counts[row],
//                 1
//             );

//             candidate_buffer[row * K + pos] = z;
//             candidate_indices[row * K + pos] = i;
//         }
//     }
// }
// template<int THREADS>
// __global__ void fw_compact_kernel(const float* logits, const float* H, int K, float* candidate_buffer, int* candidate_indices, int* candidate_counts)
// {
//     int row = blockIdx.x;
//     int tid = threadIdx.x;

//     __shared__ int write_idx;

//     if (tid == 0)
//         write_idx = 0;

//     __syncthreads();

//     float h1 = H[row];

//     for (int i = tid; i < K; i += blockDim.x) {
//         float z = logits[row * K + i];

//         if (z > h1) {
//             int pos = atomicAdd(&write_idx, 1);
//             candidate_buffer[row * K + pos] = z;
//             candidate_indices[row * K + pos] = i;

//         }
//     }

//     __syncthreads();

//     if (tid == 0)
//         candidate_counts[row] = write_idx;
// }

template<int THREADS>
__global__ void fw_compact_singleblock_test(
    const float* logits,
    const float* pre_threshold,
    int K,
    float* candidate_buffer,
    int* candidate_indices,
    int* candidate_counts)
{
    int row = blockIdx.x;
    int tid = threadIdx.x;

    __shared__ int write_idx;

    if (tid == 0)
        write_idx = 0;

    __syncthreads();

    float threshold = pre_threshold[row];

    for (int i = tid; i < K; i += blockDim.x) {

        float z = logits[row * K + i];

        if (z > threshold) {
            int pos = atomicAdd(&write_idx, 1);

            candidate_buffer[row * K + pos] = z;
            candidate_indices[row * K + pos] = i;
        }
    }

    __syncthreads();

    if (tid == 0)
        candidate_counts[row] = write_idx;
}

template<int RADIX_BITS, int THREADS>
__global__
void fw_radix_buffer_kernel(const float* logits, float* H, int K, const int64_t* targets, float* losses, int* kappas, float* candidate_buffer, const int* candidate_counts, const int* pre_C, const float* pre_S, const uint32_t* pre_prefix,const uint32_t* pre_prefix_mask, const float* pre_threshold)
{
    int row_id = blockIdx.x;
    int tid = threadIdx.x;

    constexpr int NUM_BINS = 1 << RADIX_BITS;
    constexpr int NUM_WARPS = THREADS / WARP_SIZE; 
    constexpr int INITIAL_SHIFT = 32 - RADIX_BITS;
    __shared__ float histogram_threshold;

    __shared__ int      warp_counts[NUM_WARPS][NUM_BINS];
    __shared__ float    warp_sums[NUM_WARPS][NUM_BINS];
    __shared__ uint32_t warp_min_keys[NUM_WARPS][NUM_BINS];
    __shared__ uint32_t warp_max_keys[NUM_WARPS][NUM_BINS];

    // constexpr int NUM_PARTITIONS = 8;

    // __shared__ int part_counts[NUM_PARTITIONS][NUM_BINS];
    // __shared__ float part_sums[NUM_PARTITIONS][NUM_BINS];
    // __shared__ uint32_t part_min_keys[NUM_PARTITIONS][NUM_BINS];
    // __shared__ uint32_t part_max_keys[NUM_PARTITIONS][NUM_BINS];

    __shared__ int      bin_counts[NUM_BINS];
    __shared__ float    bin_sums[NUM_BINS];
    __shared__ uint32_t bin_min_keys[NUM_BINS];
    __shared__ uint32_t bin_max_keys[NUM_BINS];

    __shared__ float h;
    __shared__ float threshold;
    __shared__ int   C;
    __shared__ float S;
    __shared__ uint32_t prefix;
    __shared__ uint32_t prefix_mask;

    __shared__ int stop;
    __shared__ int refine;
    if (tid == 0) {
        h = H[row_id];
        threshold = pre_threshold[row_id];
        C = pre_C[row_id];
        S = pre_S[row_id];
        prefix = pre_prefix[row_id];
        prefix_mask = pre_prefix_mask[row_id];
        stop = (prefix_mask == 0u) ? 1 : 0;
        refine = 0;
    }

    __syncthreads();

    int shift = INITIAL_SHIFT - RADIX_BITS;   
    int num_passes = 0;

    while (shift >= 0 && !stop) {
        num_passes++;
        if (tid == 0) {
            refine = 0;
            histogram_threshold = threshold;
        }       
        __syncthreads();
        const float* buffer_logits = candidate_buffer;
        int num_values = candidate_counts[row_id];
        // reset_partition_histograms<NUM_PARTITIONS,NUM_BINS>(part_counts,part_sums,part_min_keys,part_max_keys,tid);
        reset_warp_histograms<NUM_WARPS, NUM_BINS>(warp_counts,warp_sums,warp_min_keys,warp_max_keys,tid);
        __syncthreads();

        // accumulate_partition_histograms<RADIX_BITS,NUM_WARPS,NUM_PARTITIONS,NUM_BINS>(buffer_logits,row_id,K,num_values,histogram_threshold,shift,part_counts,part_sums,part_min_keys,part_max_keys,prefix,prefix_mask);
        accumulate_warp_histograms<RADIX_BITS,NUM_WARPS,NUM_BINS>(buffer_logits,row_id,K,num_values,histogram_threshold,shift,warp_counts,warp_sums,warp_min_keys,warp_max_keys,prefix,prefix_mask);
        __syncthreads();

        // if constexpr (NUM_PARTITIONS == 1) {
        //     for (int b = tid; b < NUM_BINS; b += blockDim.x) {
        //         bin_counts[b]   = part_counts[0][b];
        //         bin_sums[b]     = part_sums[0][b];
        //         bin_min_keys[b] = part_min_keys[0][b];
        //         bin_max_keys[b] = part_max_keys[0][b];
        //     }
        // }
        // else{
            // reduce_partition_histograms<NUM_PARTITIONS,NUM_BINS>(part_counts,part_sums,part_min_keys,part_max_keys,bin_counts,bin_sums,bin_min_keys,bin_max_keys);
        // }
        reduce_warp_histograms<NUM_WARPS,NUM_BINS>(warp_counts,warp_sums,warp_min_keys,warp_max_keys,bin_counts,bin_sums,bin_min_keys,bin_max_keys);
        __syncthreads();

        if (tid == 0) {
            int current_bin = NUM_BINS - 1;
            while (current_bin >= 0 &&
                bin_counts[current_bin] == 0) {
                current_bin--;
            }

            if (current_bin < 0) {
                stop = 1;
            }
            else {
                while (current_bin >= 0) {
                    if (bin_counts[current_bin] == 0) {
                        current_bin--;
                        continue;
                    }
                    int candidate_C = C + bin_counts[current_bin];
                    float candidate_S = S + bin_sums[current_bin];
                    float new_h = (candidate_S - 1.0f) / candidate_C;

                    uint32_t new_h_key = float_to_radix_bits(new_h);

                    uint32_t min_key = bin_min_keys[current_bin];
                    uint32_t max_key = bin_max_keys[current_bin];
                    // 1. Whole bin survives new_h
                    if (new_h_key < min_key) {
                        C = candidate_C;
                        S = candidate_S;
                        h = new_h;
                        threshold = new_h;
                        // Try next lower bin
                        current_bin--;
                    }
                    // 2. new_h cuts through this bin
                    else if (new_h_key < max_key) {
                        // printf(
                        //     "row=%d pass=%d candidates=%d refine_bin=%d bin_count=%d\n",
                        //     row_id,
                        //     num_passes,
                        //     num_values,
                        //     current_bin,
                        //     bin_counts[current_bin]
                        // );
                        threshold = new_h;
                        prefix |= ((uint32_t)current_bin << shift);
                        prefix_mask |= ((uint32_t)(NUM_BINS - 1) << shift);
                        refine = 1;  
                        break;
                    }

                    // 3. new_h >= max of this bin
                    else {
                        stop = 1;
                        break;
                    }
                }
                if (!stop && !refine) {
                    stop = 1;
                }
            }
        }

        __syncthreads();
        if (stop) {
            break;
        }
        // Move to the next radix bits.
        shift -= RADIX_BITS;
    }
    // Radix search is finished.
    if (tid == 0) {
        H[row_id] = h;
        kappas[row_id] = C;
        int64_t y = targets[row_id];
        float z_y = logits[row_id*K+y];
        losses[row_id] = 1.0f - z_y + h;
        // printf("row %d: passes = %d\n", row_id, num_passes);
        
    }
}

template<int RADIX_BITS, int THREADS>
__global__
void fw_radix_kernel(const float* logits, float* H, int K, const int64_t* targets, float* losses, int* kappas)
{
    int row_id = blockIdx.x;
    int tid     = threadIdx.x;
    constexpr int NUM_BINS = 1 << RADIX_BITS;
    constexpr int NUM_WARPS = THREADS / WARP_SIZE; 
    constexpr int INITIAL_SHIFT = 32 - RADIX_BITS;

    __shared__ int      warp_counts[NUM_WARPS][NUM_BINS];
    __shared__ float    warp_sums[NUM_WARPS][NUM_BINS];
    __shared__ uint32_t warp_min_keys[NUM_WARPS][NUM_BINS];
    __shared__ uint32_t warp_max_keys[NUM_WARPS][NUM_BINS];

    __shared__ int      bin_counts[NUM_BINS];
    __shared__ float    bin_sums[NUM_BINS];
    __shared__ uint32_t bin_min_keys[NUM_BINS];
    __shared__ uint32_t bin_max_keys[NUM_BINS];

    __shared__ float h;
    __shared__ float threshold;
    __shared__ int   C;
    __shared__ float S;
    __shared__ uint32_t prefix;
    __shared__ uint32_t prefix_mask;

    __shared__ int stop;
    __shared__ int refine;
    if (tid == 0) {
        h = H[row_id];
        threshold = H[row_id];
        C = 0;
        S = 0.0f;
        prefix      = 0u;
        prefix_mask = 0u;
        stop = 0;
        refine = 0;
    }

    __syncthreads();

    int shift = INITIAL_SHIFT;   
    int num_passes = 0;

    while (shift >= 0) {
        num_passes++;
          if (tid == 0) {
            refine = 0;
        }
        __syncthreads();
       
        reset_warp_histograms<NUM_WARPS, NUM_BINS>(warp_counts,warp_sums,warp_min_keys,warp_max_keys,tid);
        __syncthreads();

        accumulate_warp_histograms<RADIX_BITS,NUM_WARPS,NUM_BINS>(logits,row_id,K,K,threshold,shift,warp_counts,warp_sums,warp_min_keys,warp_max_keys,prefix,prefix_mask);
        __syncthreads();

        reduce_warp_histograms<NUM_WARPS,NUM_BINS>(warp_counts,warp_sums,warp_min_keys,warp_max_keys,bin_counts,bin_sums,bin_min_keys,bin_max_keys);
        __syncthreads();

        if (tid == 0) {
            int current_bin = NUM_BINS - 1;
            while (current_bin >= 0 &&
                bin_counts[current_bin] == 0) {
                current_bin--;
            }

            if (current_bin < 0) {
                stop = 1;
            }
            else {
                while (current_bin >= 0) {
                    if (bin_counts[current_bin] == 0) {
                        current_bin--;
                        continue;
                    }
                    int candidate_C = C + bin_counts[current_bin];
                    float candidate_S = S + bin_sums[current_bin];
                    float new_h = (candidate_S - 1.0f) / candidate_C;

                    uint32_t new_h_key = float_to_radix_bits(new_h);

                    uint32_t min_key = bin_min_keys[current_bin];
                    uint32_t max_key = bin_max_keys[current_bin];
                    // 1. Whole bin survives new_h
                    if (new_h_key < min_key) {
                        C = candidate_C;
                        S = candidate_S;
                        h = new_h;
                        threshold = new_h;
                        // Try next lower bin
                        current_bin--;
                    }
                    // 2. new_h cuts through this bin
                    else if (new_h_key < max_key) {

                        threshold = new_h;

                        prefix |= ((uint32_t)current_bin << shift);
                        prefix_mask |= ((uint32_t)(NUM_BINS - 1) << shift);
                        refine = 1;  
                        break;
                    }

                    // 3. new_h >= max of this bin
                    else {
                        stop = 1;
                        break;
                    }
                }
                if (!stop && !refine) {
                    stop = 1;
                }
            }
        }

        __syncthreads();
        if (stop) {
            break;
        }
        // Move to the next radix bits.
        shift -= RADIX_BITS;
    }
    // Radix search is finished.
    if (tid == 0) {
        H[row_id] = h;
        kappas[row_id] = C;
        int64_t y = targets[row_id];
        float z_y = logits[row_id*K+y];
        losses[row_id] = 1.0f - z_y + h;
       // printf("row %d: passes = %d\n", row_id, num_passes);
    }
}

template<int RADIX_BITS, int NUM_WARPS, int NUM_BINS>
__device__ __forceinline__
void accumulate_shared_histograms(const float* shared_logits, int K, float threshold, int shift, int warp_counts[NUM_WARPS][NUM_BINS], float warp_sums[NUM_WARPS][NUM_BINS], uint32_t warp_min_keys[NUM_WARPS][NUM_BINS], uint32_t warp_max_keys[NUM_WARPS][NUM_BINS], uint32_t prefix, uint32_t prefix_mask)
{
    int tid = threadIdx.x;
    int warp = threadIdx.x >> 5;
    for (int i = tid; i < K; i += blockDim.x) {
        float z = shared_logits[i];
        if (z > threshold) {
            uint32_t key = float_to_radix_bits(z);
            if ((key & prefix_mask) == prefix) {
                int bin = get_radix_bin<RADIX_BITS>(key, shift);
                atomicAdd(&warp_counts[warp][bin], 1);
                atomicAdd(&warp_sums[warp][bin], z);
                atomicMin(&warp_min_keys[warp][bin], key);
                atomicMax(&warp_max_keys[warp][bin], key);
            }
        }
    }
}
template<int RADIX_BITS, int THREADS, int MAX_K>
__global__
void fw_radix_shared_kernel(const float* logits, float* H, int K, const int64_t* targets, float* losses, int* kappas, bool* selected_logits)
{
    int row_id = blockIdx.x;
    int tid     = threadIdx.x;
    constexpr int NUM_BINS = 1 << RADIX_BITS;
    constexpr int NUM_WARPS = THREADS / WARP_SIZE; 
    constexpr int INITIAL_SHIFT = 32 - RADIX_BITS;

    __shared__ float shared_logits[MAX_K];

    __shared__ int warp_counts[NUM_WARPS][NUM_BINS];
    __shared__ float warp_sums[NUM_WARPS][NUM_BINS];
    __shared__ uint32_t warp_min_keys[NUM_WARPS][NUM_BINS];
    __shared__ uint32_t warp_max_keys[NUM_WARPS][NUM_BINS];

    __shared__ int bin_counts[NUM_BINS];
    __shared__ float bin_sums[NUM_BINS];
    __shared__ uint32_t bin_min_keys[NUM_BINS];
    __shared__ uint32_t bin_max_keys[NUM_BINS];

    __shared__ float h;
    __shared__ float threshold;
    __shared__ int C;
    __shared__ float S;
    __shared__ uint32_t prefix;
    __shared__ uint32_t prefix_mask;

    __shared__ int stop;
    __shared__ int refine;
    int lane = tid & 31;
    int warp = tid >> 5;

    __shared__ float warp_maxes[NUM_WARPS];

    float local_max = -INFINITY;

    for (int i = tid; i < K; i += blockDim.x) {
        float z = logits[row_id * K + i];
        shared_logits[i] = z;
        local_max = fmaxf(local_max, z);
    }

    local_max = warp_max(local_max);

    if (lane == 0) {
        warp_maxes[warp] = local_max;
    }
    __syncthreads();

    if (warp == 0) {
        float val = (lane < NUM_WARPS) ? warp_maxes[lane] : -INFINITY;
        val = warp_max(val);
        if (lane == 0) {
            h = val - 1.0f;
            threshold = h;
            C = 0;
            S = 0.0f;
            prefix = 0u;
            prefix_mask = 0u;
            stop = 0;
            refine = 0;
        }
    }
    __syncthreads();

    int shift = INITIAL_SHIFT;   
    int num_passes = 0;

    while (shift >= 0) {
        num_passes++;
          if (tid == 0) {
            refine = 0;
        }
        __syncthreads();
       
        reset_warp_histograms<NUM_WARPS, NUM_BINS>(warp_counts,warp_sums,warp_min_keys,warp_max_keys,tid);
        __syncthreads();

        accumulate_shared_histograms<RADIX_BITS, NUM_WARPS, NUM_BINS>(shared_logits, K, threshold, shift, warp_counts, warp_sums, warp_min_keys, warp_max_keys, prefix, prefix_mask );
        __syncthreads();

        reduce_warp_histograms<NUM_WARPS,NUM_BINS>(warp_counts,warp_sums,warp_min_keys,warp_max_keys,bin_counts,bin_sums,bin_min_keys,bin_max_keys);
        __syncthreads();

        if (tid == 0) {
            int current_bin = NUM_BINS - 1;
            while (current_bin >= 0 &&
                bin_counts[current_bin] == 0) {
                current_bin--;
            }

            if (current_bin < 0) {
                stop = 1;
            }
            else {
                while (current_bin >= 0) {
                    if (bin_counts[current_bin] == 0) {
                        current_bin--;
                        continue;
                    }
                    int candidate_C = C + bin_counts[current_bin];
                    float candidate_S = S + bin_sums[current_bin];
                    float new_h = (candidate_S - 1.0f) / candidate_C;

                    uint32_t new_h_key = float_to_radix_bits(new_h);

                    uint32_t min_key = bin_min_keys[current_bin];
                    uint32_t max_key = bin_max_keys[current_bin];
                    // 1. Whole bin survives new_h
                    if (new_h_key < min_key) {
                        C = candidate_C;
                        S = candidate_S;
                        h = new_h;
                        threshold = new_h;
                        // Try next lower bin
                        current_bin--;
                    }
                    // 2. new_h cuts through this bin
                    else if (new_h_key < max_key) {

                        threshold = new_h;

                        prefix |= ((uint32_t)current_bin << shift);
                        prefix_mask |= ((uint32_t)(NUM_BINS - 1) << shift);
                        refine = 1;  
                        break;
                    }

                    // 3. new_h >= max of this bin
                    else {
                        stop = 1;
                        break;
                    }
                }
                if (!stop && !refine) {
                    stop = 1;
                }
            }
        }

        __syncthreads();
        if (stop) {
            break;
        }
        // Move to the next 4 radix bits.
        shift -= RADIX_BITS;
    }
    // Radix search is finished.
    __syncthreads();

    for (int j = tid; j < K; j += blockDim.x) {
        selected_logits[row_id * K + j] =  (shared_logits[j] > h);
    }
    __syncthreads();

    if (tid == 0) {
        H[row_id] = h;
        kappas[row_id] = C;
        int64_t y = targets[row_id];
        float z_y = shared_logits[y];
        losses[row_id] = 1.0f - z_y + h;
    }
}

__global__ void fw_backward_kernel( const float* logits, const float* H, const int* kappas, const int64_t* targets, float* grad, int N, int K) 
{ 
    int row_id = blockIdx.x; 
    int tid = threadIdx.x; 
    int kappa = kappas[row_id]; 
    int64_t y = targets[row_id]; 
    float h = H[row_id]; 
    float inv_kappa = 1.0f / (float)kappa; 
    float inv_N = 1.0f / (float)N; 
    for (int j = tid; j < K; j += blockDim.x)
    { 
        float z = logits[row_id * K + j]; 
        float g = 0.0f;
        if (z > h){ 
            g += inv_kappa; 
        } 
        if (j == y) { 
            g -= 1.0f; 
        } 
        grad[row_id * K + j] = g * inv_N; 
    } 
}

std::vector<torch::Tensor> fw_radix_forward_small(torch::Tensor logits,torch::Tensor targets)
{
    TORCH_CHECK(logits.is_cuda(), "logits must be a CUDA tensor");
    TORCH_CHECK(logits.dim() == 2, "logits must have shape [N, K]");
    TORCH_CHECK(logits.scalar_type() == torch::kFloat32,
                "logits must be float32");

    const int64_t N = logits.size(0);
    const int64_t K = logits.size(1);
    auto selected_logits = torch::empty( {N, K}, logits.options().dtype(torch::kBool) );
    auto H = torch::empty({N}, logits.options());
    auto losses = torch::empty({N}, logits.options());
    auto kappa = torch::empty({N},torch::TensorOptions().dtype(torch::kInt32).device(logits.device()));

    dim3 grid(N);

    if (K <= 100) {
        constexpr int threads = 32;
        fw_radix_shared_kernel<8, threads, 100> <<<grid, threads>>>( logits.data_ptr<float>(), H.data_ptr<float>(), K, targets.data_ptr<int64_t>(), losses.data_ptr<float>(), kappa.data_ptr<int>(), selected_logits.data_ptr<bool>());
    }
    else if(K <= 5000) {
        constexpr int threads = 128;
        fw_radix_shared_kernel<8, threads, 5000> <<<grid, threads>>>( logits.data_ptr<float>(), H.data_ptr<float>(), K, targets.data_ptr<int64_t>(), losses.data_ptr<float>(), kappa.data_ptr<int>(), selected_logits.data_ptr<bool>());
    }
    else{
        constexpr int threads = 128;
        fw_radix_shared_kernel<4, threads, 10000> <<<grid, threads>>>( logits.data_ptr<float>(), H.data_ptr<float>(), K, targets.data_ptr<int64_t>(), losses.data_ptr<float>(), kappa.data_ptr<int>(), selected_logits.data_ptr<bool>());

    }

    auto mean_loss = losses.mean();

    return {H, losses, mean_loss, kappa, selected_logits};
}


std::vector<torch::Tensor> fw_radix_forward(torch::Tensor logits, torch::Tensor targets)
{
    TORCH_CHECK(logits.is_cuda(), "logits must be a CUDA tensor");
    TORCH_CHECK(logits.dim() == 2, "logits must have shape [N, K]");
    TORCH_CHECK(logits.scalar_type() == torch::kFloat32, "logits must be float32");

    const int64_t N = logits.size(0);
    const int64_t K = logits.size(1);

    auto candidate_buffer = torch::empty_like(logits);
    auto candidate_indices = torch::empty( {N, K}, torch::TensorOptions() .dtype(torch::kInt32) .device(logits.device()) );
    // auto candidate_counts = torch::empty( {N}, torch::TensorOptions() .dtype(torch::kInt32) .device(logits.device()) );
    auto candidate_counts = torch::zeros({N}, torch::TensorOptions()
        .dtype(torch::kInt32)
        .device(logits.device()));

    auto H = torch::empty({N}, logits.options()); //one value per sample
    auto losses = torch::empty({N}, logits.options()); //one value per sample
    auto kappa = torch::empty({N}, torch::TensorOptions().dtype(torch::kInt32).device(logits.device())); //one value per sample

    dim3 grid(N);
    if (K < 100000) {
        constexpr int T = 256;
        // H = at::amax(logits, {1}) - 1.0f;
        constexpr int BLOCKS_PER_ROW = 8; 
        constexpr int RADIX_BITS = 8;
        constexpr int NUM_BINS = 1 << RADIX_BITS;

        auto partial_counts = torch::empty( {N, BLOCKS_PER_ROW, NUM_BINS}, torch::TensorOptions() .dtype(torch::kInt32) .device(logits.device()) ); 
        auto partial_sums = torch::empty( {N, BLOCKS_PER_ROW, NUM_BINS}, logits.options() ); 
        auto partial_min_keys = torch::empty( {N, BLOCKS_PER_ROW, NUM_BINS}, torch::TensorOptions() .dtype(torch::kInt32) .device(logits.device()) ); 
        auto partial_max_keys = torch::empty( {N, BLOCKS_PER_ROW, NUM_BINS}, torch::TensorOptions() .dtype(torch::kInt32) .device(logits.device()) );
        auto pre_C = torch::empty( {N}, torch::TensorOptions() .dtype(torch::kInt32) .device(logits.device()) ); 
        auto pre_S = torch::empty( {N}, logits.options() );
        auto pre_prefix = torch::empty( {N}, torch::TensorOptions() .dtype(torch::kInt32) .device(logits.device()) ); 
        auto pre_prefix_mask = torch::empty( {N}, torch::TensorOptions() .dtype(torch::kInt32) .device(logits.device()) );
        auto pre_threshold = torch::empty({N}, logits.options());

        dim3 hist_grid(N * BLOCKS_PER_ROW); 
        fw_precompact_histogram_kernel< RADIX_BITS, T, BLOCKS_PER_ROW ><<<hist_grid, T>>>( logits.data_ptr<float>(), K, partial_counts.data_ptr<int>(), partial_sums.data_ptr<float>(), reinterpret_cast<uint32_t*>( partial_min_keys.data_ptr<int>() ), reinterpret_cast<uint32_t*>( partial_max_keys.data_ptr<int>() ) );

        dim3 reduce_grid(N); 
        fw_precompact_reduce_kernel< RADIX_BITS, T, BLOCKS_PER_ROW ><<<reduce_grid, T>>>( partial_counts.data_ptr<int>(), partial_sums.data_ptr<float>(), reinterpret_cast<uint32_t*>( partial_min_keys.data_ptr<int>() ), reinterpret_cast<uint32_t*>( partial_max_keys.data_ptr<int>() ), H.data_ptr<float>(), pre_C.data_ptr<int>(),pre_S.data_ptr<float>(), reinterpret_cast<uint32_t*>(pre_prefix.data_ptr<int>()), reinterpret_cast<uint32_t*>(pre_prefix_mask.data_ptr<int>()), pre_threshold.data_ptr<float>());

        dim3 compact_grid(N * BLOCKS_PER_ROW); 
        fw_compact_multiblock_kernel< T, BLOCKS_PER_ROW ><<<compact_grid, T>>>( logits.data_ptr<float>(), H.data_ptr<float>(), pre_threshold.data_ptr<float>(),K, candidate_buffer.data_ptr<float>(),candidate_indices.data_ptr<int>(), candidate_counts.data_ptr<int>() );
        fw_radix_buffer_kernel<RADIX_BITS, 256><<<grid, 256>>>(logits.data_ptr<float>(), H.data_ptr<float>(), K, targets.data_ptr<int64_t>(), losses.data_ptr<float>(), kappa.data_ptr<int>(),candidate_buffer.data_ptr<float>(), candidate_counts.data_ptr<int>(), pre_C.data_ptr<int>(),pre_S.data_ptr<float>(), reinterpret_cast<uint32_t*>(pre_prefix.data_ptr<int>()),reinterpret_cast<uint32_t*>(pre_prefix_mask.data_ptr<int>()),  pre_threshold.data_ptr<float>()) ;
    }
    else{        
        constexpr int T = 256;
        // H = at::amax(logits, {1}) - 1.0f;
        constexpr int BLOCKS_PER_ROW = 8; 
        constexpr int RADIX_BITS = 8;
        constexpr int NUM_BINS = 1 << RADIX_BITS;

        auto partial_counts = torch::empty( {N, BLOCKS_PER_ROW, NUM_BINS}, torch::TensorOptions() .dtype(torch::kInt32) .device(logits.device()) ); 
        auto partial_sums = torch::empty( {N, BLOCKS_PER_ROW, NUM_BINS}, logits.options() ); 
        auto partial_min_keys = torch::empty( {N, BLOCKS_PER_ROW, NUM_BINS}, torch::TensorOptions() .dtype(torch::kInt32) .device(logits.device()) ); 
        auto partial_max_keys = torch::empty( {N, BLOCKS_PER_ROW, NUM_BINS}, torch::TensorOptions() .dtype(torch::kInt32) .device(logits.device()) );
        auto pre_C = torch::empty( {N}, torch::TensorOptions() .dtype(torch::kInt32) .device(logits.device()) ); 
        auto pre_S = torch::empty( {N}, logits.options() );
        auto pre_prefix = torch::empty( {N}, torch::TensorOptions() .dtype(torch::kInt32) .device(logits.device()) ); 
        auto pre_prefix_mask = torch::empty( {N}, torch::TensorOptions() .dtype(torch::kInt32) .device(logits.device()) );
        auto pre_threshold = torch::empty({N}, logits.options());

        dim3 hist_grid(N * BLOCKS_PER_ROW); 
        fw_precompact_histogram_kernel< RADIX_BITS, T, BLOCKS_PER_ROW ><<<hist_grid, T>>>( logits.data_ptr<float>(), K, partial_counts.data_ptr<int>(), partial_sums.data_ptr<float>(), reinterpret_cast<uint32_t*>( partial_min_keys.data_ptr<int>() ), reinterpret_cast<uint32_t*>( partial_max_keys.data_ptr<int>() ) );

        dim3 reduce_grid(N); 
        fw_precompact_reduce_kernel< RADIX_BITS, T, BLOCKS_PER_ROW ><<<reduce_grid, T>>>( partial_counts.data_ptr<int>(), partial_sums.data_ptr<float>(), reinterpret_cast<uint32_t*>( partial_min_keys.data_ptr<int>() ), reinterpret_cast<uint32_t*>( partial_max_keys.data_ptr<int>() ), H.data_ptr<float>(), pre_C.data_ptr<int>(),pre_S.data_ptr<float>(), reinterpret_cast<uint32_t*>(pre_prefix.data_ptr<int>()), reinterpret_cast<uint32_t*>(pre_prefix_mask.data_ptr<int>()), pre_threshold.data_ptr<float>());
        
        dim3 compact_grid(N * BLOCKS_PER_ROW); 
        fw_compact_multiblock_kernel< T, BLOCKS_PER_ROW ><<<compact_grid, T>>>( logits.data_ptr<float>(), H.data_ptr<float>(), pre_threshold.data_ptr<float>(), K, candidate_buffer.data_ptr<float>(),candidate_indices.data_ptr<int>(), candidate_counts.data_ptr<int>() );
        fw_radix_buffer_kernel<RADIX_BITS, 256><<<grid, 256>>>(logits.data_ptr<float>(), H.data_ptr<float>(), K, targets.data_ptr<int64_t>(), losses.data_ptr<float>(), kappa.data_ptr<int>(),candidate_buffer.data_ptr<float>(), candidate_counts.data_ptr<int>(), pre_C.data_ptr<int>(),pre_S.data_ptr<float>(), reinterpret_cast<uint32_t*>(pre_prefix.data_ptr<int>()),reinterpret_cast<uint32_t*>(pre_prefix_mask.data_ptr<int>()),  pre_threshold.data_ptr<float>()) ;
    }

    auto mean_loss = losses.mean();

    return {H, losses, mean_loss, kappa, candidate_buffer, candidate_indices, candidate_counts};
}

__global__ void fw_backward_candidate_kernel(
    const float* candidate_buffer,
    const int* candidate_indices,
    const int* candidate_counts,
    const float* H,
    const int* kappas,
    const int64_t* targets,
    float* grad,
    int N,
    int K)
{
    int row = blockIdx.x;
    int tid = threadIdx.x;

    float h = H[row];
    int kappa = kappas[row];
    int count = candidate_counts[row];

    float selected_grad =
        (1.0f / (float)kappa) * (1.0f / (float)N);

    // Only scan compact candidates
    for (int p = tid; p < count; p += blockDim.x) {
        float z = candidate_buffer[row * K + p];
        if (z > h) {
            int j = candidate_indices[row * K + p];
            grad[row * K + j] = selected_grad;
        }
    }

    __syncthreads();

    if (tid == 0) {
        int64_t y = targets[row];
        grad[row * K + y] -= 1.0f / (float)N;
    }
}

torch::Tensor fw_radix_backward(torch::Tensor logits, torch::Tensor targets, torch::Tensor H, torch::Tensor kappas, torch::Tensor candidate_buffer, torch::Tensor candidate_indices, torch::Tensor candidate_counts)
{
    int N = logits.size(0);
    int K = logits.size(1);

    auto grad = torch::zeros_like(logits);

    constexpr int THREADS = 256;
    dim3 grid(N);

    fw_backward_candidate_kernel<<<grid, THREADS>>>(
        candidate_buffer.data_ptr<float>(),
        candidate_indices.data_ptr<int>(),
        candidate_counts.data_ptr<int>(),
        H.data_ptr<float>(),
        kappas.data_ptr<int>(),
        targets.data_ptr<int64_t>(),
        grad.data_ptr<float>(),
        N,
        K
    );

    return grad;
}

__global__ void fw_backward_mask_kernel(const int64_t* __restrict__ targets, const bool* __restrict__ selected_logits, const int* __restrict__ kappas, float* __restrict__ grad, int N, int K)
{
    int row = blockIdx.x;
    int tid = threadIdx.x;

    int kappa = kappas[row];
    int64_t y = targets[row];

    float inv_N = 1.0f / static_cast<float>(N);
    float selected_grad = inv_N / static_cast<float>(kappa);

    size_t row_offset =
        static_cast<size_t>(row) * K;

    for (int j = tid; j < K; j += blockDim.x) {

        size_t idx = row_offset + j;

        float g = selected_logits[idx]
                    ? selected_grad
                    : 0.0f;

        if (j == y) {
            g -= inv_N;;
        }

        grad[idx] = g;
    }
}

torch::Tensor fw_radix_backward_small( torch::Tensor targets,torch::Tensor selected_logits, torch::Tensor kappas )
{
    int N = selected_logits.size(0);
    int K = selected_logits.size(1);
    auto grad = torch::empty( {N, K}, torch::TensorOptions() .dtype(torch::kFloat32) .device(selected_logits.device()) );
    int THREADS;

    if (K <= 2000)
        THREADS = 64;
    else if(K <= 5000)
        THREADS = 128;
    else
        THREADS = 256;

    dim3 grid(N);
    dim3 block(THREADS);
    fw_backward_mask_kernel<<<grid, block>>>( targets.data_ptr<int64_t>(), selected_logits.data_ptr<bool>(), kappas.data_ptr<int>(), grad.data_ptr<float>(), N, K );
    return grad;
}