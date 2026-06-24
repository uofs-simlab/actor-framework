#include <cuda.h>
#include <curand_kernel.h>

extern "C" __global__
void monteCarloKernel(int seed, int total_samples, int* out_count) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int n_threads = gridDim.x * blockDim.x;

    curandState_t rng;
    curand_init(seed, tid, 0, &rng);

    long long local_count = 0;
    for (int i = tid; i < total_samples; i += n_threads) {
        float x = curand_uniform(&rng);
        float y = curand_uniform(&rng);
        if (x * x + y * y <= 1.0f)
            local_count++;
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        local_count += __shfl_down_sync(0xffffffff, local_count, offset);

    if ((threadIdx.x & 31) == 0)
        atomicAdd(out_count, static_cast<int>(local_count));
}