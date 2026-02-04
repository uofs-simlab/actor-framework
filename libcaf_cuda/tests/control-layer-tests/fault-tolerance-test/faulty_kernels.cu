#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <iostream>
#include <cstdlib>

// Step 1: Initialize denominators with ~50% zeros using cuRAND
extern "C" __global__ void init_denominators(float* denominators, int n, unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    curandState state;
    curand_init(seed, idx, 0, &state);
    float rand_val = curand_uniform(&state);
    denominators[idx] = (rand_val < 0.5f) ? 0.0f : 1.0f;  // ~50% chance of zero
}

// Step 2: Perform division (potential div by zero -> Inf)
extern "C" __global__ void perform_division(float* numerators, float* denominators, float* results, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    results[idx] = numerators[idx] / denominators[idx];  // Triggers Inf if denominator == 0
}

// Step 3: Simple reduction to sum results (propagates Inf if present)
extern "C" __global__ void sum_results(float* results, float* final_sum, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        // atomic add each element directly to the final sum
        atomicAdd(final_sum, results[idx]);
    }
}
