extern "C" __global__
void check_stability(int n, const float* x, const float* r, int* err) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // If any value in the solution or residual is invalid, set the error flag
        if (isnan(x[idx]) || isinf(x[idx]) || isnan(r[idx]) || isinf(r[idx])) {
            atomicExch(err, 1);
        }
    }
}