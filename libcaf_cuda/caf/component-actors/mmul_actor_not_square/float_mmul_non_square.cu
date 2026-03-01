extern "C" __global__
void mmul_non_square(const float* A, const float* B, float* C,
                     int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // row in C
    int col = blockIdx.x * blockDim.x + threadIdx.x; // col in C

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}
