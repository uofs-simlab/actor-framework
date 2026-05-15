extern "C" __global__ void conv1d(const int* A, const int* K, int* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int W = 5; // Fixed filter width for benchmark simplicity
    if (idx < N) {
        int sum = 0;
        int halfW = W / 2;
        for (int i = 0; i < W; ++i) {
            int col = idx + i - halfW;
            if (col >= 0 && col < N) {
                sum += A[col] * K[i];
            }
        }
        C[idx] = sum;
    }
}
