extern "C" __global__
void extract_diag_inv(int n, const int* row_ptr, const int* col_ind, const float* val, float* d_inv) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float d = 0.0f;
        // Search for the diagonal element (A[i][i]) in the sparse row
        for (int j = row_ptr[i]; j < row_ptr[i+1]; j++) {
            if (col_ind[j] == i) {
                d = val[j];
                break;
            }
        }
        d_inv[i] = (d != 0.0f) ? 1.0f / d : 1.0f;
    }
}