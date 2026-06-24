extern "C" {

__global__ void update_p_float(int n, const float* r, float* p, const float* rho, const float* old_rho, int iteration, float threshold) {
    if (rho[0] <= threshold) return;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        if (iteration == 0) {
            p[i] = r[i];
        } else {
            float denom = old_rho[0];
            float beta = (denom > 1e-30f) ? (rho[0] / denom) : 0.0f;
            p[i] = r[i] + beta * p[i];
        }
    }
}

__global__ void update_x_r_float(int n, float* x, float* r, const float* p, const float* w, const float* rho, const float* dot_pw, float threshold) {
    if (rho[0] <= threshold) return;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float denom = dot_pw[0];
        float alpha = (abs(denom) > 1e-30f) ? (rho[0] / denom) : 0.0f;
        x[i] += alpha * p[i];
        r[i] -= alpha * w[i];
    }
}

__global__ void update_p_double(int n, const double* r, double* p, const double* rho, const double* old_rho, int iteration, double threshold) {
    if (rho[0] <= threshold) return;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        if (iteration == 0) {
            p[i] = r[i];
        } else {
            double denom = old_rho[0];
            double beta = (denom > 1e-35) ? (rho[0] / denom) : 0.0;
            p[i] = r[i] + beta * p[i];
        }
    }
}

__global__ void update_x_r_double(int n, double* x, double* r, const double* p, const double* w, const double* rho, const double* dot_pw, double threshold) {
    if (rho[0] <= threshold) return;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        double denom = dot_pw[0];
        double alpha = (abs(denom) > 1e-35) ? (rho[0] / denom) : 0.0;
        x[i] += alpha * p[i];
        r[i] -= alpha * w[i];
    }
}

}