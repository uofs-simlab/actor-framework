extern "C" {

__global__ void update_p_float(int n, const float* r, float* p, const float* rho, const float* old_rho, int iteration) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        if (iteration == 0) {
            p[i] = r[i];
        } else {
            float beta = rho[0] / old_rho[0];
            p[i] = r[i] + beta * p[i];
        }
    }
}

__global__ void update_x_r_float(int n, float* x, float* r, const float* p, const float* w, const float* rho, const float* dot_pw) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float alpha = rho[0] / dot_pw[0];
        x[i] += alpha * p[i];
        r[i] -= alpha * w[i];
    }
}

__global__ void update_p_double(int n, const double* r, double* p, const double* rho, const double* old_rho, int iteration) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        if (iteration == 0) {
            p[i] = r[i];
        } else {
            double beta = rho[0] / old_rho[0];
            p[i] = r[i] + beta * p[i];
        }
    }
}

__global__ void update_x_r_double(int n, double* x, double* r, const double* p, const double* w, const double* rho, const double* dot_pw) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        double alpha = rho[0] / dot_pw[0];
        x[i] += alpha * p[i];
        r[i] -= alpha * w[i];
    }
}

}