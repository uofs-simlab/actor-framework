#pragma once

#include <vector>
#include <string>
#include <cstdint>
#include <memory>
#include <random>
#include <chrono>

enum SolverType { CGS_SOLVER, BICSTAB_SOLVER };

// Structure to hold raw data from the binary file
struct SparseMatrixCOO {
    int32_t rows;
    int32_t cols;
    int32_t nnz;
    std::vector<int32_t> row_indices;
    std::vector<int32_t> col_indices;
    std::vector<float> values;
};

// Structure optimized for high-performance solvers
struct SparseMatrixCSR {
    int32_t rows;
    int32_t cols;
    int32_t nnz;
    std::vector<int32_t> row_ptr;    // Size: rows + 1
    std::vector<int32_t> col_indices;// Size: nnz
    std::vector<float> values;       // Size: nnz
};

struct MatrixData {
    std::vector<int32_t> row_ptr;
    std::vector<int32_t> col_indices;
    std::vector<float> values;
    std::vector<float> b;
    std::vector<float> x_guess;
    int32_t rows;
    int32_t cols;
    int32_t nnz;
};

struct MatrixTask {
    std::string path;
    SolverType type;
    std::shared_ptr<MatrixData> data;
};



struct Partition {
    size_t begin;
    size_t end;
    std::vector<int> devices;
    std::vector<int> streams;
};

// Function to slurp the binary data into memory
SparseMatrixCOO load_binary_coo(const std::string& filepath);

// Converts COO to CSR format for solver compatibility
SparseMatrixCSR convert_coo_to_csr(const SparseMatrixCOO& coo);

// Compute b = A * x using CSR layout (Sparse Matrix-Vector Multiplication)
std::vector<float> compute_rhs_spmv(const SparseMatrixCSR& A, const std::vector<float>& x);

std::vector<MatrixTask> scan_for_matrices(const std::string& dir, SolverType type);
int generate_random_sleep_ms(int min_ms, int max_ms);

// Workload generation helpers
std::chrono::milliseconds generate_random_interval(
    std::mt19937& rng,
    double mean_ms);

std::vector<MatrixTask> generate_batch(
    const std::vector<MatrixTask>& matrix_pool,
    std::mt19937& rng,
    size_t batch_size);