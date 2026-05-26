#pragma once

#include <vector>
#include <string>
#include <cstdint>

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

// Function to slurp the binary data into memory
SparseMatrixCOO load_binary_coo(const std::string& filepath);

// Converts COO to CSR format for solver compatibility
SparseMatrixCSR convert_coo_to_csr(const SparseMatrixCOO& coo);

// Compute b = A * x using CSR layout (Sparse Matrix-Vector Multiplication)
std::vector<float> compute_rhs_spmv(const SparseMatrixCSR& A, const std::vector<float>& x);