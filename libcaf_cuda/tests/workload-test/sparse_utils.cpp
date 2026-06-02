#include "sparse_utils.hpp"
#include <fstream>
#include <stdexcept>
#include <filesystem>
#include <random>
namespace fs = std::filesystem;

SparseMatrixCOO load_binary_coo(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open matrix file: " + filepath);
    }

    SparseMatrixCOO coo;
    
    file.read(reinterpret_cast<char*>(&coo.rows), sizeof(int32_t));
    file.read(reinterpret_cast<char*>(&coo.cols), sizeof(int32_t));
    file.read(reinterpret_cast<char*>(&coo.nnz), sizeof(int32_t));

    coo.row_indices.resize(coo.nnz);
    coo.col_indices.resize(coo.nnz);
    coo.values.resize(coo.nnz);

    file.read(reinterpret_cast<char*>(coo.row_indices.data()), coo.nnz * sizeof(int32_t));
    file.read(reinterpret_cast<char*>(coo.col_indices.data()), coo.nnz * sizeof(int32_t));
    file.read(reinterpret_cast<char*>(coo.values.data()), coo.nnz * sizeof(float));

    return coo;
}

SparseMatrixCSR convert_coo_to_csr(const SparseMatrixCOO& coo) {
    SparseMatrixCSR csr;
    csr.rows = coo.rows;
    csr.cols = coo.cols;
    csr.nnz = coo.nnz;
    
    csr.row_ptr.assign(csr.rows + 1, 0);
    csr.col_indices.resize(csr.nnz);
    csr.values.resize(csr.nnz);

    for (int32_t i = 0; i < coo.nnz; ++i) {
        csr.row_ptr[coo.row_indices[i] + 1]++;
    }

    for (int32_t i = 0; i < csr.rows; ++i) {
        csr.row_ptr[i + 1] += csr.row_ptr[i];
    }

    std::vector<int32_t> current_row_pos = csr.row_ptr;

    for (int32_t i = 0; i < coo.nnz; ++i) {
        int32_t row = coo.row_indices[i];
        int32_t dest_pos = current_row_pos[row]++;
        csr.col_indices[dest_pos] = coo.col_indices[i];
        csr.values[dest_pos] = coo.values[i];
    }

    return csr;
}

std::vector<float> compute_rhs_spmv(const SparseMatrixCSR& A, const std::vector<float>& x) {
    std::vector<float> b(A.rows, 0.0f);
    
    for (int32_t i = 0; i < A.rows; ++i) {
        float sum = 0.0f;
        int32_t row_start = A.row_ptr[i];
        int32_t row_end = A.row_ptr[i + 1];
        
        for (int32_t j = row_start; j < row_end; ++j) {
            sum += A.values[j] * x[A.col_indices[j]];
        }
        b[i] = sum;
    }
    return b;
}

std::vector<MatrixTask> scan_for_matrices(const std::string& dir, SolverType type) {
    std::vector<MatrixTask> tasks;
    if (!fs::exists(dir)) return tasks;
    for (const auto& entry : fs::directory_iterator(dir)) {
        if (entry.path().extension() == ".bin") {
            auto coo = load_binary_coo(entry.path().string());
            auto csr = convert_coo_to_csr(coo);
            auto data = std::make_shared<MatrixData>();
            data->rows = csr.rows;
            data->cols = csr.cols;
            data->nnz = csr.nnz;
            data->b = compute_rhs_spmv(csr, std::vector<float>(csr.cols, 1.0f));
            data->row_ptr = std::move(csr.row_ptr);
            data->col_indices = std::move(csr.col_indices);
            data->values = std::move(csr.values);
            data->x_guess.assign(data->cols, 0.0f);
            
            tasks.push_back({entry.path().string(), type, data});
        }
    }
    return tasks;
}

int generate_random_sleep_ms(int min_ms, int max_ms) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(min_ms, max_ms);
    return dis(gen);
}