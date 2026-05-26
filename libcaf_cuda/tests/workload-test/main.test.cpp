#include <caf/all.hpp>
#include <caf/cuda/all.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <numeric>
#include <cmath>
#include <algorithm>
#include "caf/actorSOLVE/sparse-matrix-solvers/sparse-CGS-actor/sparse-CGS-actor.hpp"

using namespace caf;
using namespace caf::cuda;

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
SparseMatrixCOO load_binary_coo(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open matrix file: " + filepath);
    }

    SparseMatrixCOO coo;
    
    // 1. Read the 12-byte header
    file.read(reinterpret_cast<char*>(&coo.rows), sizeof(int32_t));
    file.read(reinterpret_cast<char*>(&coo.cols), sizeof(int32_t));
    file.read(reinterpret_cast<char*>(&coo.nnz), sizeof(int32_t));

    // Allocate memory vectors
    coo.row_indices.resize(coo.nnz);
    coo.col_indices.resize(coo.nnz);
    coo.values.resize(coo.nnz);

    // 2. Stream the blocks continuously
    file.read(reinterpret_cast<char*>(coo.row_indices.data()), coo.nnz * sizeof(int32_t));
    file.read(reinterpret_cast<char*>(coo.col_indices.data()), coo.nnz * sizeof(int32_t));
    file.read(reinterpret_cast<char*>(coo.values.data()), coo.nnz * sizeof(float));

    return coo;
}

// Converts COO to CSR format for solver compatibility
SparseMatrixCSR convert_coo_to_csr(const SparseMatrixCOO& coo) {
    SparseMatrixCSR csr;
    csr.rows = coo.rows;
    csr.cols = coo.cols;
    csr.nnz = coo.nnz;
    
    csr.row_ptr.assign(csr.rows + 1, 0);
    csr.col_indices.resize(csr.nnz);
    csr.values.resize(csr.nnz);

    // Step 1: Count elements per row
    for (int32_t i = 0; i < coo.nnz; ++i) {
        csr.row_ptr[coo.row_indices[i] + 1]++;
    }

    // Step 2: Cumulative sum to build row pointers
    for (int32_t i = 0; i < csr.rows; ++i) {
        csr.row_ptr[i + 1] += csr.row_ptr[i];
    }

    // Step 3: Copy tracking array to insert elements in order
    std::vector<int32_t> current_row_pos = csr.row_ptr;

    // Step 4: Fill column and value arrays
    for (int32_t i = 0; i < coo.nnz; ++i) {
        int32_t row = coo.row_indices[i];
        int32_t dest_pos = current_row_pos[row]++;
        csr.col_indices[dest_pos] = coo.col_indices[i];
        csr.values[dest_pos] = coo.values[i];
    }

    return csr;
}

// Compute b = A * x using CSR layout (Sparse Matrix-Vector Multiplication)
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

void caf_main(actor_system& sys) {
    auto& args = sys.config().args_remainder;
    if (args.empty()) {
        std::cout << "Usage: workload-test <path_to_matrix_bin>\n";
        std::cout << "Example: workload-test /scratch/nqr159/matrix-collection/matrices/spd/bcsstk08.bin\n";
        return;
    }

    // Initialize GPU Manager with cuBLAS and cuSPARSE enabled
    manager::init(sys, manager_config(true, true));
    std::string filepath = args[0];

    try {
        std::cout << "Loading binary file: " << filepath << " ...\n";
        SparseMatrixCOO coo_matrix = load_binary_coo(filepath);
        
        std::cout << "-> Matrix Loaded. Components: " 
                  << "Rows=" << coo_matrix.rows 
                  << ", Cols=" << coo_matrix.cols 
                  << ", NNZ=" << coo_matrix.nnz << "\n";

        std::cout << "Converting to Compressed Sparse Row (CSR) format...\n";
        SparseMatrixCSR A = convert_coo_to_csr(coo_matrix);

        // --- PREPARE BENCHMARK VECTORS ---
        std::cout << "Generating test vectors (b = A * x_true)...\n";
        
        // 1. Create a known ideal solution vector x_true (filled with 1.0f)
        std::vector<float> x_true(A.cols, 1.0f);
        
        // 2. Compute true right-hand side b
        std::vector<float> b = compute_rhs_spmv(A, x_true);
        
        // 3. Allocate an initial guess vector x filled with zeros
        std::vector<float> x_guess(A.cols, 0.0f);

        std::cout << "\n=========================================\n";
        std::cout << " Ready for Solver Execution!\n";
        std::cout << "=========================================\n";
        std::cout << "Arrays allocated and verified:\n";
        std::cout << " - A.values size:      " << A.values.size() << " elements\n";
        std::cout << " - A.row_ptr size:     " << A.row_ptr.size() << " elements\n";
        std::cout << " - Vector b size:      " << b.size() << " elements\n";
        std::cout << " - Vector x_guess size:" << x_guess.size() << " elements\n\n";
        
        scoped_actor self{sys};
        float tolerance = 1e-5f;
        int max_iter = 2000;

        // Spawn the Sparse CG Actor
        auto solver = sys.spawn<sparse_cg_actor>(
            create_in_arg(A.row_ptr), create_in_arg(A.col_indices), create_in_arg(A.values),
            create_in_arg(b), create_in_out_arg(x_guess),
            matrix_format::csr, A.rows, A.nnz, tolerance, max_iter, 0, 0, actor_cast<actor>(self));

        std::cout << "[INFO] Starting Solver Actor...\n";
        self->mail(start_atom_v).send(solver);

        self->receive(
            [&](std::vector<float> result_x) {
                std::cout << "\n=========================================\n";
                std::cout << " Solver Finished!\n";
                std::cout << "=========================================\n";
                std::cout << "Verification (Expected values near 1.0):\n";
                std::cout << "First 5 elements: ";
                for (int i = 0; i < std::min(5, (int)result_x.size()); ++i) {
                    std::cout << result_x[i] << " ";
                }
                std::cout << "\n";
            }
        );

    } catch (const std::exception& e) {
        std::cerr << "CRITICAL EXCEPTION: " << e.what() << "\n";
    }
    manager::shutdown();
}
CAF_MAIN(id_block::cuda)
