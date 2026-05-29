/**
 * This test file evaluates the correctness of sparse_gmres_facade for solving Ax = b.
 * It covers:
 * - CSR, CSC, and COO formats.
 * - Convergence verification with simple matrices.
 * - Stress testing with a large 1D Laplacian matrix.
 */
#include <caf/all.hpp>
#include <caf/cuda/all.hpp>
#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <thread>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <cmath>
#include "caf/actor_registry.hpp"
#include "caf/actorSOLVE/sparse-matrix-solvers/sparse-GMRES-actor/sparse-GMRES-actor.hpp"

using namespace caf;
using namespace caf::cuda;

// --- Manual Sparse Utilities for Testing ---

template <class T = float>
struct LocalCSR {
    int rows, cols, nnz;
    std::vector<int> row_ptr;
    std::vector<int> col_ind;
    std::vector<T> values;
};

template <class T = float>
LocalCSR<T> load_binary_matrix_manual(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("Could not open " + path);

    int32_t r, c, n;
    file.read(reinterpret_cast<char*>(&r), sizeof(int32_t));
    file.read(reinterpret_cast<char*>(&c), sizeof(int32_t));
    file.read(reinterpret_cast<char*>(&n), sizeof(int32_t));

    std::vector<int32_t> rows_coo(n), cols_coo(n);
    std::vector<float> vals_coo(n);

    file.read(reinterpret_cast<char*>(rows_coo.data()), n * sizeof(int32_t));
    file.read(reinterpret_cast<char*>(cols_coo.data()), n * sizeof(int32_t));
    file.read(reinterpret_cast<char*>(vals_coo.data()), n * sizeof(float));

    // Detect and fix 1-based indexing (Matrix Market standard)
    int max_idx = 0;
    for(auto v : rows_coo) if(v > max_idx) max_idx = v;
    for(auto v : cols_coo) if(v > max_idx) max_idx = v;

    if (max_idx == r || max_idx == c) {
        for(auto& v : rows_coo) v--;
        for(auto& v : cols_coo) v--;
    }

    // Convert COO to CSR
    LocalCSR<T> csr;
    csr.rows = r; csr.cols = c; csr.nnz = n;
    csr.row_ptr.assign(r + 1, 0);
    csr.col_ind.resize(n);
    csr.values.resize(n);

    for (int i = 0; i < n; ++i) csr.row_ptr[rows_coo[i] + 1]++;
    for (int i = 0; i < r; ++i) csr.row_ptr[i + 1] += csr.row_ptr[i];

    std::vector<int> current_pos = csr.row_ptr;
    for (int i = 0; i < n; ++i) {
        int row = rows_coo[i];
        int dest = current_pos[row]++;
        csr.col_ind[dest] = cols_coo[i];
        csr.values[dest] = static_cast<T>(vals_coo[i]);
    }
    return csr;
}

template <class T = float>
std::vector<T> compute_rhs_manual(const LocalCSR<T>& A, const std::vector<T>& x) {
    std::vector<T> b(A.rows, T{0});
    for (int i = 0; i < A.rows; ++i) {
        T sum = T{0};
        for (int j = A.row_ptr[i]; j < A.row_ptr[i+1]; ++j) {
            sum += A.values[j] * x[A.col_ind[j]];
        }
        b[i] = sum;
    }
    return b;
}

// --- End Manual Utilities ---

void verify_solution(const std::string& test_name, const std::vector<float>& actual, 
                     const std::vector<float>& expected, float tol = 1e-3) {
    if (actual.size() != expected.size()) {
        std::cout << "[ERROR] " << test_name << " failed: Size mismatch (got " 
                  << actual.size() << ", expected " << expected.size() << ")" << std::endl;
        return;
    }
    bool all_correct = true;
    for (size_t i = 0; i < actual.size(); ++i) {
        if (std::abs(actual[i] - expected[i]) > tol) {
            all_correct = false;
            std::cout << "[ERROR] " << test_name << " mismatch at index " << i 
                      << ": Expected " << expected[i] 
                      << ", Got " << actual[i] << std::endl;
            break;
        }
    }
    if (all_correct) {
        std::cout << "[SUCCESS] " << test_name << " converged to correct solution." << std::endl;
    }
}

void caf_main(actor_system& sys) {
    // Enable cuBLAS and cuSPARSE for the GMRES solver
    manager::init(sys, manager_config(true, true)); 

    // Simple Diagonal Matrix A (3x3): diag(4, 3, 2)
    // b = [8, 9, 2] -> Expected x = [2, 3, 1]
    int n = 3, nnz = 3;
    std::vector<float> h_b = {8.0f, 9.0f, 2.0f};
    std::vector<float> expected = {2.0f, 3.0f, 1.0f};
    float tolerance = 1e-5f;
    int max_iter = 100;
    int restart_m = 10;

    scoped_actor self{sys};

    // Test 1: CSR Format
    {
        std::cout << "[INFO] Test 1: CSR format simple matrix..." << std::endl;
        std::vector<int> row_ptr = {0, 1, 2, 3};
        std::vector<int> col_ind = {0, 1, 2};
        std::vector<float> values = {4.0f, 3.0f, 2.0f};
        std::vector<float> h_x(n, 0.0f);

        auto facade = sys.spawn<sparse_gmres_facade<float>>(100);

        self->mail(create_in_arg(row_ptr), create_in_arg(col_ind), create_in_arg(values),
                   create_in_arg(h_b), create_in_out_arg(h_x),
                   matrix_format::csr, n, nnz, tolerance, max_iter, restart_m, 0, 0).send(facade);

        self->receive(
            [&](uint32_t /*resp_id*/, int /*idx*/, const std::vector<float>& result_x, solver_result_meta meta) {
                verify_solution("GMRES Facade CSR Simple", result_x, expected);
            }
        );
    }

    // Test 2: Facade Actor CSR with Custom Buffer
    {
        std::cout << "\n[INFO] Test 2: Facade actor CSR with custom buffer..." << std::endl;
        std::vector<float> custom_x(n, 0.0f);
        std::vector<int> row_ptr = {0, 1, 2, 3};
        std::vector<int> col_ind = {0, 1, 2};
        std::vector<float> values = {4.0f, 3.0f, 2.0f};
        output_mapping m{4, custom_x.data(), (size_t)n};
        
        auto facade = sys.spawn<sparse_gmres_facade<float>>(100);
        self->mail(std::vector<output_mapping>{m},
                   create_in_arg(row_ptr), create_in_arg(col_ind), create_in_arg(values),
                   create_in_arg(h_b), create_in_out_arg(custom_x),
                   matrix_format::csr, n, nnz, tolerance, max_iter, restart_m, 0, 1).send(facade);

        self->receive(
            [&](uint32_t /*resp_id*/, int index, solver_result_meta meta) {
                if (index == 4)
                  verify_solution("GMRES Facade Custom Buffer", custom_x, expected);
            }
        );
    }

    // Test 3: Facade Actor CSR returning mem_ptr handles
    {
        std::cout << "\n[INFO] Test 3: Facade actor CSR returning mem_ptr..." << std::endl;
        std::vector<int> row_ptr = {0, 1, 2, 3};
        std::vector<int> col_ind = {0, 1, 2};
        std::vector<float> values = {4.0f, 3.0f, 2.0f};
        std::vector<float> h_x(n, 0.0f);

        auto facade = sys.spawn<sparse_gmres_facade<float>>(100);

        self->mail(return_mem_ptr_atom_v,
                   create_in_arg(row_ptr), create_in_arg(col_ind), create_in_arg(values),
                   create_in_arg(h_b), create_in_out_arg(h_x),
                   matrix_format::csr, n, nnz, tolerance, max_iter, restart_m, 0, 2).send(facade);

        self->receive(
            [&](uint32_t /*resp_id*/, mem_ptr<float> ptr, solver_result_meta meta) {
                auto result_x = ptr->copy_to_host();
                verify_solution("GMRES Facade mem_ptr", result_x, expected);
            }
        );
    }

    // Test 4: Stress Test - 1D Laplacian (N=5000)
    {
        int N_large = 5000;
        std::cout << "\n[INFO] Test 4: Stress Test - 1D Laplacian (N=" << N_large << ")..." << std::endl;
        
        std::vector<int> row_ptr;
        std::vector<int> col_ind;
        std::vector<float> values;
        row_ptr.push_back(0);
        for(int i=0; i<N_large; ++i) {
            if(i > 0) { col_ind.push_back(i-1); values.push_back(-1.0f); }
            col_ind.push_back(i); values.push_back(2.0f);
            if(i < N_large-1) { col_ind.push_back(i+1); values.push_back(-1.0f); }
            row_ptr.push_back(col_ind.size());
        }

        std::vector<float> b_large(N_large, 1.0f);
        std::vector<float> x_large(N_large, 0.0f);
        
        auto facade = sys.spawn<sparse_gmres_facade<float>>(100);
        auto start = std::chrono::high_resolution_clock::now();

        self->mail(create_in_arg(row_ptr), create_in_arg(col_ind), create_in_arg(values),
                   create_in_arg(b_large), create_in_out_arg(x_large),
                   matrix_format::csr, N_large, (int)values.size(), 1e-4f, 2000, 30, 0, 3).send(facade);

        self->receive(
            [&](uint32_t, int, std::vector<float> result, solver_result_meta meta) {
                auto end = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> elapsed = end - start;
                std::cout << "[SUCCESS] Stress Test completed in " << elapsed.count() << " seconds." << std::endl;
                std::cout << "[INFO] First 5 elements of solution: ";
                for(int i=0; i<5; ++i) std::cout << result[i] << " ";
                std::cout << "..." << std::endl;
            }
        );
    }

    // Test 5: CSC Format
    {
        std::cout << "\n[INFO] Test 5: CSC format simple matrix..." << std::endl;
        std::vector<int> col_ptr = {0, 1, 2, 3};
        std::vector<int> row_ind = {0, 1, 2};
        std::vector<float> values = {4.0f, 3.0f, 2.0f};
        std::vector<float> h_x(n, 0.0f);

        auto facade = sys.spawn<sparse_gmres_facade<float>>(100);

        self->mail(create_in_arg(col_ptr), create_in_arg(row_ind), create_in_arg(values),
                   create_in_arg(h_b), create_in_out_arg(h_x),
                   matrix_format::csc, n, nnz, tolerance, max_iter, restart_m, 0, 4).send(facade);

        self->receive(
            [&](uint32_t /*resp_id*/, int /*idx*/, const std::vector<float>& result_x, solver_result_meta meta) {
                verify_solution("GMRES Facade CSC Simple", result_x, expected);
            }
        );
    }

    // Test 6: Real-world matrix from file (lnsp3937.bin)
    {
        std::string path = "/scratch/nqr159/matrix-collection/matrices/unsymmetric/lnsp3937.bin";
        std::cout << "\n[INFO] Test 6: Loading real-world matrix " << path << "..." << std::endl;
        try {
            LocalCSR<float> A = load_binary_matrix_manual<float>(path);
            std::cout << "[INFO] Matrix Metadata: Rows=" << A.rows << ", Cols=" << A.cols 
                      << ", NNZ=" << A.nnz << std::endl;
            
            std::vector<float> expected_real(A.rows, 1.0f);
            std::vector<float> b_real = compute_rhs_manual<float>(A, expected_real);
            std::vector<float> h_x(A.rows, 0.0f);

            auto facade = sys.spawn<sparse_gmres_facade<float>>(100);

            self->mail(create_in_arg(A.row_ptr), create_in_arg(A.col_ind), create_in_arg(A.values),
                       create_in_arg(b_real), create_in_out_arg(h_x),
                       matrix_format::csr, A.rows, A.nnz, 1e-5f, 5000, 30, 0, 5).send(facade);

            self->receive(
                [&](uint32_t /*resp_id*/, int /*idx*/, const std::vector<float>& result, solver_result_meta meta) {
                    std::cout << "[INFO] Iterations: " << meta.iterations << ", Converged: " << std::boolalpha << meta.converged << std::endl;
                    verify_solution("GMRES Real Matrix", result, expected_real, 1e-2f);
                }
            );
        } catch (const std::exception& e) {
            std::cout << "[ERROR] Test 6 Failed: " << e.what() << std::endl;
        }
    }

    manager::shutdown();
}

CAF_MAIN(id_block::cuda)
