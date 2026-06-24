/**
 * This test file evaluates the correctness of sparse_bicgstab_actor for solving Ax = b.
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
#include "caf/actorSOLVE/sparse-matrix-solvers/sparse-BiCGSTAB-actor/sparse-BiCGSTAB-actor.hpp"

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
    // If the maximum index found equals the dimension 'r', it is 1-based.
    int max_idx = 0;
    for(auto v : rows_coo) if(v > max_idx) max_idx = v;
    for(auto v : cols_coo) if(v > max_idx) max_idx = v;

    if (max_idx == r || max_idx == c) {
        std::cout << "[INFO] 1-based indexing detected. Converting to 0-based..." << std::endl;
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

template <class T = float>
void verify_solution(const std::string& test_name, const std::vector<T>& actual, 
                     const std::vector<T>& expected, T tol = 1e-3) {
    if (actual.size() != expected.size()) {
        std::cout << "[ERROR] " << test_name << " failed: Size mismatch (got " 
                  << actual.size() << ", expected " << expected.size() << ")" << std::endl;
        return;
    }
    bool all_correct = true;
    for (size_t i = 0; i < actual.size(); ++i) {
        if (std::isnan(actual[i]) || std::isinf(actual[i]) || std::abs(actual[i] - expected[i]) > tol) {
            all_correct = false;
            std::cout << "[ERROR] " << test_name << " mismatch at index " << i 
                      << ": Expected " << expected[i] 
                      << ", Got " << actual[i] << (std::isnan(actual[i]) ? " (NaN)" : "") << std::endl;
            break;
        }
    }
    if (all_correct) {
        std::cout << "[SUCCESS] " << test_name << " converged to correct solution." << std::endl;
    }
}

void caf_main(actor_system& sys) {
    // Enable cuBLAS and cuSPARSE for the CG solver
    manager::init(sys, manager_config(true, true)); 

    // Simple Diagonal Matrix A (3x3): diag(4, 3, 2)
    // b = [8, 9, 2] -> Expected x = [2, 3, 1]
    int n = 3, nnz = 3;
    std::vector<float> h_b = {8.0f, 9.0f, 2.0f};
    std::vector<float> expected = {2.0f, 3.0f, 1.0f};
    float tolerance = 1e-5f;
    int max_iter = 100;

    scoped_actor self{sys};

    // Test 1: CSR Format
    {
        std::cout << "[INFO] Test 1: CSR format simple matrix..." << std::endl;
        std::vector<int> row_ptr = {0, 1, 2, 3};
        std::vector<int> col_ind = {0, 1, 2};
        std::vector<float> values = {4.0f, 3.0f, 2.0f};
        std::vector<float> h_x(n, 0.0f);

        auto solver = sys.spawn<sparse_bicgstab_actor<float>>(
            create_in_arg(row_ptr), create_in_arg(col_ind), create_in_arg(values),
            create_in_arg(h_b), create_in_out_arg(h_x),
            matrix_format::csr, n, nnz, tolerance, max_iter, 0, 0, actor_cast<actor>(self));

        self->mail(start_atom_v).send(solver);
        self->receive(
            [&](std::vector<float> result_x, solver_result_meta meta) {
                std::cout << "[INFO] Iterations: " << meta.iterations << ", Converged: " << std::boolalpha << meta.converged << std::endl;
                verify_solution("CSR Simple", result_x, expected, tolerance);
            }
        );
    }

    // Test 2: CSC Format
    {
        std::cout << "\n[INFO] Test 2: CSC format simple matrix..." << std::endl;
        std::vector<int> col_ptr = {0, 1, 2, 3};
        std::vector<int> row_ind = {0, 1, 2};
        std::vector<float> values = {4.0f, 3.0f, 2.0f};
        std::vector<float> h_x(n, 0.0f);

        auto solver = sys.spawn<sparse_bicgstab_actor<float>>(
            create_in_arg(col_ptr), create_in_arg(row_ind), create_in_arg(values),
            create_in_arg(h_b), create_in_out_arg(h_x),
            matrix_format::csc, n, nnz, tolerance, max_iter, 0, 1, actor_cast<actor>(self));

        self->mail(start_atom_v).send(solver);
        self->receive(
            [&](std::vector<float> result_x, solver_result_meta meta) {
                verify_solution("CSC Simple", result_x, expected, tolerance);
            }
        );
    }

    // Test 3: COO Format
    {
        std::cout << "\n[INFO] Test 3: COO format simple matrix..." << std::endl;
        std::vector<int> row_ind = {0, 1, 2};
        std::vector<int> col_ind = {0, 1, 2};
        std::vector<float> values = {4.0f, 3.0f, 2.0f};
        std::vector<float> h_x(n, 0.0f);

        auto solver = sys.spawn<sparse_bicgstab_actor<float>>(
            create_in_arg(row_ind), create_in_arg(col_ind), create_in_arg(values),
            create_in_arg(h_b), create_in_out_arg(h_x),
            matrix_format::coo, n, nnz, tolerance, max_iter, 0, 2, actor_cast<actor>(self));

        self->mail(start_atom_v).send(solver);
        self->receive(
            [&](std::vector<float> result_x, solver_result_meta meta) {
                verify_solution("COO Simple", result_x, expected, tolerance);
            }
        );
    }

    // Test 4: Tridiagonal matrix (CSR) - Known solution x = [1, 1, 1]
    {
        std::cout << "\n[INFO] Test 4: CSR format tridiagonal matrix (3x3)..." << std::endl;
        // Matrix: [ 2 -1  0 ]
        //         [-1  2 -1 ]
        //         [ 0 -1  2 ]
        // b = [1, 0, 1] -> Expected x = [1, 1, 1]
        std::vector<int> row_ptr = {0, 2, 5, 7};
        std::vector<int> col_ind = {0, 1, 0, 1, 2, 1, 2};
        std::vector<float> values = {2.0f, -1.0f, -1.0f, 2.0f, -1.0f, -1.0f, 2.0f};
        std::vector<float> b_tri = {1.0f, 0.0f, 1.0f};
        std::vector<float> x_tri(3, 0.0f);
        std::vector<float> expected_tri = {1.0f, 1.0f, 1.0f};

        auto solver = sys.spawn<sparse_bicgstab_actor<float>>(
            create_in_arg(row_ptr), create_in_arg(col_ind), create_in_arg(values),
            create_in_arg(b_tri), create_in_out_arg(x_tri),
            matrix_format::csr, 3, 7, tolerance, max_iter, 0, 3, actor_cast<actor>(self));

        self->mail(start_atom_v).send(solver);
        self->receive(
            [&](std::vector<float> result_x, solver_result_meta meta) {
                verify_solution("CSR Tridiagonal (3x3)", result_x, expected_tri, tolerance);
            }
        );
    }

    // Test 5: Larger Tridiagonal Correctness (N=100)
    {
        int N_mid = 100;
        std::cout << "\n[INFO] Test 5: CSR format tridiagonal correctness (N=" << N_mid << ")..." << std::endl;
        
        std::vector<int> row_ptr;
        std::vector<int> col_ind;
        std::vector<float> values;
        std::vector<float> expected_mid(N_mid, 1.0f);
        std::vector<float> b_mid(N_mid, 0.0f);

        row_ptr.push_back(0);
        for(int i=0; i<N_mid; ++i) {
            float row_sum = 0.0f;
            if(i > 0) { 
                col_ind.push_back(i-1); 
                values.push_back(-1.0f); 
                row_sum += -1.0f;
            }
            col_ind.push_back(i); 
            values.push_back(2.1f); // Diagonally dominant to ensure convergence
            row_sum += 2.1f;
            if(i < N_mid-1) { 
                col_ind.push_back(i+1); 
                values.push_back(-1.0f); 
                row_sum += -1.0f;
            }
            row_ptr.push_back(col_ind.size());
            b_mid[i] = row_sum; // b = A * ones()
        }

        std::vector<float> x_mid(N_mid, 0.0f);
        
        auto solver = sys.spawn<sparse_bicgstab_actor<float>>(
            create_in_arg(row_ptr), create_in_arg(col_ind), create_in_arg(values),
            create_in_arg(b_mid), create_in_out_arg(x_mid),
            matrix_format::csr, N_mid, (int)values.size(), 1e-5f, 500, 0, 4, actor_cast<actor>(self));

        self->mail(start_atom_v).send(solver);
        self->receive(
            [&](std::vector<float> result, solver_result_meta meta) {
                verify_solution("CSR N=100 Correctness", result, expected_mid, 1e-4f);
            }
        );
    }

    // Test 6: Ill-conditioned / High Iteration Count Test (N=5000)
    // This uses a non-symmetric tridiagonal matrix with very weak diagonal dominance.
    // A_ii = 2.0001, A_{i,i-1} = -1.1, A_{i,i+1} = -0.9.
    // The near-singularity forces BiCGSTAB to take many iterations to converge.
    {
        int N_high = 5000;
        std::cout << "\n[INFO] Test 6: High Iteration Count Test (N=" << N_high << ")..." << std::endl;

        std::vector<int> row_ptr;
        std::vector<int> col_ind;
        std::vector<float> values;
        std::vector<float> expected_high(N_high, 1.0f);
        std::vector<float> b_high(N_high, 0.0f);

        row_ptr.push_back(0);
        for(int i=0; i<N_high; ++i) {
            float row_sum = 0.0f;
            if(i > 0) {
                col_ind.push_back(i-1);
                values.push_back(-1.1f);
                row_sum += -1.1f;
            }
            col_ind.push_back(i);
            values.push_back(2.0001f); // Extremely low diagonal dominance
            row_sum += 2.0001f;
            if(i < N_high-1) {
                col_ind.push_back(i+1);
                values.push_back(-0.9f);
                row_sum += -0.9f;
            }
            row_ptr.push_back(col_ind.size());
            b_high[i] = row_sum;
        }

        std::vector<float> x_high(N_high, 0.0f);

        auto solver = sys.spawn<sparse_bicgstab_actor<float>>(
            create_in_arg(row_ptr), create_in_arg(col_ind), create_in_arg(values),
            create_in_arg(b_high), create_in_out_arg(x_high),
            matrix_format::csr, N_high, (int)values.size(), 1e-6f, 15000, 0, 5, actor_cast<actor>(self));

        self->mail(start_atom_v).send(solver);
        self->receive(
            [&](std::vector<float> result, solver_result_meta meta) {
                verify_solution("High Iteration Count Test", result, expected_high, 1e-2f);
            }
        );
    }

    // Test 7: Real-world matrix from file (lnsp3937.bin)
    {
    // Test 7a: Real-world matrix (lnsp3937.bin) - Float Precision
    {
        std::string path = "/scratch/nqr159/matrix-collection/matrices/unsymmetric/lnsp3937.bin";
        std::cout << "\n[INFO] Test 7a: Loading real-world matrix (Float) " << path << "..." << std::endl;
        try {
            LocalCSR<float> A = load_binary_matrix_manual<float>(path);
            std::cout << "[INFO] Matrix Metadata: Rows=" << A.rows << ", Cols=" << A.cols 
                      << ", NNZ=" << A.nnz << std::endl;
            
            std::vector<float> expected_real(A.rows, 1.0f);
            std::vector<float> b_real = compute_rhs_manual<float>(A, expected_real);
            std::vector<float> x_real(A.rows, 0.0f);

            auto solver = sys.spawn<sparse_bicgstab_actor<float>>(
                create_in_arg(A.row_ptr), create_in_arg(A.col_ind), create_in_arg(A.values),
                create_in_arg(b_real), create_in_out_arg(x_real),
                matrix_format::csr, A.rows, A.nnz, 1e-5f, 5000, 0, 6, actor_cast<actor>(self));

            self->mail(start_atom_v).send(solver);
            self->receive(
                [&](std::vector<float> result, solver_result_meta meta) {
                    verify_solution("Real Matrix (Float)", result, expected_real, 1e-2f);
                }
            );
        } catch (const std::exception& e) {
            std::cout << "[ERROR] Test 7a Failed: " << e.what() << std::endl;
        }
    }

    // Test 7b: Real-world matrix (lnsp3937.bin) - Double Precision
    {
        std::string path = "/scratch/nqr159/matrix-collection/matrices/unsymmetric/lnsp3937.bin";
        std::cout << "\n[INFO] Test 7b: Loading real-world matrix (Double) " << path << "..." << std::endl;
        try {
            LocalCSR<double> A = load_binary_matrix_manual<double>(path);
            std::cout << "[INFO] Matrix Metadata: Rows=" << A.rows << ", Cols=" << A.cols 
                      << ", NNZ=" << A.nnz << std::endl;
            std::cout << "[INFO] First 5 matrix values: ";
            for(int i=0; i<std::min(5, (int)A.values.size()); ++i)
                std::cout << A.values[i] << " ";
            std::cout << std::endl;

            std::vector<double> expected_real(A.rows, 1.0);
            std::vector<double> b_real = compute_rhs_manual<double>(A, expected_real);
            std::vector<double> x_real(A.rows, 0.0);

            auto solver = sys.spawn<sparse_bicgstab_actor<double>>(
                create_in_arg(A.row_ptr), create_in_arg(A.col_ind), create_in_arg(A.values),
                create_in_arg(b_real), create_in_out_arg(x_real),
                matrix_format::csr, A.rows, A.nnz, 1e-10, 5000, 0, 7, actor_cast<actor>(self));

            self->mail(start_atom_v).send(solver);
            self->receive(
                [&](std::vector<double> result, solver_result_meta meta) {
                    verify_solution<double>("Real Matrix (Double)", result, expected_real, 1e-8);
                }
            );
        } catch (const std::exception& e) {
            std::cout << "[ERROR] Test 7b Failed: " << e.what() << std::endl;
        }
    }
    }

    // Test 8: Stress Test - 1D Laplacian (N=10000)
    {
        int N_large = 10000;
        std::cout << "\n[INFO] Test 8: Stress Test - Non-Symmetric Matrix (N=" << N_large << ")..." << std::endl;
        
        std::vector<int> row_ptr;
        std::vector<int> col_ind;
        std::vector<float> values;
        row_ptr.push_back(0);
        for(int i=0; i<N_large; ++i) {
            if(i > 0) { col_ind.push_back(i-1); values.push_back(-1.5f); } // Lower
            col_ind.push_back(i); values.push_back(4.0f);                 // Diag
            if(i < N_large-1) { col_ind.push_back(i+1); values.push_back(-0.5f); } // Upper
            row_ptr.push_back(col_ind.size());
        }

        std::vector<float> b_large(N_large, 1.0f);
        std::vector<float> x_large(N_large, 0.0f);
        
        auto start = std::chrono::high_resolution_clock::now();
        auto stress_solver = sys.spawn<sparse_bicgstab_actor<float>>(
            create_in_arg(row_ptr), create_in_arg(col_ind), create_in_arg(values),
            create_in_arg(b_large), create_in_out_arg(x_large),
            matrix_format::csr, N_large, (int)values.size(), 1e-4f, 20000, 0, 7, actor_cast<actor>(self));

        self->mail(start_atom_v).send(stress_solver);
        self->receive(
            [&](std::vector<float> result, solver_result_meta meta) {
                auto end = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> elapsed = end - start;
                std::cout << "[SUCCESS] Stress Test completed in " << elapsed.count() << " seconds." << std::endl;
                std::cout << "[INFO] Iterations: " << meta.iterations << ", Converged: " << meta.converged << std::endl;
                std::cout << "[INFO] First 5 elements of solution: ";
                for(int i=0; i<5; ++i) std::cout << result[i] << " ";
                std::cout << "..." << std::endl;
            }
        );
    }

    // Test 9: Double Precision Test (CSR)
    {
        std::cout << "\n[INFO] Test 9: Double precision CSR format..." << std::endl;
        int n_d = 3, nnz_d = 3;
        std::vector<int> row_ptr = {0, 1, 2, 3};
        std::vector<int> col_ind = {0, 1, 2};
        std::vector<double> values = {4.0, 3.0, 2.0};
        std::vector<double> h_b = {8.0, 9.0, 2.0};
        std::vector<double> h_x(n_d, 0.0);
        std::vector<double> expected_d = {2.0, 3.0, 1.0};

        auto solver = sys.spawn<sparse_bicgstab_actor<double>>(
            create_in_arg(row_ptr), create_in_arg(col_ind), create_in_arg(values),
            create_in_arg(h_b), create_in_out_arg(h_x),
            matrix_format::csr, n_d, nnz_d, 1e-10f, 100, 0, 8, actor_cast<actor>(self));

        self->mail(start_atom_v).send(solver);
        self->receive(
            [&](std::vector<double> result_x, solver_result_meta meta) {
                verify_solution<double>("Double CSR Simple", result_x, expected_d, 1e-9);
            }
        );
    }

    // Test 10: Facade Actor CSR Simple
    {
        std::cout << "\n[INFO] Test 10: Facade actor CSR simple matrix..." << std::endl;
        int n_f = 3, nnz_f = 3;
        std::vector<int> row_ptr = {0, 1, 2, 3};
        std::vector<int> col_ind = {0, 1, 2};
        std::vector<float> values = {4.0f, 3.0f, 2.0f};
        std::vector<float> h_b = {8.0f, 9.0f, 2.0f};
        std::vector<float> h_x(n_f, 0.0f);
        std::vector<float> expected_f = {2.0f, 3.0f, 1.0f};

        auto facade = sys.spawn<sparse_bicgstab_facade<float>>(100);

        self->mail(create_in_arg(row_ptr), create_in_arg(col_ind), create_in_arg(values),
                   create_in_arg(h_b), create_in_out_arg(h_x),
                   matrix_format::csr, n_f, nnz_f, tolerance, max_iter, 0, 9).send(facade);

        self->receive(
            [&](uint32_t /*resp_id*/, int /*idx*/, const std::vector<float>& result_x, solver_result_meta meta) {
                verify_solution("Facade CSR Simple", result_x, expected_f, tolerance);
            }
        );
    }

    // Test 11: Facade Actor Real Matrix
    {
        std::string path = "/scratch/nqr159/matrix-collection/matrices/unsymmetric/lnsp3937.bin";
        std::cout << "\n[INFO] Test 11: Facade actor real-world matrix " << path << "..." << std::endl;
        try {
            LocalCSR<float> A = load_binary_matrix_manual<float>(path);
            std::vector<float> expected_real(A.rows, 1.0f);
            std::vector<float> b_real = compute_rhs_manual<float>(A, expected_real);
            std::vector<float> x_real(A.rows, 0.0f);

            auto facade = sys.spawn<sparse_bicgstab_facade<float>>(100);

            self->mail(create_in_arg(A.row_ptr), create_in_arg(A.col_ind), create_in_arg(A.values),
                       create_in_arg(b_real), create_in_out_arg(x_real),
                       matrix_format::csr, A.rows, A.nnz, 1e-5f, 5000, 0, 10).send(facade);

            self->receive(
                [&](uint32_t /*resp_id*/, int /*idx*/, const std::vector<float>& result, solver_result_meta meta) {
                    verify_solution("Facade Real Matrix", result, expected_real, 1e-2f);
                }
            );
        } catch (const std::exception& e) {
            std::cout << "[ERROR] Test 11 Failed: " << e.what() << std::endl;
        }
    }

    // Test 12: Facade Actor CSR with Custom Buffer
    {
        std::cout << "\n[INFO] Test 12: Facade actor CSR with custom buffer..." << std::endl;
        std::vector<float> custom_x(n, 0.0f);
        std::vector<int> row_ptr = {0, 1, 2, 3};
        std::vector<int> col_ind = {0, 1, 2};
        std::vector<float> values = {4.0f, 3.0f, 2.0f};
        output_mapping m{4, custom_x.data(), (size_t)n};
        
        auto facade = sys.spawn<sparse_bicgstab_facade<float>>(100);
        self->mail(std::vector<output_mapping>{m},
                   create_in_arg(row_ptr), create_in_arg(col_ind), create_in_arg(values),
                   create_in_arg(h_b), create_in_out_arg(custom_x),
                   matrix_format::csr, n, nnz, tolerance, max_iter, 0, 11).send(facade);

        self->receive(
            [&](uint32_t /*resp_id*/, int index, solver_result_meta meta) {
                if (index == 4)
                  verify_solution("Facade Custom Buffer", custom_x, expected);
            }
        );
    }

    // Test 13: Facade Actor CSR returning mem_ptr handles
    {
        std::cout << "\n[INFO] Test 13: Facade actor CSR returning mem_ptr..." << std::endl;
        std::vector<int> row_ptr = {0, 1, 2, 3};
        std::vector<int> col_ind = {0, 1, 2};
        std::vector<float> values = {4.0f, 3.0f, 2.0f};
        std::vector<float> h_x(n, 0.0f);

        auto facade = sys.spawn<sparse_bicgstab_facade<float>>(100);

        self->mail(return_mem_ptr_atom_v,
                   create_in_arg(row_ptr), create_in_arg(col_ind), create_in_arg(values),
                   create_in_arg(h_b), create_in_out_arg(h_x),
                   matrix_format::csr, n, nnz, tolerance, max_iter, 0, 12).send(facade);

        self->receive(
            [&](uint32_t /*resp_id*/, mem_ptr<float> ptr, solver_result_meta meta) {
                auto result_x = ptr->copy_to_host();
                verify_solution("Facade mem_ptr", result_x, expected);
            }
        );
    }

    // Test 14: Jacobi Facade Actor CSR Simple
    {
        std::cout << "\n[INFO] Test 14: Jacobi Facade actor CSR simple matrix..." << std::endl;
        std::vector<int> row_ptr = {0, 1, 2, 3};
        std::vector<int> col_ind = {0, 1, 2};
        std::vector<float> values = {4.0f, 3.0f, 2.0f};
        std::vector<float> h_x(n, 0.0f);

        auto facade = sys.spawn<sparse_bicgstab_jacobi_facade<float>>(200);

        self->mail(std::vector<output_mapping>{},
                   create_in_arg(row_ptr), create_in_arg(col_ind), create_in_arg(values),
                   create_in_arg(h_b), create_in_out_arg(h_x),
                   matrix_format::csr, n, nnz, tolerance, max_iter, 0, 13).send(facade);

        self->receive(
            [&](uint32_t /*resp_id*/, int /*idx*/, const std::vector<float>& result_x, solver_result_meta meta) {
                verify_solution("Jacobi Facade CSR Simple", result_x, expected);
            }
        );
    }

    // Test 15: Jacobi Facade Actor CSR with Custom Buffer
    {
        std::cout << "\n[INFO] Test 15: Jacobi Facade actor CSR with custom buffer..." << std::endl;
        std::vector<float> custom_x(n, 0.0f);
        std::vector<int> row_ptr = {0, 1, 2, 3};
        std::vector<int> col_ind = {0, 1, 2};
        std::vector<float> values = {4.0f, 3.0f, 2.0f};
        output_mapping m{4, custom_x.data(), (size_t)n};
        
        auto facade = sys.spawn<sparse_bicgstab_jacobi_facade<float>>(200);
        self->mail(std::vector<output_mapping>{m},
                   create_in_arg(row_ptr), create_in_arg(col_ind), create_in_arg(values),
                   create_in_arg(h_b), create_in_out_arg(custom_x),
                   matrix_format::csr, n, nnz, tolerance, max_iter, 0, 14).send(facade);

        self->receive(
            [&](uint32_t /*resp_id*/, int index, solver_result_meta meta) {
                if (index == 4)
                  verify_solution("Jacobi Facade Custom Buffer", custom_x, expected);
            }
        );
    }

    // Test 16: Jacobi Facade Actor CSR returning mem_ptr handles
    {
        std::cout << "\n[INFO] Test 16: Jacobi Facade actor CSR returning mem_ptr..." << std::endl;
        std::vector<int> row_ptr = {0, 1, 2, 3};
        std::vector<int> col_ind = {0, 1, 2};
        std::vector<float> values = {4.0f, 3.0f, 2.0f};
        std::vector<float> h_x(n, 0.0f);

        auto facade = sys.spawn<sparse_bicgstab_jacobi_facade<float>>(200);

        self->mail(return_mem_ptr_atom_v,
                   create_in_arg(row_ptr), create_in_arg(col_ind), create_in_arg(values),
                   create_in_arg(h_b), create_in_out_arg(h_x),
                   matrix_format::csr, n, nnz, tolerance, max_iter, 0, 15).send(facade);

        self->receive(
            [&](uint32_t /*resp_id*/, mem_ptr<float> ptr, solver_result_meta meta) {
                auto result_x = ptr->copy_to_host();
                verify_solution("Jacobi Facade mem_ptr", result_x, expected);
            }
        );
    }

    // Test 17: Jacobi Facade Actor CSC Simple
    {
        std::cout << "\n[INFO] Test 17: Jacobi Facade actor CSC simple matrix..." << std::endl;
        std::vector<int> col_ptr = {0, 1, 2, 3};
        std::vector<int> row_ind = {0, 1, 2};
        std::vector<float> values = {4.0f, 3.0f, 2.0f};
        std::vector<float> h_x(n, 0.0f);

        auto facade = sys.spawn<sparse_bicgstab_jacobi_facade<float>>(200);

        self->mail(std::vector<output_mapping>{},
                   create_in_arg(col_ptr), create_in_arg(row_ind), create_in_arg(values),
                   create_in_arg(h_b), create_in_out_arg(h_x),
                   matrix_format::csc, n, nnz, tolerance, max_iter, 0, 16).send(facade);

        self->receive(
            [&](uint32_t /*resp_id*/, int /*idx*/, const std::vector<float>& result_x, solver_result_meta meta) {
                verify_solution("Jacobi Facade CSC Simple", result_x, expected);
            }
        );
    }

    // Test 18: Jacobi Facade Actor Real Matrix
    {
        std::string path = "/scratch/nqr159/matrix-collection/matrices/unsymmetric/lnsp3937.bin";
        std::cout << "\n[INFO] Test 18: Jacobi Facade actor real-world matrix " << path << "..." << std::endl;
        try {
            LocalCSR<float> A = load_binary_matrix_manual<float>(path);
            std::vector<float> expected_real(A.rows, 1.0f);
            std::vector<float> b_real = compute_rhs_manual<float>(A, expected_real);
            std::vector<float> x_real(A.rows, 0.0f);

            auto facade = sys.spawn<sparse_bicgstab_jacobi_facade<float>>(200);

            self->mail(create_in_arg(A.row_ptr), create_in_arg(A.col_ind), create_in_arg(A.values),
                       create_in_arg(b_real), create_in_out_arg(x_real),
                       matrix_format::csr, A.rows, A.nnz, 1e-5f, 5000, 0, 17).send(facade);

            self->receive(
                [&](uint32_t /*resp_id*/, int /*idx*/, const std::vector<float>& result, solver_result_meta meta) {
                    verify_solution("Jacobi Facade Real Matrix", result, expected_real, 1e-2f);
                }
            );
        } catch (const std::exception& e) {
            std::cout << "[ERROR] Test 18 Failed: " << e.what() << std::endl;
        }
    }

    manager::shutdown();
}
CAF_MAIN(id_block::cuda)
