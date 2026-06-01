/**
 * This test file evaluates the correctness of sparse_cg_actor for solving Ax = b.
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
#include <algorithm>
#include <numeric>
#include <cmath>
#include "caf/actor_registry.hpp"
#include "caf/actorSOLVE/sparse-matrix-solvers/sparse-CGS-actor/sparse-CGS-actor.hpp"

using namespace caf;
using namespace caf::cuda;

template <class T = float>
void verify_solution(const std::string& test_name, const std::vector<T>& actual, 
                     const std::vector<T>& expected, T tol = static_cast<T>(1e-3)) {
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
                      << ", Got " << actual[i] << std::endl;
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

        auto solver = sys.spawn<sparse_cg_actor<float>>(
            create_in_arg(row_ptr), create_in_arg(col_ind), create_in_arg(values),
            create_in_arg(h_b), create_in_out_arg(h_x),
            matrix_format::csr, n, nnz, tolerance, max_iter, 0, 0, actor_cast<actor>(self));

        self->mail(start_atom_v).send(solver);
        self->receive(
            [&](std::vector<float> result_x, solver_result_meta meta) {
                std::cout << "[INFO] Iterations: " << meta.iterations << ", Converged: " << std::boolalpha << meta.converged << std::endl;
                verify_solution("CSR Simple", result_x, expected);
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

        auto solver = sys.spawn<sparse_cg_actor<float>>(
            create_in_arg(col_ptr), create_in_arg(row_ind), create_in_arg(values),
            create_in_arg(h_b), create_in_out_arg(h_x),
            matrix_format::csc, n, nnz, tolerance, max_iter, 0, 1, actor_cast<actor>(self));

        self->mail(start_atom_v).send(solver);
        self->receive(
            [&](std::vector<float> result_x, solver_result_meta meta) {
                verify_solution("CSC Simple", result_x, expected);
            }
        );
    }

    // Test 3: Stress Test - 1D Laplacian (N=10000)
    {
        int N_large = 10000;
        std::cout << "\n[INFO] Test 3: Stress Test - 1D Laplacian (N=" << N_large << ")..." << std::endl;
        
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
        
        auto start = std::chrono::high_resolution_clock::now();
        auto stress_solver = sys.spawn<sparse_cg_actor<float>>(
            create_in_arg(row_ptr), create_in_arg(col_ind), create_in_arg(values),
            create_in_arg(b_large), create_in_out_arg(x_large),
            matrix_format::csr, N_large, (int)values.size(), 1e-4f, 20000, 0, 2, actor_cast<actor>(self));

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

    // Test 4: Facade Actor CSR Simple
    {
        std::cout << "\n[INFO] Test 4: Facade actor CSR simple matrix..." << std::endl;
        std::vector<int> row_ptr = {0, 1, 2, 3};
        std::vector<int> col_ind = {0, 1, 2};
        std::vector<float> values = {4.0f, 3.0f, 2.0f};
        std::vector<float> h_x(n, 0.0f);

        auto facade = sys.spawn<sparse_cg_facade<float>>(100);

        self->mail(std::vector<output_mapping>{},
                   create_in_arg(row_ptr), create_in_arg(col_ind), create_in_arg(values),
                   create_in_arg(h_b), create_in_out_arg(h_x),
                   matrix_format::csr, n, nnz, tolerance, max_iter, 0, 3).send(facade);

        self->receive(
            [&](uint32_t /*resp_id*/, int /*idx*/, const std::vector<float>& result_x, solver_result_meta meta) {
                verify_solution("Facade CSR Simple", result_x, expected);
            }
        );
    }

    // Test 5: Facade Actor CSR with Custom Buffer
    {
        std::cout << "\n[INFO] Test 5: Facade actor CSR with custom buffer..." << std::endl;
        std::vector<float> custom_x(n, 0.0f);
        std::vector<int> row_ptr = {0, 1, 2, 3};
        std::vector<int> col_ind = {0, 1, 2};
        std::vector<float> values = {4.0f, 3.0f, 2.0f};
        output_mapping m{4, custom_x.data(), (size_t)n};
        
        auto facade = sys.spawn<sparse_cg_facade<float>>(100);
        self->mail(std::vector<output_mapping>{m},
                   create_in_arg(row_ptr), create_in_arg(col_ind), create_in_arg(values),
                   create_in_arg(h_b), create_in_out_arg(custom_x),
                   matrix_format::csr, n, nnz, tolerance, max_iter, 0, 4).send(facade);

        self->receive(
            [&](uint32_t /*resp_id*/, int index, solver_result_meta meta) {
                if (index == 4)
                  verify_solution("Facade Custom Buffer", custom_x, expected);
            }
        );
    }

    // Test 6: Facade Actor CSR returning mem_ptr handles
    {
        std::cout << "\n[INFO] Test 6: Facade actor CSR returning mem_ptr..." << std::endl;
        std::vector<int> row_ptr = {0, 1, 2, 3};
        std::vector<int> col_ind = {0, 1, 2};
        std::vector<float> values = {4.0f, 3.0f, 2.0f};
        std::vector<float> h_x(n, 0.0f);

        auto facade = sys.spawn<sparse_cg_facade<float>>(100);

        self->mail(return_mem_ptr_atom_v,
                   create_in_arg(row_ptr), create_in_arg(col_ind), create_in_arg(values),
                   create_in_arg(h_b), create_in_out_arg(h_x),
                   matrix_format::csr, n, nnz, tolerance, max_iter, 0, 5).send(facade);

        self->receive(
            [&](uint32_t /*resp_id*/, mem_ptr<float> ptr, solver_result_meta meta) {
                auto result_x = ptr->copy_to_host();
                verify_solution("Facade mem_ptr", result_x, expected);
            }
        );
    }

    // Test 7: Facade Actor Default (No mapping vector - hits Mode 3 handler)
    {
        std::cout << "\n[INFO] Test 7: Facade actor default (no mapping vector)..." << std::endl;
        std::vector<int> row_ptr = {0, 1, 2, 3};
        std::vector<int> col_ind = {0, 1, 2};
        std::vector<float> values = {4.0f, 3.0f, 2.0f};
        std::vector<float> h_x(n, 0.0f);

        auto facade = sys.spawn<sparse_cg_facade<float>>(100);

        self->mail(create_in_arg(row_ptr), create_in_arg(col_ind), create_in_arg(values),
                   create_in_arg(h_b), create_in_out_arg(h_x),
                   matrix_format::csr, n, nnz, tolerance, max_iter, 0, 6).send(facade);

        self->receive(
            [&](uint32_t /*resp_id*/, int /*idx*/, const std::vector<float>& result_x, solver_result_meta meta) {
                verify_solution("Facade Default", result_x, expected);
            }
        );
    }

    // Test 8: Facade Actor CSC Simple
    {
        std::cout << "\n[INFO] Test 8: Facade actor CSC simple matrix..." << std::endl;
        std::vector<int> col_ptr = {0, 1, 2, 3};
        std::vector<int> row_ind = {0, 1, 2};
        std::vector<float> values = {4.0f, 3.0f, 2.0f};
        std::vector<float> h_x(n, 0.0f);

        auto facade = sys.spawn<sparse_cg_facade<float>>(100);

        self->mail(std::vector<output_mapping>{},
                   create_in_arg(col_ptr), create_in_arg(row_ind), create_in_arg(values),
                   create_in_arg(h_b), create_in_out_arg(h_x),
                   matrix_format::csc, n, nnz, tolerance, max_iter, 0, 7).send(facade);

        self->receive(
            [&](uint32_t /*resp_id*/, int /*idx*/, const std::vector<float>& result_x, solver_result_meta meta) {
                verify_solution("Facade CSC Simple", result_x, expected);
            }
        );
    }

    // Test 9: Jacobi Facade Actor CSR Simple
    {
        std::cout << "\n[INFO] Test 9: Jacobi Facade actor CSR simple matrix..." << std::endl;
        std::vector<int> row_ptr = {0, 1, 2, 3};
        std::vector<int> col_ind = {0, 1, 2};
        std::vector<float> values = {4.0f, 3.0f, 2.0f};
        std::vector<float> h_x(n, 0.0f);

        auto facade = sys.spawn<sparse_cg_jacobi_facade<float>>(200);

        self->mail(std::vector<output_mapping>{},
                   create_in_arg(row_ptr), create_in_arg(col_ind), create_in_arg(values),
                   create_in_arg(h_b), create_in_out_arg(h_x),
                   matrix_format::csr, n, nnz, tolerance, max_iter, 0, 8).send(facade);

        self->receive(
            [&](uint32_t /*resp_id*/, int /*idx*/, const std::vector<float>& result_x, solver_result_meta meta) {
                verify_solution("Jacobi Facade CSR Simple", result_x, expected);
            }
        );
    }

    // Test 10: Jacobi Facade Actor CSR with Custom Buffer
    {
        std::cout << "\n[INFO] Test 10: Jacobi Facade actor CSR with custom buffer..." << std::endl;
        std::vector<float> custom_x(n, 0.0f);
        std::vector<int> row_ptr = {0, 1, 2, 3};
        std::vector<int> col_ind = {0, 1, 2};
        std::vector<float> values = {4.0f, 3.0f, 2.0f};
        output_mapping m{4, custom_x.data(), (size_t)n};
        
        auto facade = sys.spawn<sparse_cg_jacobi_facade<float>>(200);
        self->mail(std::vector<output_mapping>{m},
                   create_in_arg(row_ptr), create_in_arg(col_ind), create_in_arg(values),
                   create_in_arg(h_b), create_in_out_arg(custom_x),
                   matrix_format::csr, n, nnz, tolerance, max_iter, 0, 9).send(facade);

        self->receive(
            [&](uint32_t /*resp_id*/, int index, solver_result_meta meta) {
                if (index == 4)
                  verify_solution("Jacobi Facade Custom Buffer", custom_x, expected);
            }
        );
    }

    // Test 11: Jacobi Facade Actor CSR returning mem_ptr handles
    {
        std::cout << "\n[INFO] Test 11: Jacobi Facade actor CSR returning mem_ptr..." << std::endl;
        std::vector<int> row_ptr = {0, 1, 2, 3};
        std::vector<int> col_ind = {0, 1, 2};
        std::vector<float> values = {4.0f, 3.0f, 2.0f};
        std::vector<float> h_x(n, 0.0f);

        auto facade = sys.spawn<sparse_cg_jacobi_facade<float>>(200);

        self->mail(return_mem_ptr_atom_v,
                   create_in_arg(row_ptr), create_in_arg(col_ind), create_in_arg(values),
                   create_in_arg(h_b), create_in_out_arg(h_x),
                   matrix_format::csr, n, nnz, tolerance, max_iter, 0, 10).send(facade);

        self->receive(
            [&](uint32_t /*resp_id*/, mem_ptr<float> ptr, solver_result_meta meta) {
                auto result_x = ptr->copy_to_host();
                verify_solution("Jacobi Facade mem_ptr", result_x, expected);
            }
        );
    }

    // Test 12: Jacobi Facade Actor Default (No mapping vector)
    {
        std::cout << "\n[INFO] Test 12: Jacobi Facade actor default (no mapping vector)..." << std::endl;
        std::vector<int> row_ptr = {0, 1, 2, 3};
        std::vector<int> col_ind = {0, 1, 2};
        std::vector<float> values = {4.0f, 3.0f, 2.0f};
        std::vector<float> h_x(n, 0.0f);

        auto facade = sys.spawn<sparse_cg_jacobi_facade<float>>(200);

        self->mail(create_in_arg(row_ptr), create_in_arg(col_ind), create_in_arg(values),
                   create_in_arg(h_b), create_in_out_arg(h_x),
                   matrix_format::csr, n, nnz, tolerance, max_iter, 0, 11).send(facade);

        self->receive(
            [&](uint32_t /*resp_id*/, int /*idx*/, const std::vector<float>& result_x, solver_result_meta meta) {
                verify_solution("Jacobi Facade Default", result_x, expected);
            }
        );
    }

    // Test 13: Jacobi Facade Actor CSC Simple
    {
        std::cout << "\n[INFO] Test 13: Jacobi Facade actor CSC simple matrix..." << std::endl;
        std::vector<int> col_ptr = {0, 1, 2, 3};
        std::vector<int> row_ind = {0, 1, 2};
        std::vector<float> values = {4.0f, 3.0f, 2.0f};
        std::vector<float> h_x(n, 0.0f);

        auto facade = sys.spawn<sparse_cg_jacobi_facade<float>>(200);

        self->mail(std::vector<output_mapping>{},
                   create_in_arg(col_ptr), create_in_arg(row_ind), create_in_arg(values),
                   create_in_arg(h_b), create_in_out_arg(h_x),
                   matrix_format::csc, n, nnz, tolerance, max_iter, 0, 12).send(facade);

        self->receive(
            [&](uint32_t /*resp_id*/, int /*idx*/, const std::vector<float>& result_x, solver_result_meta meta) {
                verify_solution("Jacobi Facade CSC Simple", result_x, expected);
            }
        );
    }

    // Test 14: Double Precision CSR Simple
    {
        std::cout << "\n[INFO] Test 14: Double precision CSR format..." << std::endl;
        int n_d = 3, nnz_d = 3;
        std::vector<int> row_ptr = {0, 1, 2, 3};
        std::vector<int> col_ind = {0, 1, 2};
        std::vector<double> values = {4.0, 3.0, 2.0};
        std::vector<double> h_b_d = {8.0, 9.0, 2.0};
        std::vector<double> expected_d = {2.0, 3.0, 1.0};
        std::vector<double> h_x(n_d, 0.0);

        auto solver = sys.spawn<sparse_cg_actor<double>>(
            create_in_arg(row_ptr), create_in_arg(col_ind), create_in_arg(values),
            create_in_arg(h_b_d), create_in_out_arg(h_x),
            matrix_format::csr, n_d, nnz_d, 1e-10, 100, 0, 13, actor_cast<actor>(self));

        self->mail(start_atom_v).send(solver);
        self->receive(
            [&](std::vector<double> result_x, solver_result_meta meta) {
                std::cout << "[INFO] Iterations: " << meta.iterations << ", Converged: " << std::boolalpha << meta.converged << std::endl;
                verify_solution<double>("Double CSR Simple", result_x, expected_d, 1e-8);
            }
        );
    }

    // Test 15: Double Precision Facade CSR
    {
        std::cout << "\n[INFO] Test 15: Double precision facade CSR..." << std::endl;
        int n_d = 3, nnz_d = 3;
        std::vector<int> row_ptr = {0, 1, 2, 3};
        std::vector<int> col_ind = {0, 1, 2};
        std::vector<double> values = {4.0, 3.0, 2.0};
        std::vector<double> h_b_d = {8.0, 9.0, 2.0};
        std::vector<double> expected_d = {2.0, 3.0, 1.0};
        std::vector<double> h_x(n_d, 0.0);

        auto facade = sys.spawn<sparse_cg_facade<double>>(100);

        self->mail(create_in_arg(row_ptr), create_in_arg(col_ind), create_in_arg(values),
                   create_in_arg(h_b_d), create_in_out_arg(h_x),
                   matrix_format::csr, n_d, nnz_d, 1e-10, 100, 0, 14).send(facade);

        self->receive(
            [&](uint32_t /*resp_id*/, int /*idx*/, const std::vector<double>& result_x, solver_result_meta meta) {
                verify_solution<double>("Double Facade CSR", result_x, expected_d, 1e-8);
            }
        );
    }

    // Test 16: Double Precision Jacobi Facade CSR
    {
        std::cout << "\n[INFO] Test 16: Double precision Jacobi facade CSR..." << std::endl;
        int n_d = 3, nnz_d = 3;
        std::vector<int> row_ptr = {0, 1, 2, 3};
        std::vector<int> col_ind = {0, 1, 2};
        std::vector<double> values = {4.0, 3.0, 2.0};
        std::vector<double> h_b_d = {8.0, 9.0, 2.0};
        std::vector<double> expected_d = {2.0, 3.0, 1.0};
        std::vector<double> h_x(n_d, 0.0);

        auto facade = sys.spawn<sparse_cg_jacobi_facade<double>>(200);

        self->mail(create_in_arg(row_ptr), create_in_arg(col_ind), create_in_arg(values),
                   create_in_arg(h_b_d), create_in_out_arg(h_x),
                   matrix_format::csr, n_d, nnz_d, 1e-10, 100, 0, 15).send(facade);

        self->receive(
            [&](uint32_t /*resp_id*/, int /*idx*/, const std::vector<double>& result_x, solver_result_meta meta) {
                verify_solution<double>("Double Jacobi Facade CSR", result_x, expected_d, 1e-8);
            }
        );
    }

    // Test 17: Double Precision Stress Test (N=5000)
    {
        int N_st = 5000;
        std::cout << "\n[INFO] Test 17: Double precision Stress Test (N=" << N_st << ")..." << std::endl;
        
        std::vector<int> row_ptr;
        std::vector<int> col_ind;
        std::vector<double> values;
        row_ptr.push_back(0);
        for(int i=0; i<N_st; ++i) {
            if(i > 0) { col_ind.push_back(i-1); values.push_back(-1.0); }
            col_ind.push_back(i); values.push_back(2.0);
            if(i < N_st-1) { col_ind.push_back(i+1); values.push_back(-1.0); }
            row_ptr.push_back(col_ind.size());
        }

        std::vector<double> b_large(N_st, 1.0);
        std::vector<double> x_large(N_st, 0.0);
        
        auto start = std::chrono::high_resolution_clock::now();
        auto stress_solver = sys.spawn<sparse_cg_actor<double>>(
            create_in_arg(row_ptr), create_in_arg(col_ind), create_in_arg(values),
            create_in_arg(b_large), create_in_out_arg(x_large),
            matrix_format::csr, N_st, (int)values.size(), 1e-12, 10000, 0, 16, actor_cast<actor>(self));

        self->mail(start_atom_v).send(stress_solver);
        self->receive(
            [&](std::vector<double> result, solver_result_meta meta) {
                auto end = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> elapsed = end - start;
                std::cout << "[SUCCESS] Double Stress Test completed in " << elapsed.count() << " seconds." << std::endl;
                std::cout << "[INFO] Iterations: " << meta.iterations << ", Converged: " << std::boolalpha << meta.converged << std::endl;
            }
        );
    }

    // Test 18: Optimized Facade CSR Simple
    {
        std::cout << "\n[INFO] Test 18: Optimized Facade actor CSR simple matrix..." << std::endl;
        std::vector<int> row_ptr = {0, 1, 2, 3};
        std::vector<int> col_ind = {0, 1, 2};
        std::vector<float> values = {4.0f, 3.0f, 2.0f};
        std::vector<float> h_x(n, 0.0f);

        // Spawn the optimized facade subclass
        auto facade = sys.spawn<sparse_cg_facade_optimized<float>>(300);

        self->mail(create_in_arg(row_ptr), create_in_arg(col_ind), create_in_arg(values),
                   create_in_arg(h_b), create_in_out_arg(h_x),
                   matrix_format::csr, n, nnz, tolerance, max_iter, 0, 17).send(facade);

        self->receive(
            [&](uint32_t /*resp_id*/, int /*idx*/, const std::vector<float>& result_x, solver_result_meta meta) {
                std::cout << "[INFO] Iterations: " << meta.iterations << ", Converged: " << meta.converged << std::endl;
                verify_solution("Optimized Facade CSR Simple", result_x, expected);
            }
        );
    }

    // Test 19: Optimized Facade Stress Test (N=10000)
    {
        int N_large = 10000;
        std::cout << "\n[INFO] Test 19: Optimized Facade Stress Test - 1D Laplacian (N=" << N_large << ")..." << std::endl;
        
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
        
        auto facade = sys.spawn<sparse_cg_facade_optimized<float>>(301);
        auto start = std::chrono::high_resolution_clock::now();

        self->mail(create_in_arg(row_ptr), create_in_arg(col_ind), create_in_arg(values),
                   create_in_arg(b_large), create_in_out_arg(x_large),
                   matrix_format::csr, N_large, (int)values.size(), 1e-4f, 20000, 0, 18).send(facade);

        self->receive(
            [&](uint32_t /*resp_id*/, int /*idx*/, const std::vector<float>& result, solver_result_meta meta) {
                auto end = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> elapsed = end - start;
                std::cout << "[SUCCESS] Optimized Stress Test completed in " << elapsed.count() << " seconds." << std::endl;
                std::cout << "[INFO] Iterations: " << meta.iterations << ", Converged: " << meta.converged << std::endl;
            }
        );
    }

    // Test 20: Optimized Double Precision Facade CSR
    {
        std::cout << "\n[INFO] Test 20: Optimized Double precision facade CSR..." << std::endl;
        int n_d = 3, nnz_d = 3;
        std::vector<int> row_ptr = {0, 1, 2, 3};
        std::vector<int> col_ind = {0, 1, 2};
        std::vector<double> values = {4.0, 3.0, 2.0};
        std::vector<double> h_b_d = {8.0, 9.0, 2.0};
        std::vector<double> expected_d = {2.0, 3.0, 1.0};
        std::vector<double> h_x(n_d, 0.0);

        auto facade = sys.spawn<sparse_cg_facade_optimized<double>>(400);

        self->mail(create_in_arg(row_ptr), create_in_arg(col_ind), create_in_arg(values),
                   create_in_arg(h_b_d), create_in_out_arg(h_x),
                   matrix_format::csr, n_d, nnz_d, 1e-10, 100, 0, 19).send(facade);

        self->receive(
            [&](uint32_t /*resp_id*/, int /*idx*/, const std::vector<double>& result_x, solver_result_meta meta) {
                std::cout << "[INFO] Iterations: " << meta.iterations << ", Converged: " << meta.converged << std::endl;
                verify_solution<double>("Optimized Double Facade CSR", result_x, expected_d, 1e-8);
            }
        );
    }

    // Test 21: Optimized Facade CSC Simple
    {
        std::cout << "\n[INFO] Test 21: Optimized Facade actor CSC simple matrix..." << std::endl;
        std::vector<int> col_ptr = {0, 1, 2, 3};
        std::vector<int> row_ind = {0, 1, 2};
        std::vector<float> values = {4.0f, 3.0f, 2.0f};
        std::vector<float> h_x(n, 0.0f);

        auto facade = sys.spawn<sparse_cg_facade_optimized<float>>(302);

        self->mail(create_in_arg(col_ptr), create_in_arg(row_ind), create_in_arg(values),
                   create_in_arg(h_b), create_in_out_arg(h_x),
                   matrix_format::csc, n, nnz, tolerance, max_iter, 0, 20).send(facade);

        self->receive(
            [&](uint32_t /*resp_id*/, int /*idx*/, const std::vector<float>& result_x, solver_result_meta meta) {
                std::cout << "[INFO] Iterations: " << meta.iterations << ", Converged: " << meta.converged << std::endl;
                verify_solution("Optimized Facade CSC Simple", result_x, expected);
            }
        );
    }

    // Test 22: Optimized Facade COO Simple
    {
        std::cout << "\n[INFO] Test 22: Optimized Facade actor COO simple matrix..." << std::endl;
        std::vector<int> row_ind = {0, 1, 2};
        std::vector<int> col_ind = {0, 1, 2};
        std::vector<float> values = {4.0f, 3.0f, 2.0f};
        std::vector<float> h_x(n, 0.0f);

        auto facade = sys.spawn<sparse_cg_facade_optimized<float>>(303);

        self->mail(create_in_arg(row_ind), create_in_arg(col_ind), create_in_arg(values),
                   create_in_arg(h_b), create_in_out_arg(h_x),
                   matrix_format::coo, n, nnz, tolerance, max_iter, 0, 21).send(facade);

        self->receive(
            [&](uint32_t /*resp_id*/, int /*idx*/, const std::vector<float>& result_x, solver_result_meta meta) {
                std::cout << "[INFO] Iterations: " << meta.iterations << ", Converged: " << meta.converged << std::endl;
                verify_solution("Optimized Facade COO Simple", result_x, expected);
            }
        );
    }

    manager::shutdown();
}

CAF_MAIN(id_block::cuda)
