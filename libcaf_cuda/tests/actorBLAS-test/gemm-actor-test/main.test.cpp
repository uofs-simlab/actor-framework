#include <caf/all.hpp>
#include <caf/cuda/all.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include "caf/actorBLAS/gemm-actor/gemm-actor.hpp"

using namespace caf;
using namespace caf::cuda;

template <typename T>
void verify_gemm(int m, int n, int k, T alpha, const std::vector<T>& A, const std::vector<T>& B, T beta, const std::vector<T>& C_init, const std::vector<T>& C_res) {
    bool passed = true;
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            T sum = 0;
            for (int l = 0; l < k; ++l) {
                sum += A[i * k + l] * B[l * n + j];
            }
            T expected = alpha * sum + beta * C_init[i * n + j];
            if (std::abs(C_res[i * n + j] - expected) > 1e-3) {
                std::cout << "[ERROR] Mismatch at (" << i << "," << j << "): Expected " << expected << ", Got " << C_res[i * n + j] << std::endl;
                passed = false;
                break;
            }
        }
        if (!passed) break;
    }
    if (passed) {
        std::cout << "[SUCCESS] GEMM operation produced correct results." << std::endl;
    }
}

void caf_main(actor_system& sys) {
    manager_config config(true);
    manager::init(sys, config);

    int m = 4, n = 4, k = 4;
    std::vector<float> h_A_f(m * k, 1.0f);
    std::vector<float> h_B_f(k * n, 2.0f);
    std::vector<float> h_C_f(m * n, 0.0f);

    std::vector<double> h_A_d(m * k, 1.0);
    std::vector<double> h_B_d(k * n, 2.0);
    std::vector<double> h_C_d(m * n, 0.0);

    auto float_gemm = sys.spawn<gemm_actor<float>>(1);
    auto double_gemm = sys.spawn<gemm_actor<double>>(2);

    scoped_actor self{sys};

    // Test 1: Float GEMM, Host buffers, standard return
    {
        std::cout << "[INFO] Test 1: Float GEMM, Host buffers..." << std::endl;
        self->mail(create_in_arg(h_A_f), create_in_arg(h_B_f), create_out_arg(h_C_f), m, n, k).send(float_gemm);
        self->receive(
            [&](int reply_id, int arg_index, std::vector<float> data) {
                if (arg_index == 2) { // index 2 corresponds to matrix C
                    verify_gemm(m, n, k, 1.0f, h_A_f, h_B_f, 0.0f, h_C_f, data);
                }
            }
        );
    }

    // Test 2: Double GEMM, Host buffers, standard return
    {
        std::cout << "\n[INFO] Test 2: Double GEMM, Host buffers..." << std::endl;
        self->mail(create_in_arg(h_A_d), create_in_arg(h_B_d), create_out_arg(h_C_d), m, n, k).send(double_gemm);
        self->receive(
            [&](int reply_id, int arg_index, std::vector<double> data) {
                if (arg_index == 2) {
                    verify_gemm(m, n, k, 1.0, h_A_d, h_B_d, 0.0, h_C_d, data);
                }
            }
        );
    }

    // Test 3: Float GEMM, mem_ptr inputs
    {
        std::cout << "\n[INFO] Test 3: Float GEMM, mem_ptr inputs..." << std::endl;
        command_runner<in<float>, in<float>, out<float>> runner;
        auto ptrs = runner.transfer_memory(0, 0, create_in_arg(h_A_f), create_in_arg(h_B_f), create_out_arg(h_C_f));
        self->mail(std::get<0>(ptrs), std::get<1>(ptrs), std::get<2>(ptrs), m, n, k).send(float_gemm);
        self->receive(
            [&](int reply_id, int arg_index, std::vector<float> data) {
                if (arg_index == 2) {
                    verify_gemm(m, n, k, 1.0f, h_A_f, h_B_f, 0.0f, h_C_f, data);
                }
            }
        );
    }

    // Test 4: Float GEMM, return_mem_ptr_atom
    {
        std::cout << "\n[INFO] Test 4: Float GEMM, return_mem_ptr_atom..." << std::endl;
        self->mail(return_mem_ptr_atom{}, create_in_arg(h_A_f), create_in_arg(h_B_f), create_out_arg(h_C_f), m, n, k).send(float_gemm);
        self->receive(
            [&](int reply_id, mem_ptr<float> A, mem_ptr<float> B, mem_ptr<float> C) {
                command_runner<float> runner;
                auto data = runner.copy_to_host(C);
                verify_gemm(m, n, k, 1.0f, h_A_f, h_B_f, 0.0f, h_C_f, data);
            }
        );
    }

    self->send_exit(float_gemm, exit_reason::user_shutdown);
    self->send_exit(double_gemm, exit_reason::user_shutdown);
    manager::shutdown();
}

CAF_MAIN(id_block::cuda)
