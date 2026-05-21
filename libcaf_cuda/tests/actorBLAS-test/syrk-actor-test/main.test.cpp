#include <caf/all.hpp>
#include <caf/cuda/all.hpp>
#include <iostream>
#include <vector>
#include "caf/actorBLAS/syrk-actor/syrk-actor.hpp"

using namespace caf;
using namespace caf::cuda;

void verify_syrk_correctness(int n, int k, float alpha, float beta, 
                             const std::vector<float>& C_result, float initial_c_val = 0.0f) {
    // Expected value for C = alpha * (A * A^T) + beta * C_init
    // Since A is all 1.0s (n x k), each element of A*A^T is k.
    // expected = alpha * (k) + beta * initial_c_val
    float expected_val = alpha * static_cast<float>(k) + beta * initial_c_val; 
    bool all_correct = true;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j <= i; ++j) { // Check lower triangle
            float val = C_result[i * n + j];
            if (std::abs(val - expected_val) > 1e-4) {
                all_correct = false;
                std::cout << "[ERROR] Mismatch at index (" << i << "," << j << "): "
                          << "Expected " << expected_val << ", Got " << val << std::endl;
                break;
            }
        }
    }

    if (all_correct) {
        std::cout << "[SUCCESS] SYRK actor produced correct lower triangle results." << std::endl;
    }
}

void caf_main(actor_system& sys) {
    manager_config config(true); 
    manager::init(sys, config);

    int n = 32;
    int k = 64;
    float alpha = 1.0f;
    float beta = 0.0f;

    std::vector<float> h_A(n * k, 1.0f);
    std::vector<float> h_C(n * n, 0.0f);

    auto blas_actor = sys.spawn<syrk_actor>(1);

    auto A_arg = create_in_arg(h_A);
    auto C_arg = create_in_out_arg(h_C);

    scoped_actor self{sys};

    // Test 1: Standard host-buffer based call
    {
        std::cout << "[INFO] Test 1: Testing syrk_actor with host-buffer arguments..." << std::endl;
        self->mail(A_arg, C_arg, n, k, alpha, beta).send(blas_actor);
        self->receive(
            [&](int reply_id, int arg_index, std::vector<float> data) {
                if (arg_index == 1) { // index 1 corresponds to C_arg
                    verify_syrk_correctness(n, k, alpha, beta, data);
                }
            }
        );
        std::cout << "[INFO] Test 1 complete." << std::endl;
    }

    // Test 2: mem_ptr inputs
    {
        std::cout << "\n[INFO] Test 2: Testing syrk_actor with mem_ptr inputs..." << std::endl;
        command_runner<in<float>, in_out<float>> setup_runner;
        // Manually transfer data to the GPU to get mem_ptr handles
        auto results = setup_runner.transfer_memory(0, 0, create_in_arg(h_A), create_in_out_arg(h_C));
        auto A_ptr = std::get<0>(results);
        auto C_ptr = std::get<1>(results);

        self->mail(A_ptr, C_ptr, n, k, alpha, beta).send(blas_actor);

        self->receive(
            [&](int reply_id, int arg_index, std::vector<float> data) {
                if (arg_index == 1) {
                    verify_syrk_correctness(n, k, alpha, beta, data);
                }
            }
        );
        std::cout << "[INFO] Test 2 complete." << std::endl;
    }

    // Test 3: Routing control (device/stream) + mem_ptr
    {
        std::cout << "\n[INFO] Test 3: Testing syrk_actor with specific device/stream and mem_ptr..." << std::endl;
        int device_num = 0;
        int stream_id = 42; 
        command_runner<in<float>, in_out<float>> setup_runner;
        auto results = setup_runner.transfer_memory(device_num, stream_id, create_in_arg(h_A), create_in_out_arg(h_C));
        
        self->mail(device_num, stream_id, std::get<0>(results), std::get<1>(results), n, k, alpha, beta).send(blas_actor);

        self->receive(
            [&](int reply_id, int arg_index, std::vector<float> data) {
                if (arg_index == 1) {
                    verify_syrk_correctness(n, k, alpha, beta, data);
                }
            }
        );
        std::cout << "[INFO] Test 3 complete." << std::endl;
    }

    // Test 4: return_mem_ptr_atom (returning device handles)
    {
        std::cout << "\n[INFO] Test 4: Testing syrk_actor with return_mem_ptr_atom..." << std::endl;
        self->mail(return_mem_ptr_atom{}, A_arg, C_arg, n, k, alpha, beta).send(blas_actor);

        // We expect two messages back: the data handles (mem_ptrs) and the signal (-1)
        self->receive(
            [&](int reply_id, mem_ptr<float> A, mem_ptr<float> C) {
                command_runner<float> runner;
                // Since syrk is async, we copy C back to host to verify
                auto host_C = runner.copy_to_host(C);
                verify_syrk_correctness(n, k, alpha, beta, host_C);
            }
        );
        std::cout << "[INFO] Test 4 complete." << std::endl;
    }

    // Test 5: Accumulation test (beta != 0)
    {
        std::cout << "\n[INFO] Test 5: Testing syrk_actor with accumulation (beta=1.0)..." << std::endl;
        float beta_accum = 1.0f;
        std::vector<float> h_C_ones(n * n, 1.0f); // initial C is all 1.0s
        auto C_accum_arg = create_in_out_arg(h_C_ones);

        self->mail(A_arg, C_accum_arg, n, k, alpha, beta_accum).send(blas_actor);

        self->receive(
            [&](int reply_id, int arg_index, std::vector<float> data) {
                if (arg_index == 1) {
                    // Expected: alpha*k + beta*1.0 = 1.0*64 + 1.0*1.0 = 65.0
                    verify_syrk_correctness(n, k, alpha, beta_accum, data, 1.0f);
                }
            }
        );
        std::cout << "[INFO] Test 5 complete." << std::endl;
    }

    self->send_exit(blas_actor, exit_reason::user_shutdown);
    manager::shutdown();
}

CAF_MAIN(id_block::cuda)
