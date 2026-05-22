#include <caf/all.hpp>
#include <caf/cuda/all.hpp>
#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <thread>
#include <algorithm>
#include <numeric>
#include "caf/actor_registry.hpp"
#include "caf/actorBLAS/gemv-actor/gemv-actor.hpp"

using namespace caf;
using namespace caf::cuda;

// Verify GEMV result: y = alpha * A * x + beta * y
// With A (all 3.0s) and x (all 4.0s), the dot product sum is (3.0 * 4.0 * n).
// We use alpha = 1.0 / n to normalize the output to exactly 12.0f as requested.
void verify_gemv_correctness(int m, int n, float alpha, float beta, 
                             const std::vector<float>& y_result) {
    float expected_val = alpha * (3.0f * 4.0f * n); 
    bool all_correct = true;
    for (size_t i = 0; i < y_result.size(); ++i) {
        if (std::abs(y_result[i] - expected_val) > 1e-4) {
            all_correct = false;
            std::cout << "[ERROR] Mismatch at index " << i 
                      << ": Expected " << expected_val 
                      << ", Got " << y_result[i] << std::endl;
            break;
        }
    }

    if (all_correct) {
        std::cout << "[SUCCESS] GEMV actor produced correct results." << std::endl;
        std::cout << "          Output vector is filled with: " << expected_val << std::endl;
    }
}

void caf_main(actor_system& sys) {
    // Initialize the manager with BLAS enabled to initialize cuBLAS handles
    manager_config config(true); 
    manager::init(sys, config);

    // Matrix dimensions 64x64 (> 32x32)
    int m = 64;
    int n = 64;
    
    // Scalar alpha set to 1/n to normalize the dot product sum (3*4*n) to 12.0
    float alpha = 1.0f / static_cast<float>(n);
    float beta = 0.0f;

    // Initialize host data
    std::vector<float> h_A(m * n, 3.0f);
    std::vector<float> h_x(n, 4.0f);
    std::vector<float> h_y(m, 0.0f);

    // Spawn the gemv_actor
    auto blas_actor = sys.spawn<gemv_actor>(1);

    // Prepare arguments using wrapper tags
    auto A_arg = create_in_arg(h_A);
    auto x_arg = create_in_arg(h_x);
    auto y_arg = create_out_arg(h_y);

    scoped_actor self{sys};

    // Test 1: Standard host-buffer based call
    {
        std::cout << "[INFO] Test 1: Testing gemv_actor with host-buffer arguments..." << std::endl;
        self->mail(A_arg, x_arg, y_arg, m, n, alpha, beta).send(blas_actor);
        self->receive(
            [&](int reply_id, int arg_index, std::vector<float> data) {
                if (arg_index == 2) { // index 2 corresponds to y_arg
                    verify_gemv_correctness(m, n, alpha, beta, data);
                }
            }
        );
        std::cout << "[INFO] Test 1 complete." << std::endl;
    }

    // Test 2: mem_ptr inputs
    {
        std::cout << "\n[INFO] Test 2: Testing gemv_actor with mem_ptr inputs..." << std::endl;
        command_runner<in<float>, in<float>, out<float>> setup_runner;
        // Manually transfer data to the GPU to get mem_ptr handles
        auto results = setup_runner.transfer_memory(0, 0, create_in_arg(h_A), create_in_arg(h_x), create_out_arg(h_y));
        auto A_ptr = std::get<0>(results);
        auto x_ptr = std::get<1>(results);
        auto y_ptr = std::get<2>(results);

        self->mail(A_ptr, x_ptr, y_ptr, m, n, alpha, beta).send(blas_actor);

        self->receive(
            [&](int reply_id, int arg_index, std::vector<float> data) {
                if (arg_index == 2) {
                    verify_gemv_correctness(m, n, alpha, beta, data);
                }
            }
        );
        std::cout << "[INFO] Test 2 complete." << std::endl;
    }

    // Test 3: Routing control (device/stream) + mem_ptr
    {
        std::cout << "\n[INFO] Test 3: Testing gemv_actor with specific device/stream and mem_ptr..." << std::endl;
        int device_num = 0;
        int stream_id = 42; 
        command_runner<in<float>, in<float>, out<float>> setup_runner;
        auto results = setup_runner.transfer_memory(device_num, stream_id, create_in_arg(h_A), create_in_arg(h_x), create_out_arg(h_y));
        
        self->mail(device_num, stream_id, std::get<0>(results), std::get<1>(results), std::get<2>(results), m, n, alpha, beta).send(blas_actor);

        self->receive(
            [&](int reply_id, int arg_index, std::vector<float> data) {
                if (arg_index == 2) {
                    verify_gemv_correctness(m, n, alpha, beta, data);
                }
            }
        );
        std::cout << "[INFO] Test 3 complete." << std::endl;
    }

    // Test 4: return_mem_ptr_atom (returning device handles)
    {
        std::cout << "\n[INFO] Test 4: Testing gemv_actor with return_mem_ptr_atom..." << std::endl;
        self->mail(return_mem_ptr_atom{}, A_arg, x_arg, y_arg, m, n, alpha, beta).send(blas_actor);

        // We expect two messages back: the data (mem_ptrs) and the signal (-1)
        self->receive(
            [&](int reply_id, mem_ptr<float> A, mem_ptr<float> x, mem_ptr<float> y) {
                command_runner<float> runner;
                auto host_y = runner.copy_to_host(y);
                verify_gemv_correctness(m, n, alpha, beta, host_y);
            }
        );
        std::cout << "[INFO] Test 4 complete." << std::endl;
    }

    self->send_exit(blas_actor, exit_reason::user_shutdown);
    manager::shutdown();
}
CAF_MAIN(id_block::cuda)
