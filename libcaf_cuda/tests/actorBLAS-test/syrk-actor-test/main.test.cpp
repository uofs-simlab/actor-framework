#include <caf/all.hpp>
#include <caf/cuda/all.hpp>
#include <iostream>
#include <vector>
#include "caf/actorBLAS/syrk-actor/syrk-actor.hpp"

using namespace caf;
using namespace caf::cuda;

void verify_syrk_correctness(int n, int k, float alpha, float beta, 
                             const std::vector<float>& C_result) {
    // Expected value for C = alpha * (A * A^T) + beta * C_init
    // Since A is all 1.0s (n x k), each element of A*A^T is k.
    // With alpha=1 and beta=0, expected is exactly k.
    // Note: cuBLAS SYRK only updates the lower triangle by default in our implementation.
    float expected_val = alpha * static_cast<float>(k); 
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
            },
            [&](int reply_id, int signal) {
                if (signal == -1) {
                    std::cout << "[INFO] Test 1 complete." << std::endl;
                }
            }
        );
    }

    self->send_exit(blas_actor, exit_reason::user_shutdown);
    manager::shutdown();
}

CAF_MAIN(id_block::cuda)