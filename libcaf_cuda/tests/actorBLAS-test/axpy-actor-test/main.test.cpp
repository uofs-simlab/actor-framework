#include <caf/all.hpp>
#include <caf/cuda/all.hpp>
#include <iostream>
#include <vector>
#include "caf/actorBLAS/axpy-actor/axpy-actor.hpp"

using namespace caf;
using namespace caf::cuda;

void verify_axpy_correctness(int n, float alpha, 
                             const std::vector<float>& x,
                             const std::vector<float>& y_initial,
                             const std::vector<float>& y_result) {
    bool all_correct = true;
    for (int i = 0; i < n; ++i) {
        float expected = alpha * x[i] + y_initial[i];
        if (std::abs(y_result[i] - expected) > 1e-4) {
            all_correct = false;
            std::cout << "[ERROR] Mismatch at index " << i << ": "
                      << "Expected " << expected << ", Got " << y_result[i] << std::endl;
            break;
        }
    }

    if (all_correct) {
        std::cout << "[SUCCESS] AXPY actor produced correct results." << std::endl;
    }
}

void caf_main(actor_system& sys) {
    manager_config config(true); 
    manager::init(sys, config);

    int n = 1024;
    float alpha = 2.0f;

    std::vector<float> h_x(n, 1.0f);
    std::vector<float> h_y(n, 3.0f);

    auto blas_actor = sys.spawn<axpy_actor>(1);

    auto x_arg = create_in_arg(h_x);
    auto y_arg = create_in_out_arg(h_y);

    scoped_actor self{sys};

    // Test 1: Standard host-buffer based call
    {
        std::cout << "[INFO] Test 1: Testing axpy_actor with host-buffer arguments..." << std::endl;
        self->mail(x_arg, y_arg, n, alpha).send(blas_actor);
        self->receive(
            [&](int reply_id, int arg_index, std::vector<float> data) {
                if (arg_index == 1) { // index 1 corresponds to y_arg
                    verify_axpy_correctness(n, alpha, h_x, h_y, data);
                }
            }
        );
        std::cout << "[INFO] Test 1 complete." << std::endl;
    }

    // Test 2: mem_ptr inputs
    {
        std::cout << "\n[INFO] Test 2: Testing axpy_actor with mem_ptr inputs..." << std::endl;
        command_runner<in<float>, in_out<float>> setup_runner;
        auto results = setup_runner.transfer_memory(0, 0, create_in_arg(h_x), create_in_out_arg(h_y));
        auto x_ptr = std::get<0>(results);
        auto y_ptr = std::get<1>(results);

        self->mail(x_ptr, y_ptr, n, alpha).send(blas_actor);

        self->receive(
            [&](int reply_id, int arg_index, std::vector<float> data) {
                if (arg_index == 1) {
                    verify_axpy_correctness(n, alpha, h_x, h_y, data);
                }
            }
        );
        std::cout << "[INFO] Test 2 complete." << std::endl;
    }

    // Test 3: return_mem_ptr_atom
    {
        std::cout << "\n[INFO] Test 3: Testing axpy_actor with return_mem_ptr_atom..." << std::endl;
        self->mail(return_mem_ptr_atom{}, x_arg, y_arg, n, alpha).send(blas_actor);

        self->receive(
            [&](int reply_id, mem_ptr<float> x, mem_ptr<float> y) {
                command_runner<float> runner;
                auto host_y = runner.copy_to_host(y);
                verify_axpy_correctness(n, alpha, h_x, h_y, host_y);
            }
        );
        std::cout << "[INFO] Test 3 complete." << std::endl;
    }

    self->send_exit(blas_actor, exit_reason::user_shutdown);
    manager::shutdown();
}

CAF_MAIN(id_block::cuda)
