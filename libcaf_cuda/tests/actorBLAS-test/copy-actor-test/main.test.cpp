#include <caf/all.hpp>
#include <caf/cuda/all.hpp>
#include <iostream>
#include <vector>
#include "caf/actorBLAS/copy-actor/copy-actor.hpp"

using namespace caf;
using namespace caf::cuda;

void verify_copy_correctness(int n, const std::vector<float>& source, const std::vector<float>& result) {
    bool all_correct = true;
    for (int i = 0; i < n; ++i) {
        if (std::abs(source[i] - result[i]) > 1e-4) {
            all_correct = false;
            std::cout << "[ERROR] Mismatch at index " << i << ": "
                      << "Expected " << source[i] << ", Got " << result[i] << std::endl;
            break;
        }
    }

    if (all_correct) {
        std::cout << "[SUCCESS] Copy actor produced correct results." << std::endl;
    }
}

void caf_main(actor_system& sys) {
    manager_config config(true); 
    manager::init(sys, config);

    int n = 1024;
    std::vector<float> h_x(n);
    for (int i = 0; i < n; ++i) {
        h_x[i] = static_cast<float>(i);
    }
    std::vector<float> h_y(n, 0.0f);

    auto blas_actor = sys.spawn<copy_actor>(1);

    auto x_arg = create_in_arg(h_x);
    auto y_arg = create_out_arg(h_y);

    scoped_actor self{sys};

    // Test 1: Standard host-buffer based call
    {
        std::cout << "[INFO] Test 1: Testing copy_actor with host-buffer arguments..." << std::endl;
        self->mail(x_arg, y_arg, n).send(blas_actor);
        self->receive(
            [&](int reply_id, int arg_index, std::vector<float> data) {
                if (arg_index == 1) { // index 1 corresponds to y_arg
                    verify_copy_correctness(n, h_x, data);
                }
            }
        );
        std::cout << "[INFO] Test 1 complete." << std::endl;
    }

    // Test 2: mem_ptr inputs
    {
        std::cout << "\n[INFO] Test 2: Testing copy_actor with mem_ptr inputs..." << std::endl;
        command_runner<in<float>, out<float>> setup_runner;
        // Manually transfer data to the GPU to get mem_ptr handles
        auto results = setup_runner.transfer_memory(0, 0, create_in_arg(h_x), create_out_arg(h_y));
        auto x_ptr = std::get<0>(results);
        auto y_ptr = std::get<1>(results);

        self->mail(x_ptr, y_ptr, n).send(blas_actor);

        self->receive(
            [&](int reply_id, int arg_index, std::vector<float> data) {
                if (arg_index == 1) {
                    verify_copy_correctness(n, h_x, data);
                }
            }
        );
        std::cout << "[INFO] Test 2 complete." << std::endl;
    }

    // Test 3: Routing control (device/stream) + mem_ptr
    {
        std::cout << "\n[INFO] Test 3: Testing copy_actor with specific device/stream and mem_ptr..." << std::endl;
        int device_num = 0;
        int stream_id = 42; 
        command_runner<in<float>, out<float>> setup_runner;
        auto results = setup_runner.transfer_memory(device_num, stream_id, create_in_arg(h_x), create_out_arg(h_y));
        
        self->mail(device_num, stream_id, std::get<0>(results), std::get<1>(results), n).send(blas_actor);

        self->receive(
            [&](int reply_id, int arg_index, std::vector<float> data) {
                if (arg_index == 1) {
                    verify_copy_correctness(n, h_x, data);
                }
            }
        );
        std::cout << "[INFO] Test 3 complete." << std::endl;
    }

    // Test 4: return_mem_ptr_atom (returning device handles)
    {
        std::cout << "\n[INFO] Test 4: Testing copy_actor with return_mem_ptr_atom..." << std::endl;
        int device_num = 0;
        int stream_id = 42;
        self->mail(return_mem_ptr_atom{}, device_num, stream_id, x_arg, y_arg, n).send(blas_actor);

        // We expect the data handles (mem_ptrs) back
        self->receive(
            [&](int reply_id, mem_ptr<float> x, mem_ptr<float> y) {
                command_runner<float> runner;
                // Since the operation is async, copy the returned device pointer back to host to verify
                auto host_y = runner.copy_to_host(y);
                verify_copy_correctness(n, h_x, host_y);
            }
        );
        std::cout << "[INFO] Test 4 complete." << std::endl;
    }

    self->send_exit(blas_actor, exit_reason::user_shutdown);
    manager::shutdown();
}

CAF_MAIN(id_block::cuda)
