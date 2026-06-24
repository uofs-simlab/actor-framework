#include <caf/all.hpp>
#include <caf/cuda/all.hpp>
#include <iostream>
#include <vector>
#include "caf/actorBLAS/dot-actor/dot-actor.hpp"

using namespace caf;
using namespace caf::cuda;

void verify_dot_correctness(float expected, const std::vector<float>& result) {
    if (result.empty()) {
        std::cout << "[ERROR] Result vector is empty." << std::endl;
        return;
    }

    if (std::abs(result[0] - expected) > 1e-4) {
        std::cout << "[ERROR] Dot product mismatch: "
                  << "Expected " << expected << ", Got " << result[0] << std::endl;
    } else {
        std::cout << "[SUCCESS] Dot actor produced correct results." << std::endl;
    }
}

void caf_main(actor_system& sys) {
    manager_config config(true); 
    manager::init(sys, config);

    int n = 1024;
    std::vector<float> h_x(n, 1.0f);
    std::vector<float> h_y(n, 2.0f);
    std::vector<float> h_res(1, 0.0f);

    float expected = static_cast<float>(n) * 1.0f * 2.0f;

    auto blas_actor = sys.spawn<dot_actor>(1);

    auto x_arg = create_in_arg(h_x);
    auto y_arg = create_in_arg(h_y);
    auto res_arg = create_out_arg(h_res);

    scoped_actor self{sys};

    // Test 1: Standard host-buffer based call
    {
        std::cout << "[INFO] Test 1: Testing dot_actor with host-buffer arguments..." << std::endl;
        self->mail(x_arg, y_arg, res_arg, n).send(blas_actor);
        self->receive(
            [&](int reply_id, float data) {
                verify_dot_correctness(expected, {data});
            }
        );
        std::cout << "[INFO] Test 1 complete." << std::endl;
    }

    // Test 2: mem_ptr inputs
    {
        std::cout << "\n[INFO] Test 2: Testing dot_actor with mem_ptr inputs..." << std::endl;
        command_runner<in<float>, in<float>, out<float>> setup_runner;
        // Manually transfer data to the GPU to get mem_ptr handles
        auto results = setup_runner.transfer_memory(0, 0, create_in_arg(h_x), create_in_arg(h_y), create_out_arg(h_res));
        auto x_ptr = std::get<0>(results);
        auto y_ptr = std::get<1>(results);
        auto res_ptr = std::get<2>(results);

        self->mail(x_ptr, y_ptr, res_ptr, n).send(blas_actor);

        self->receive(
            [&](int reply_id, float data) {
                verify_dot_correctness(expected, {data});
            }
        );
        std::cout << "[INFO] Test 2 complete." << std::endl;
    }

    // Test 3: Routing control (device/stream) + mem_ptr
    {
        std::cout << "\n[INFO] Test 3: Testing dot_actor with specific device/stream and mem_ptr..." << std::endl;
        int device_num = 0;
        int stream_id = 42; 
        command_runner<in<float>, in<float>, out<float>> setup_runner;
        auto results = setup_runner.transfer_memory(device_num, stream_id, create_in_arg(h_x), create_in_arg(h_y), create_out_arg(h_res));
        
        self->mail(device_num, stream_id, std::get<0>(results), std::get<1>(results), std::get<2>(results), n).send(blas_actor);

        self->receive(
            [&](int reply_id, float data) {
                verify_dot_correctness(expected, {data});
            }
        );
        std::cout << "[INFO] Test 3 complete." << std::endl;
    }

    // Test 4: return_mem_ptr_atom (returning device handles)
    {
        std::cout << "\n[INFO] Test 4: Testing dot_actor with return_mem_ptr_atom..." << std::endl;
        self->mail(return_mem_ptr_atom{}, x_arg, y_arg, res_arg, n).send(blas_actor); // This line will now work

        // We expect the data handles (mem_ptrs) back
        self->receive(
            [&](int reply_id, mem_ptr<float> x, mem_ptr<float> y, mem_ptr<float> res) {
                command_runner<float> runner;
                // Since dot is async, we copy res back to host to verify
                auto host_res = runner.copy_to_host(res);
                verify_dot_correctness(expected, host_res);
            }
        );
        std::cout << "[INFO] Test 4 complete." << std::endl;
    }

    self->send_exit(blas_actor, exit_reason::user_shutdown);
    manager::shutdown();
}

CAF_MAIN(id_block::cuda)
