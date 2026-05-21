#include <caf/all.hpp>
#include <caf/cuda/all.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include "caf/actorBLAS/nrm2-actor/nrm2-actor.hpp"

using namespace caf;
using namespace caf::cuda;

void verify_nrm2_correctness(int n, const std::vector<float>& x, float result) {
    double sum = 0;
    for (float val : x) {
        sum += (double)val * val;
    }
    float expected = (float)std::sqrt(sum);

    if (std::abs(result - expected) > 1e-4) {
        std::cout << "[ERROR] Mismatch: Expected " << expected << ", Got " << result << std::endl;
    } else {
        std::cout << "[SUCCESS] NRM2 actor produced correct results: " << result << std::endl;
    }
}

void caf_main(actor_system& sys) {
    manager_config config(true); 
    manager::init(sys, config);

    int n = 1024;
    std::vector<float> h_x(n, 1.0f);

    auto blas_actor = sys.spawn<nrm2_actor>(1);
    auto x_arg = create_in_arg(h_x);
    auto res_arg = create_out_arg_with_size<float>(1);

    scoped_actor self{sys};

    // Test 1: Standard host-buffer based call
    {
        std::cout << "[INFO] Test 1: Testing nrm2_actor with host-buffer arguments..." << std::endl;
        self->mail(x_arg, res_arg, n).send(blas_actor);
        self->receive(
            [&](int reply_id, int arg_index, std::vector<float> data) {
                if (arg_index == 1) { // index 1 corresponds to res_arg
                    verify_nrm2_correctness(n, h_x, data[0]);
                }
            }
        );
        std::cout << "[INFO] Test 1 complete." << std::endl;
    }

    // Test 2: mem_ptr inputs
    {
        std::cout << "\n[INFO] Test 2: Testing nrm2_actor with mem_ptr inputs..." << std::endl;
        command_runner<in<float>, out<float>> setup_runner;
        auto results = setup_runner.transfer_memory(0, 0, create_in_arg(h_x), create_out_arg_with_size<float>(1));
        auto x_ptr = std::get<0>(results);
        auto res_ptr = std::get<1>(results);

        self->mail(x_ptr, res_ptr, n).send(blas_actor);

        self->receive(
            [&](int reply_id, int arg_index, std::vector<float> data) {
                if (arg_index == 1) {
                    verify_nrm2_correctness(n, h_x, data[0]);
                }
            }
        );
        std::cout << "[INFO] Test 2 complete." << std::endl;
    }

    // Test 3: return_mem_ptr_atom
    {
        std::cout << "\n[INFO] Test 3: Testing nrm2_actor with return_mem_ptr_atom..." << std::endl;
        self->mail(return_mem_ptr_atom{}, x_arg, res_arg, n).send(blas_actor);

        self->receive(
            [&](int reply_id, mem_ptr<float> x, mem_ptr<float> res) {
                command_runner<float> runner;
                auto host_res = runner.copy_to_host(res);
                verify_nrm2_correctness(n, h_x, host_res[0]);
            }
        );
        std::cout << "[INFO] Test 3 complete." << std::endl;
    }

    self->send_exit(blas_actor, exit_reason::user_shutdown);
    manager::shutdown();
}

CAF_MAIN(id_block::cuda)

