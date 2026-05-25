/**
 * This test file evaluates the correctness of spmv_actor for sparse matrix-vector multiplication.
 * It covers:
 * - CSR, CSC, and COO formats.
 * - Input types: host wrappers (in/out) and device handles (mem_ptr).
 * - Reply modes: data result and device handle return (return_mem_ptr_atom).
 * - Explicit routing with device_num and stream_id.
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
#include "caf/actor_registry.hpp"
#include "caf/actorSPARSE/spmv-actor/spmv-actor.hpp"

using namespace caf;
using namespace caf::cuda;

void verify_spmv(const std::string& test_name, const std::vector<float>& actual, 
                 const std::vector<float>& expected) {
    if (actual.size() != expected.size()) {
        std::cout << "[ERROR] " << test_name << " failed: Size mismatch (got " 
                  << actual.size() << ", expected " << expected.size() << ")" << std::endl;
        return;
    }
    bool all_correct = true;
    for (size_t i = 0; i < actual.size(); ++i) {
        if (std::abs(actual[i] - expected[i]) > 1e-4) {
            all_correct = false;
            std::cout << "[ERROR] " << test_name << " mismatch at index " << i 
                      << ": Expected " << expected[i] 
                      << ", Got " << actual[i] << std::endl;
            break;
        }
    }
    if (all_correct) {
        std::cout << "[SUCCESS] " << test_name << " passed." << std::endl;
    }
}

void caf_main(actor_system& sys) {
    manager::init(sys, manager_config(true, true)); // Enable cuBLAS and cuSPARSE

    // Matrix A (3x3): [ 1 0 2; 0 0 3; 4 5 6 ]
    // Vector x: [1, 2, 3]
    // Expected y: [7, 9, 32]
    int m = 3, n = 3, nnz = 6;
    std::vector<float> h_x = {1.0f, 2.0f, 3.0f};
    std::vector<float> h_y_init = {0.0f, 0.0f, 0.0f};
    std::vector<float> expected = {7.0f, 9.0f, 32.0f};

    int device_num = 0;
    int stream_id = 10;

    auto spmv = sys.spawn<spmv_actor>(1);
    scoped_actor self{sys};

    // Test 1: CSR - Host Wrappers - Explicit Routing
    {
        std::cout << "[INFO] Test 1: CSR format, host wrappers, explicit routing (dev=" << device_num << ", stream=" << stream_id << ")..." << std::endl;
        std::vector<int> row_ptr = {0, 2, 3, 6};
        std::vector<int> col_ind = {0, 2, 2, 0, 1, 2};
        std::vector<float> values = {1, 2, 3, 4, 5, 6};

        self->mail(csr_atom{}, device_num, stream_id, 
                   create_in_arg(row_ptr), create_in_arg(col_ind), create_in_arg(values), 
                   create_in_arg(h_x), create_out_arg(h_y_init), 
                   m, n, nnz).send(spmv);

        self->receive(
            [&](int reply_id, int arg_index, std::vector<float> data) {
                verify_spmv("CSR Host Wrapper Routing", data, expected);
            }
        );
    }

    // Test 2: CSC - mem_ptr - Explicit Routing
    {
        std::cout << "\n[INFO] Test 2: CSC format, mem_ptr inputs, explicit routing..." << std::endl;
        std::vector<int> col_ptr = {0, 2, 3, 6};
        std::vector<int> row_ind = {0, 2, 2, 0, 1, 2};
        std::vector<float> values = {1, 4, 5, 2, 3, 6};

        command_runner<in<int>, in<int>, in<float>, in<float>, out<float>> runner;
        auto results = runner.transfer_memory(device_num, stream_id, 
                                             create_in_arg(col_ptr), create_in_arg(row_ind), 
                                             create_in_arg(values), create_in_arg(h_x), 
                                             create_out_arg(h_y_init));

        self->mail(csc_atom{}, device_num, stream_id, 
                   std::get<0>(results), std::get<1>(results), std::get<2>(results), 
                   std::get<3>(results), std::get<4>(results), 
                   m, n, nnz).send(spmv);

        self->receive(
            [&](int reply_id, int arg_index, std::vector<float> data) {
                verify_spmv("CSC mem_ptr Routing", data, expected);
            }
        );
    }

    // Test 3: COO - mem_ptr - return_mem_ptr_atom - Explicit Routing
    {
        std::cout << "\n[INFO] Test 3: COO format, return_mem_ptr_atom, mem_ptr inputs, explicit routing..." << std::endl;
        std::vector<int> row_ind = {0, 0, 1, 2, 2, 2};
        std::vector<int> col_ind = {0, 2, 2, 0, 1, 2};
        std::vector<float> values = {1, 2, 3, 4, 5, 6};

        command_runner<in<int>, in<int>, in<float>, in<float>, out<float>> runner;
        auto results = runner.transfer_memory(device_num, stream_id, 
                                             create_in_arg(row_ind), create_in_arg(col_ind), 
                                             create_in_arg(values), create_in_arg(h_x), 
                                             create_out_arg(h_y_init));

        self->mail(return_mem_ptr_atom{}, coo_atom{}, device_num, stream_id, 
                   std::get<0>(results), std::get<1>(results), std::get<2>(results), 
                   std::get<3>(results), std::get<4>(results), 
                   m, n, nnz, 1.0f, 0.0f).send(spmv);

        self->receive(
            [&](int reply_id, mem_ptr<int>, mem_ptr<int>, mem_ptr<float>, 
                mem_ptr<float>, mem_ptr<float> y_ptr) {
                command_runner<float> cr;
                auto host_y = cr.copy_to_host(y_ptr);
                verify_spmv("COO return_mem_ptr Routing", host_y, expected);
            }
        );
    }

    // Test 4: CSR - Host Wrappers - Default Routing (Lottery Scheduler)
    {
        std::cout << "\n[INFO] Test 4: CSR format, host wrappers, default routing..." << std::endl;
        std::vector<int> row_ptr = {0, 2, 3, 6};
        std::vector<int> col_ind = {0, 2, 2, 0, 1, 2};
        std::vector<float> values = {1, 2, 3, 4, 5, 6};

        self->mail(csr_atom{}, create_in_arg(row_ptr), create_in_arg(col_ind), create_in_arg(values), 
                   create_in_arg(h_x), create_out_arg(h_y_init), 
                   m, n, nnz).send(spmv);

        self->receive(
            [&](int reply_id, int arg_index, std::vector<float> data) {
                verify_spmv("CSR Host Wrapper Default", data, expected);
            }
        );
    }

    self->send_exit(spmv, exit_reason::user_shutdown);
    manager::shutdown();
}
CAF_MAIN(id_block::cuda)
