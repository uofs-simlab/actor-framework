/**
 * This test file evaluates the correctness of spmm_actor for sparse matrix-matrix multiplication.
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
#include "caf/actorSPARSE/actorSPARSE.hpp"

using namespace caf;
using namespace caf::cuda;

void verify_spmm(const std::string& test_name, const std::vector<float>& actual, 
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

    // Matrix A (3x2): [ 1 0; 0 2; 3 4 ]
    // Matrix B (2x2): [ 5 6; 7 8 ]
    // Expected C (3x2): [ 5 6; 14 16; 43 50 ]
    int m = 3, n = 2, k = 2, nnz = 4;
    std::vector<float> h_B = {5.0f, 6.0f, 7.0f, 8.0f};
    std::vector<float> h_C_init(m * n, 0.0f);
    std::vector<float> expected = {5.0f, 6.0f, 14.0f, 16.0f, 43.0f, 50.0f};

    int device_num = 0;
    int stream_id = 10;

    auto spmm = sys.spawn<spmm_actor>(1);
    scoped_actor self{sys};

    // Test 1: CSR - Host Wrappers - Explicit Routing
    {
        std::cout << "[INFO] Test 1: SpMM CSR format, host wrappers, explicit routing..." << std::endl;
        std::vector<int> row_ptr = {0, 1, 2, 4};
        std::vector<int> col_ind = {0, 1, 0, 1};
        std::vector<float> values = {1, 2, 3, 4};

        self->mail(csr_atom{}, device_num, stream_id, 
                   create_in_arg(row_ptr), create_in_arg(col_ind), create_in_arg(values), 
                   create_in_arg(h_B), create_out_arg(h_C_init), 
                   m, n, k, nnz).send(spmm);

        self->receive(
            [&](int reply_id, int arg_index, std::vector<float> data) {
                verify_spmm("CSR Host Wrapper Routing", data, expected);
            }
        );
    }

    // Test 2: CSC - mem_ptr - Explicit Routing
    {
        std::cout << "\n[INFO] Test 2: SpMM CSC format, mem_ptr inputs, explicit routing..." << std::endl;
        std::vector<int> col_ptr = {0, 2, 4};
        std::vector<int> row_ind = {0, 2, 1, 2};
        std::vector<float> values = {1, 3, 2, 4};

        command_runner<in<int>, in<int>, in<float>, in<float>, out<float>> runner;
        auto results = runner.transfer_memory(device_num, stream_id, 
                                             create_in_arg(col_ptr), create_in_arg(row_ind), 
                                             create_in_arg(values), create_in_arg(h_B), 
                                             create_out_arg(h_C_init));

        self->mail(csc_atom{}, device_num, stream_id, 
                   std::get<0>(results), std::get<1>(results), std::get<2>(results), 
                   std::get<3>(results), std::get<4>(results), 
                   m, n, k, nnz).send(spmm);

        self->receive(
            [&](int reply_id, int arg_index, std::vector<float> data) {
                verify_spmm("CSC mem_ptr Routing", data, expected);
            }
        );
    }

    // Test 3: COO - mem_ptr - return_mem_ptr_atom - Default Routing
    {
        std::cout << "\n[INFO] Test 3: SpMM COO format, return_mem_ptr_atom, mem_ptr inputs..." << std::endl;
        std::vector<int> row_ind = {0, 1, 2, 2};
        std::vector<int> col_ind = {0, 1, 0, 1};
        std::vector<float> values = {1, 2, 3, 4};

        command_runner<in<int>, in<int>, in<float>, in<float>, out<float>> runner;
        auto results = runner.transfer_memory(device_num, stream_id, 
                                             create_in_arg(row_ind), create_in_arg(col_ind), 
                                             create_in_arg(values), create_in_arg(h_B), 
                                             create_out_arg(h_C_init));

        self->mail(return_mem_ptr_atom{}, coo_atom{}, 
                   std::get<0>(results), std::get<1>(results), std::get<2>(results), 
                   std::get<3>(results), std::get<4>(results), 
                   m, n, k, nnz).send(spmm);

        self->receive(
            [&](int reply_id, mem_ptr<int>, mem_ptr<int>, mem_ptr<float>, 
                mem_ptr<float>, mem_ptr<float> C_ptr) {
                command_runner<float> cr;
                auto host_C = cr.copy_to_host(C_ptr);
                verify_spmm("COO return_mem_ptr Routing", host_C, expected);
            }
        );
    }

    // Test 4: CSR - mem_ptr - Explicit Routing
    {
        std::cout << "\n[INFO] Test 4: SpMM CSR format, mem_ptr inputs, explicit routing..." << std::endl;
        std::vector<int> row_ptr = {0, 1, 2, 4};
        std::vector<int> col_ind = {0, 1, 0, 1};
        std::vector<float> values = {1, 2, 3, 4};

        command_runner<in<int>, in<int>, in<float>, in<float>, out<float>> runner;
        auto results = runner.transfer_memory(device_num, stream_id, 
                                             create_in_arg(row_ptr), create_in_arg(col_ind), 
                                             create_in_arg(values), create_in_arg(h_B), 
                                             create_out_arg(h_C_init));

        self->mail(csr_atom{}, device_num, stream_id, 
                   std::get<0>(results), std::get<1>(results), std::get<2>(results), 
                   std::get<3>(results), std::get<4>(results), 
                   m, n, k, nnz).send(spmm);

        self->receive(
            [&](int reply_id, int arg_index, std::vector<float> data) {
                verify_spmm("CSR mem_ptr Routing", data, expected);
            }
        );
    }


    // Test 5: CSR - Host Wrappers - Default Routing (Lottery Scheduler)
    {
        std::cout << "\n[INFO] Test 5: SpMM CSR format, host wrappers, default routing..." << std::endl;
        std::vector<int> row_ptr = {0, 1, 2, 4};
        std::vector<int> col_ind = {0, 1, 0, 1};
        std::vector<float> values = {1, 2, 3, 4};

        self->mail(csr_atom{}, create_in_arg(row_ptr), create_in_arg(col_ind), create_in_arg(values), 
                   create_in_arg(h_B), create_out_arg(h_C_init), 
                   m, n, k, nnz).send(spmm);

        self->receive(
            [&](int reply_id, int arg_index, std::vector<float> data) {
                verify_spmm("CSR Host Wrapper Default", data, expected);
            }
        );
    }

    // Test 4: CSR - Host Wrappers - Default Routing (Lottery Scheduler)
    {
        std::cout << "\n[INFO] Test 6: SpMM COO format, mem_ptr inputs, explicit routing..." << std::endl;
        std::vector<int> row_ind = {0, 1, 2, 2};
        std::vector<int> col_ind = {0, 1, 0, 1};
        std::vector<float> values = {1, 2, 3, 4};

        command_runner<in<int>, in<int>, in<float>, in<float>, out<float>> runner;
        auto results = runner.transfer_memory(device_num, stream_id, 
                                             create_in_arg(row_ind), create_in_arg(col_ind), 
                                             create_in_arg(values), create_in_arg(h_B), 
                                             create_out_arg(h_C_init));

        self->mail(coo_atom{}, device_num, stream_id, 
                   std::get<0>(results), std::get<1>(results), std::get<2>(results), 
                   std::get<3>(results), std::get<4>(results), 
                   m, n, k, nnz).send(spmm);

        self->receive(
            [&](int reply_id, int arg_index, std::vector<float> data) {
                verify_spmm("COO mem_ptr Routing", data, expected);
            }
        );
    }

    // Test 7: CSC - mem_ptr - return_mem_ptr_atom - Explicit Routing
    {
        std::cout << "\n[INFO] Test 7: SpMM CSC format, return_mem_ptr_atom, mem_ptr inputs, explicit routing..." << std::endl;
        std::vector<int> col_ptr = {0, 2, 4};
        std::vector<int> row_ind = {0, 2, 1, 2};
        std::vector<float> values = {1, 3, 2, 4};

        command_runner<in<int>, in<int>, in<float>, in<float>, out<float>> runner;
        auto results = runner.transfer_memory(device_num, stream_id, 
                                             create_in_arg(col_ptr), create_in_arg(row_ind), 
                                             create_in_arg(values), create_in_arg(h_B), 
                                             create_out_arg(h_C_init));

        self->mail(return_mem_ptr_atom{}, csc_atom{}, device_num, stream_id, 
                   std::get<0>(results), std::get<1>(results), std::get<2>(results), 
                   std::get<3>(results), std::get<4>(results), 
                   m, n, k, nnz).send(spmm);

        self->receive(
            [&](int reply_id, mem_ptr<int>, mem_ptr<int>, mem_ptr<float>, 
                mem_ptr<float>, mem_ptr<float> C_ptr) {
                command_runner<float> cr;
                auto host_C = cr.copy_to_host(C_ptr);
                verify_spmm("CSC return_mem_ptr Routing", host_C, expected);
            }
        );
    }

    // Test 8: COO - mem_ptr - return_mem_ptr_atom - Explicit Routing
    {
        std::cout << "\n[INFO] Test 8: SpMM COO format, return_mem_ptr_atom, mem_ptr inputs, explicit routing..." << std::endl;
        std::vector<int> row_ind = {0, 1, 2, 2};
        std::vector<int> col_ind = {0, 1, 0, 1};
        std::vector<float> values = {1, 2, 3, 4};

        command_runner<in<int>, in<int>, in<float>, in<float>, out<float>> runner;
        auto results = runner.transfer_memory(device_num, stream_id, 
                                             create_in_arg(row_ind), create_in_arg(col_ind), 
                                             create_in_arg(values), create_in_arg(h_B), 
                                             create_out_arg(h_C_init));

        self->mail(return_mem_ptr_atom{}, coo_atom{}, device_num, stream_id, 
                   std::get<0>(results), std::get<1>(results), std::get<2>(results), 
                   std::get<3>(results), std::get<4>(results), 
                   m, n, k, nnz).send(spmm);

        self->receive(
            [&](int reply_id, mem_ptr<int>, mem_ptr<int>, mem_ptr<float>, 
                mem_ptr<float>, mem_ptr<float> C_ptr) {
                command_runner<float> cr;
                auto host_C = cr.copy_to_host(C_ptr);
                verify_spmm("COO return_mem_ptr Routing", host_C, expected);
            }
        );
    }

    // Test 9: CSR - mem_ptr - return_mem_ptr_atom - Explicit Routing
    {
        std::cout << "\n[INFO] Test 9: SpMM CSR format, return_mem_ptr_atom, mem_ptr inputs, explicit routing..." << std::endl;
        std::vector<int> row_ptr = {0, 1, 2, 4};
        std::vector<int> col_ind = {0, 1, 0, 1};
        std::vector<float> values = {1, 2, 3, 4};

        command_runner<in<int>, in<int>, in<float>, in<float>, out<float>> runner;
        auto results = runner.transfer_memory(device_num, stream_id, 
                                             create_in_arg(row_ptr), create_in_arg(col_ind), 
                                             create_in_arg(values), create_in_arg(h_B), 
                                             create_out_arg(h_C_init));

        self->mail(return_mem_ptr_atom{}, csr_atom{}, device_num, stream_id, 
                   std::get<0>(results), std::get<1>(results), std::get<2>(results), 
                   std::get<3>(results), std::get<4>(results), 
                   m, n, k, nnz).send(spmm);

        self->receive(
            [&](int reply_id, mem_ptr<int>, mem_ptr<int>, mem_ptr<float>, 
                mem_ptr<float>, mem_ptr<float> C_ptr) {
                command_runner<float> cr;
                auto host_C = cr.copy_to_host(C_ptr);
                verify_spmm("CSR return_mem_ptr Routing", host_C, expected);
            }
        );
    }
    self->send_exit(spmm, exit_reason::user_shutdown);
    manager::shutdown();
}
CAF_MAIN(id_block::cuda)
