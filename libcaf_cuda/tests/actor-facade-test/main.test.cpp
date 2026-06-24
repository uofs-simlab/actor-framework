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
using namespace caf;
using namespace std::chrono_literals;

// CPU Matrix Multiplication for correctness verification
void verify_mmul(const std::vector<int>& a, const std::vector<int>& b, 
                 const std::vector<int>& c, int N) {
    std::vector<int> expected(N * N, 0);
    for (int i = 0; i < N; ++i) {
        for (int k = 0; k < N; ++k) {
            int aik = a[i * N + k];
            for (int j = 0; j < N; ++j) {
                expected[i * N + j] += aik * b[k * N + j];
            }
        }
    }
    bool correct = true;
    for (size_t i = 0; i < expected.size(); ++i) {
        if (c[i] != expected[i]) {
            correct = false;
            break;
        }
    }
    if (correct) {
        std::cout << "[SUCCESS] Matrix multiplication result is correct." << std::endl;
    } else {
        std::cout << "[FAILURE] Matrix multiplication result mismatch!" << std::endl;
    }
}

// State for the test manager actor
struct test_actor_state {
    std::vector<int> h_a;
    std::vector<int> h_b;
    std::vector<int> h_c;
    int N;
    std::chrono::steady_clock::time_point start_time;
};

// Actor behavior to test the actor_facade
caf::behavior mmul_facade_test(caf::stateful_actor<test_actor_state>* self, 
                               caf::actor facade, int N) {
    self->state().N = N;
    // Initialize matrices with some values
    self->state().h_a.assign(N * N, 1);
    self->state().h_b.assign(N * N, 2);
    self->state().h_c.resize(N * N);

    // Wrap host buffers in tagged arguments
    auto arg1 = caf::cuda::create_in_arg(self->state().h_a);
    auto arg2 = caf::cuda::create_in_arg(self->state().h_b);
    auto arg3 = caf::cuda::create_out_arg_with_size<int>(N * N);
    auto arg4 = caf::cuda::create_in_arg(N);

    std::cout << "[INFO] Launching Basic actor_facade test..." << std::endl;
    self->state().start_time = std::chrono::steady_clock::now();
    
    // Basic call: Copy everything back by default
    self->mail(arg1, arg2, arg3, arg4).send(facade);

    return {
        [=](int r_id, int index, std::vector<int> data) {
            if (index == 2) {
                self->state().h_c = std::move(data);
                auto end_time = std::chrono::steady_clock::now();
                std::chrono::duration<double> elapsed = end_time - self->state().start_time;
                std::cout << "[BASIC] Latency: " << elapsed.count() << "s" << std::endl;
                verify_mmul(self->state().h_a, self->state().h_b, self->state().h_c, self->state().N);
                self->quit();
            }
        }
        // No longer expecting index == -1 completion signal
    };
}

// New Separate Actor for Advanced Testing (Stream, Device, Selective Index)
caf::behavior mmul_advanced_facade_test(caf::stateful_actor<test_actor_state>* self, 
                                        caf::actor facade, int N,
                                        int device_num, int stream_id,
                                        std::vector<int> output_indices) {
    self->state().N = N;
    self->state().h_a.assign(N * N, 3); // Use different values to ensure fresh run
    self->state().h_b.assign(N * N, 4);
    self->state().h_c.resize(N * N);

    auto arg1 = caf::cuda::create_in_arg(self->state().h_a);
    auto arg2 = caf::cuda::create_in_arg(self->state().h_b);
    auto arg3 = caf::cuda::create_out_arg_with_size<int>(N * N);
    auto arg4 = caf::cuda::create_in_arg(N);

    std::cout << "[INFO] Launching Advanced actor_facade test:" << std::endl;
    std::cout << "       Device: " << device_num << ", Stream: " << stream_id 
              << ", Indices: { ";
    for(int i : output_indices) std::cout << i << " ";
    std::cout << "}" << std::endl;
              
    self->state().start_time = std::chrono::steady_clock::now();
    
    // Advanced call: Use specific routing and selective index copy-back
    self->mail(device_num, stream_id, output_indices, arg1, arg2, arg3, arg4).send(facade);

    return {
        [=](int r_id, int index, std::vector<int> data) {
            if (index == 2) {
                self->state().h_c = std::move(data);

                // Verify that we ONLY get index 2, as requested in output_indices
                bool requested = std::find(output_indices.begin(), output_indices.end(), index) != output_indices.end();
                if (!requested) {
                    std::cout << "[ERROR] Received unrequested index: " << index << std::endl;
                }

                auto end_time = std::chrono::steady_clock::now();
                std::chrono::duration<double> elapsed = end_time - self->state().start_time;

                std::cout << "===== Advanced Performance Result =====" << std::endl;
                std::cout << "Round-trip Latency: " << elapsed.count() << " seconds" << std::endl;
                verify_mmul(self->state().h_a, self->state().h_b, self->state().h_c, self->state().N);

                self->send_exit(facade, exit_reason::user_shutdown);
                self->quit();
            }
        }
    };
}

// Test actor behavior for output_mapping
caf::behavior mmul_mapping_test(caf::stateful_actor<test_actor_state>* self, 
                                caf::actor facade, int N) {
    self->state().N = N;
    self->state().h_a.assign(N * N, 5);
    self->state().h_b.assign(N * N, 6);
    self->state().h_c.assign(N * N, 0); // Target buffer for direct copy

    auto arg1 = caf::cuda::create_in_arg(self->state().h_a);
    auto arg2 = caf::cuda::create_in_arg(self->state().h_b);
    auto arg3 = caf::cuda::create_out_arg_with_size<int>(N * N);
    auto arg4 = caf::cuda::create_in_arg(N);

    // Define the output mapping for index 2 (Matrix C)
    output_mapping m{2, self->state().h_c.data(), self->state().h_c.size()};
    std::vector<output_mapping> mappings{m};

    std::cout << "[INFO] Launching output_mapping test..." << std::endl;
    self->state().start_time = std::chrono::steady_clock::now();
    
    // Launch using the mapping-enabled overload
    self->mail(mappings, arg1, arg2, arg3, arg4).send(facade);

    return {
        [=](int r_id, int index, std::vector<int> data) {
            // This should not be called for index 2 because it is mapped
            if (index == 2) {
                std::cout << "[ERROR] Received data vector for mapped index 2!" << std::endl;
            }
        },
        [=](int r_id, int index) {
            if (index == 2) {
                std::cout << "[MAPPING] Received notification for mapped index 2." << std::endl;
                auto end_time = std::chrono::steady_clock::now();
                std::chrono::duration<double> elapsed = end_time - self->state().start_time;

                std::cout << "===== Mapping Performance Result =====" << std::endl;
                std::cout << "Round-trip Latency: " << elapsed.count() << " seconds" << std::endl;

                // Verify the data was copied directly into h_c
                verify_mmul(self->state().h_a, self->state().h_b, self->state().h_c, self->state().N);

                self->send_exit(facade, exit_reason::user_shutdown);
                self->quit();
            }
        }
    };
}

// New Separate Actor for testing mem_ptr return
caf::behavior mmul_mem_ptr_return_test(caf::stateful_actor<test_actor_state>* self, 
                                       caf::actor facade, int N) {
    self->state().N = N;
    self->state().h_a.assign(N * N, 7); // Use different values
    self->state().h_b.assign(N * N, 8);
    self->state().h_c.resize(N * N); // This will be filled from mem_ptr copy

    auto arg1 = caf::cuda::create_in_arg(self->state().h_a);
    auto arg2 = caf::cuda::create_in_arg(self->state().h_b);
    auto arg3 = caf::cuda::create_out_arg_with_size<int>(N * N);
    auto arg4 = caf::cuda::create_in_arg(N);

    std::cout << "[INFO] Launching mem_ptr return test..." << std::endl;
    self->state().start_time = std::chrono::steady_clock::now();
    
    // Request mem_ptrs back directly
    self->mail(return_mem_ptr_atom_v, arg1, arg2, arg3, arg4).send(facade);

    return {
        [=](int r_id, caf::cuda::mem_ptr<int> a_ptr, caf::cuda::mem_ptr<int> b_ptr, 
            caf::cuda::mem_ptr<int> c_ptr, caf::cuda::mem_ptr<int> n_ptr) {
            
            auto end_time = std::chrono::steady_clock::now();
            std::chrono::duration<double> elapsed = end_time - self->state().start_time;
            std::cout << "[MEM_PTR_RETURN] Latency: " << elapsed.count() << "s" << std::endl;

            // Only copy the output mem_ptr (c_ptr) to host, as IN arguments cannot be copied back.
            std::vector<int> h_c_from_gpu = c_ptr->copy_to_host();
            
            // Verify the output matrix C using the original host inputs
            verify_mmul(self->state().h_a, self->state().h_b, h_c_from_gpu, self->state().N);

            // Optionally, verify mem_ptr metadata for all returned pointers
            if (a_ptr && b_ptr && c_ptr && n_ptr) {
                if (a_ptr->size() == self->state().N * self->state().N && a_ptr->access() == IN &&
                    b_ptr->size() == self->state().N * self->state().N && b_ptr->access() == IN &&
                    c_ptr->size() == self->state().N * self->state().N && c_ptr->access() == OUT &&
                    n_ptr->size() == 1 && n_ptr->access() == IN) {
                    std::cout << "[SUCCESS] All mem_ptrs returned are valid and metadata is correct." << std::endl;
                } else {
                    std::cout << "[FAILURE] Mem_ptr metadata mismatch!" << std::endl;
                }
            } else {
                std::cout << "[FAILURE] One or more mem_ptrs returned are null!" << std::endl;
            }

            self->send_exit(facade, exit_reason::user_shutdown);
            self->quit();
        }
    };
}

void caf_main(caf::actor_system& sys) {
    // Initialize the CUDA subsystem
    caf::cuda::manager::init(sys);
    auto& mgr = caf::cuda::manager::get();

    // Problem size
    int N = 1024;
    int THREADS = 32;
    int BLOCKS = (N + THREADS - 1) / THREADS;
    caf::cuda::nd_range dims(BLOCKS, BLOCKS, 1, THREADS, THREADS, 1);

    // Spawn the actor_facade using the CUBIN file.
    // Signature: matrixMul(const int* a, const int* b, int* c, int N)
    auto facade = mgr.spawnFromCUBIN(
        "../mmul.cubin", "matrixMul", dims,
        in<int>{}, in<int>{}, out<int>{}, in<int>{});

    // 1. Run Basic Test
    auto basic_tester = sys.spawn(mmul_facade_test, facade, N);
    
    // 2. Run Advanced Test (Stream 777, Device 0, Index 2 Only)
    // We can spawn it now; it will execute after the basic tester finishes or in parallel.
    // Note: If you want sequential execution, use request().then() or a supervisor.
    sys.spawn(mmul_advanced_facade_test, facade, N, 0, 777, std::vector<int>{2});

    // 3. Run Mapping Test
    sys.spawn(mmul_mapping_test, facade, N);

    // 4. Run mem_ptr return test
    sys.spawn(mmul_mem_ptr_return_test, facade, N);

    sys.await_all_actors_done();
    caf::cuda::manager::shutdown();
}

CAF_MAIN(id_block::cuda)
