#include <caf/all.hpp>
#include <caf/cuda/all.hpp>
#include <iostream>
#include <vector>
#include <chrono>

using namespace caf;
using namespace std::chrono_literals;

struct mapping_throughput_state {
    int total_expected = 0;
    int results_received = 0;
    std::chrono::steady_clock::time_point start_time;
    int N;
    std::vector<int> h_a;
    std::vector<int> h_b;
    std::vector<int> h_c_global; // The persistent buffer
    in<int> arg1;
    in<int> arg2;
    out<int> arg3;
    in<int> arg4;
};

caf::behavior throughput_mapping_manager(caf::stateful_actor<mapping_throughput_state>* self, 
                                         caf::actor facade, int N, int iterations) {
    auto& st = self->state();
    st.total_expected = iterations;
    st.N = N;

    // Initialize data and reuseable kernel arguments
    st.h_a.assign(N * N, 2);
    st.h_b.assign(N * N, 3);
    st.h_c_global.assign(N * N, 0); // Pre-allocate the global destination
    
    st.arg1 = caf::cuda::create_in_arg(st.h_a);
    st.arg2 = caf::cuda::create_in_arg(st.h_b);
    st.arg3 = caf::cuda::create_out_arg_with_size<int>(N * N);
    st.arg4 = caf::cuda::create_in_arg(N);

    // Define the mapping once
    output_mapping mapping{2, st.h_c_global.data(), st.h_c_global.size()};
    std::vector<output_mapping> mappings = {mapping};

    st.start_time = std::chrono::steady_clock::now();

    for (int i = 0; i < iterations; ++i) {
        // Send using the mappings overload
        self->mail(mappings, st.arg1, st.arg2, st.arg3, st.arg4).send(facade);
    }

    return {
        [=](int r_id, int index) {
            if (index == 2) { // Received notification for index 2 (Matrix C)
                if (++self->state().results_received == self->state().total_expected) {
                    auto end_time = std::chrono::steady_clock::now();
                    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                        end_time - self->state().start_time).count();
                    
                    std::cout << "[MAPPING THROUGHPUT TEST] matrix_size=" << self->state().N
                              << " iterations=" << self->state().total_expected
                              << ", total_time=" << elapsed << " ms" << std::endl;
                    
                    self->send_exit(facade, exit_reason::user_shutdown);
                    self->quit();
                }
            }
        }
    };
}

void run_mapping_throughput_test(caf::actor_system& sys, int matrix_size, int iterations) {
    caf::cuda::manager::init(sys);
    auto& mgr = caf::cuda::manager::get();

    int THREADS = 32;
    int BLOCKS = (matrix_size + THREADS - 1) / THREADS;
    caf::cuda::nd_range dims(BLOCKS, BLOCKS, 1, THREADS, THREADS, 1);

    auto facade = mgr.spawnFromCUBIN(
        "../mmul.cubin", "matrixMul", dims,
        in<int>{}, in<int>{}, out<int>{}, in<int>{});

    sys.spawn(throughput_mapping_manager, facade, matrix_size, iterations);
    sys.await_all_actors_done();
    caf::cuda::manager::shutdown();
}

void caf_main(caf::actor_system& sys) {
    int fixed_size = 1000;
    for (int i = 1000; i <= 10000; i += 1000) {
        run_mapping_throughput_test(sys, fixed_size, i);
    }
}

int main(int argc, char** argv) {
    core::init_global_meta_objects();
    actor_system_config cfg;
    cfg.set("caf.scheduler.max-threads", 1);
    cfg.set("caf.scheduler.policy", "sharing");

    auto err = cfg.parse(argc, argv);
    if (err) return EXIT_FAILURE;
    if (cfg.helptext_printed()) return 0;

    actor_system sys{cfg};
    caf_main(sys);
    return 0;
}