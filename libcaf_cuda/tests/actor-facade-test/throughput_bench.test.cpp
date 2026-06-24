#include <caf/all.hpp>
#include <caf/cuda/all.hpp>
#include <iostream>
#include <vector>
#include <chrono>

using namespace caf;
using namespace std::chrono_literals;

struct throughput_state {
    int total_expected = 0;
    int results_received = 0;
    std::chrono::steady_clock::time_point start_time;
    int N;
    std::vector<int> h_a;
    std::vector<int> h_b;
    in<int> arg1;
    in<int> arg2;
    out<int> arg3;
    in<int> arg4;
};

caf::behavior throughput_manager(caf::stateful_actor<throughput_state>* self, 
                                 caf::actor facade, int N, int iterations) {
    auto& st = self->state();
    st.total_expected = iterations;
    st.N = N;

    // Initialize data and reuseable kernel arguments in state
    st.h_a.assign(N * N, 2);
    st.h_b.assign(N * N, 3);
    st.arg1 = caf::cuda::create_in_arg(st.h_a);
    st.arg2 = caf::cuda::create_in_arg(st.h_b);
    st.arg3 = caf::cuda::create_out_arg_with_size<int>(N * N);
    st.arg4 = caf::cuda::create_in_arg(N);

    // Only copy back index 2 (Matrix C)
    std::vector<int> output_indices = {2};
    st.start_time = std::chrono::steady_clock::now();

    for (int i = 0; i < iterations; ++i) {
        self->mail(output_indices, st.arg1, st.arg2, st.arg3, st.arg4).send(facade);
    }

    return {
        [=](int r_id, int index, std::vector<int> data) {
            if (index == 2) { // Matrix C arrived
                if (++self->state().results_received == self->state().total_expected) {
                    auto end_time = std::chrono::steady_clock::now();
                    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                        end_time - self->state().start_time).count();
                    
                    std::cout << "[THROUGHPUT TEST] matrix_size=" << self->state().N
                              << " iterations=" << self->state().total_expected
                              << ", total_time=" << elapsed << " ms" << std::endl;
                    
                    self->send_exit(facade, exit_reason::user_shutdown);
                    self->quit();
                }
            }
        }
    };
}

void run_throughput_test(caf::actor_system& sys, int matrix_size, int iterations) {
    caf::cuda::manager::init(sys);
    auto& mgr = caf::cuda::manager::get();

    int THREADS = 32;
    int BLOCKS = (matrix_size + THREADS - 1) / THREADS;
    caf::cuda::nd_range dims(BLOCKS, BLOCKS, 1, THREADS, THREADS, 1);

    // Spawn one facade to handle the stream
    auto facade = mgr.spawnFromCUBIN(
        "../mmul.cubin", "matrixMul", dims,
        in<int>{}, in<int>{}, out<int>{}, in<int>{});

    sys.spawn(throughput_manager, facade, matrix_size, iterations);
    sys.await_all_actors_done();
    caf::cuda::manager::shutdown();
}

void caf_main(caf::actor_system& sys) {
    int fixed_size = 1000;
    for (int i = 1000; i <= 10000; i += 1000) {
        run_throughput_test(sys, fixed_size, i);
    }
}

int main(int argc, char** argv) {
    core::init_global_meta_objects();
    actor_system_config cfg;
    
    // Single thread configuration per user snippet
    cfg.set("caf.scheduler.max-threads", 1);
    cfg.set("caf.scheduler.policy", "sharing");
    
    auto err = cfg.parse(argc, argv);
    if (err) return EXIT_FAILURE;
    if (cfg.helptext_printed()) return 0;

    actor_system sys{cfg};
    caf_main(sys);
    
    return 0;
}
