#include <caf/all.hpp>
#include <caf/cuda/all.hpp>
#include <iostream>
#include <vector>
#include <chrono>

using namespace caf;
using namespace std::chrono_literals;

struct latency_test_state {
    std::chrono::steady_clock::time_point start_time;
    int N;
};

caf::behavior latency_manager(caf::stateful_actor<latency_test_state>* self, 
                              caf::actor facade, int N) {
    self->state().N = N;
    
    std::vector<int> h_a(N * N, 2);
    std::vector<int> h_b(N * N, 3);
    
    auto arg1 = caf::cuda::create_in_arg(std::move(h_a));
    auto arg2 = caf::cuda::create_in_arg(std::move(h_b));
    auto arg3 = caf::cuda::create_out_arg_with_size<int>(N * N);
    auto arg4 = caf::cuda::create_in_arg(N);

    self->state().start_time = std::chrono::steady_clock::now();
    
    // Launch the work
    self->mail(arg1, arg2, arg3, arg4).send(facade);

    return {
        [=](int r_id, int index, std::vector<int> data) {
            // We don't need to do anything with the data for timing
        },
        [=](int r_id, int index) {
            if (index == -1) { // Completion signal
                auto end_time = std::chrono::steady_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                    end_time - self->state().start_time).count();
                
                std::cout << "[LATENCY TEST] matrix_size=" << self->state().N 
                          << ", time=" << elapsed << " ms" << std::endl;
                
                self->send_exit(facade, exit_reason::user_shutdown);
                self->quit();
            }
        }
    };
}

void run_latency_test(caf::actor_system& sys, int matrix_size) {
    caf::cuda::manager::init(sys);
    auto& mgr = caf::cuda::manager::get();

    int THREADS = 32;
    int BLOCKS = (matrix_size + THREADS - 1) / THREADS;
    caf::cuda::nd_range dims(BLOCKS, BLOCKS, 1, THREADS, THREADS, 1);

    // Spawn facade
    auto facade = mgr.spawnFromCUBIN(
        "../mmul.cubin", "matrixMul", dims,
        in<int>{}, in<int>{}, out<int>{}, in<int>{});

    sys.spawn(latency_manager, facade, matrix_size);
    sys.await_all_actors_done();
    caf::cuda::manager::shutdown();
}

void caf_main(caf::actor_system& sys) {
    std::vector<int> sizes = {1000, 4000, 8000, 12000};
    for (int size : sizes) {
        run_latency_test(sys, size);
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

CAF_MAIN(id_block::cuda)