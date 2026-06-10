#include <caf/all.hpp>
#include <caf/cuda/all.hpp>
#include <caf/cuda/control-layer/all-control-layer.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>

using namespace caf;
using namespace caf::cuda;

// A generic command runner to provide access to CUDA stream callbacks
static command_runner<in<int>, in<int>, out<int>, in<int>> runner;

// MatrixPool structure from mmul-random-batch-benchmark
struct MatrixPool {
    std::unordered_map<int, std::vector<int>> A;
    std::unordered_map<int, std::vector<int>> B;
};

struct task_actor_state {
    program_ptr prog;
    int N_val;
    std::shared_ptr<MatrixPool> pool;
};

// create_matrix_pool_random function from mmul-random-batch-benchmark
MatrixPool create_matrix_pool_random(
    int num_sizes,
    int min_N,
    int max_N,
    unsigned int seed
) {
    MatrixPool pool;
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(min_N, max_N);
    std::unordered_set<int> used;
    while (used.size() < static_cast<size_t>(num_sizes)) {
        int N = dist(rng);
        if (used.insert(N).second) {
            pool.A[N] = std::vector<int>(N * N, 1);
            pool.B[N] = std::vector<int>(N * N, 2); // Changed to 2 for distinct input
        }
    }
    return pool;
}

// This actor represents a single task that requests permission from the scheduler.
behavior task_actor_fun(stateful_actor<task_actor_state>* self, caf::actor exit_actor) {
    return {
        [=](response_token_ptr res) mutable {
            if (res->getType() == LAUNCH_RESPONSE) {
                auto& st = self->state();
                
                // We need to cast the base response_token to access the specific nd_range stored in it.
                auto launch_res = static_cast<launch_response_token*>(res.get());
                int N = st.N_val;

                // 1. Setup GPU arguments.
                // Fetch data from the shared pool only when scheduled to save RAM
                auto in_a = create_in_arg(st.pool->A.at(N));
                auto in_b = create_in_arg(st.pool->B.at(N));
                auto out_c = create_out_arg_with_size<int>(N * N);
                auto in_n = create_in_arg(N);

                // 2. Launch Work asynchronously using the stream and device assigned by the scheduler.
                auto result_tuple = runner.run_async(st.prog, launch_res->getRange(), res, in_a, in_b, out_c, in_n);
                auto d_c = std::get<2>(result_tuple);

                // 3. Asynchronous Copyback.
                // Allocate a local buffer for the result to keep the total system memory low.
                auto h_c = std::make_shared<std::vector<int>>(N * N);
                // The launch_response_token is released inside the callback 
                // to signal to the scheduler that the resource is free. (No serial verification here)
                runner.copy_to_host_async(d_c, h_c->data(), h_c->size(), [res, exit_actor, h_c](int* /*ptr*/, size_t /*sz*/) {
                    res->release();
                    anon_mail(1).send(exit_actor);
                });
            }
        }
    };
}

// Helper function to initialize task_actor_state
behavior make_task_actor_behavior(stateful_actor<task_actor_state>* self, program_ptr prog, int N_val, std::shared_ptr<MatrixPool> pool, caf::actor exit_actor) {
    auto& st = self->state();
    st.prog = std::move(prog);
    st.N_val = N_val;
    st.pool = std::move(pool);
    return task_actor_fun(self, exit_actor); // Pass exit_actor to task_actor_fun
}

template <class Fn>
double time_run(Fn&& fn) {
    auto start = std::chrono::steady_clock::now();
    fn();
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    return elapsed.count();
}

void run_scheduler_integration_scaling_test(actor_system& sys) {
    const int min_N = 32;
    const int max_N = 2048;
    const int num_distinct_sizes = 10;
    const std::vector<int> actor_counts = {50000};

    // Generate deterministic random pool once
    auto pool_ptr = std::make_shared<MatrixPool>(
        create_matrix_pool_random(num_distinct_sizes, min_N, max_N, 42));
    
    std::vector<int> available_Ns;
    for (const auto& pair : pool_ptr->A) available_Ns.push_back(pair.first);

    for (int num_tasks : actor_counts) {
        manager_config config;
        manager::init(sys, config);
        auto& mgr = manager::get();

        // Start scheduler with 4 streams and depth 2 (8 slots) to force queuing
        mgr.toggle_scheduler_actor(4, 2);

        auto program = mgr.create_program_from_cubin("../mmul.cubin", "matrixMul");
        const int THREADS = 32;

        std::vector<token_ptr> tokens;
        std::mt19937 rng(42);
        std::uniform_int_distribution<size_t> dist_N_idx(0, available_Ns.size() - 1);

        std::cout << "=====================================\n";
        std::cout << "Scheduler Test | tasks=" << num_tasks << "\n";

        // Spawn the exit actor for this specific test run (moved inside the loop)
        auto exit_actor = sys.spawn(caf::cuda::exit_actor_fun, num_tasks);

        // Spawn task actors and prepare tokens
        for (int i = 0; i < num_tasks; ++i) {
            int current_N = available_Ns[dist_N_idx(rng)];
            nd_range range((current_N + THREADS - 1) / THREADS, 
                           (current_N + THREADS - 1) / THREADS, 1, 
                           THREADS, THREADS, 1);
            
            auto worker = sys.spawn(make_task_actor_behavior,
                                    program,
                                    current_N, 
                                    pool_ptr,
                                    exit_actor); // Pass exit_actor to task actors

            tokens.push_back(make_launch_token(program, range, 0, 
                             "task_" + std::to_string(i), worker));
        }

        double elapsed = time_run([&]() {
            std::cout << "[MAIN] Dispatching batch to scheduler..." << std::endl;
            mgr.send_scheduler_actor_message(std::move(tokens));
        });

        std::cout << "Run complete. Time: " << elapsed << " s\n";
    }
    sys.await_all_actors_done(); // Wait for all actors, including exit_actor, to finish
}

void caf_main(actor_system& sys) {
    run_scheduler_integration_scaling_test(sys);
    std::cout << "[MAIN] Integration test complete." << std::endl;
}

CAF_MAIN(id_block::cuda_control)
