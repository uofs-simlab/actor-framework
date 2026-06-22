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
#include <iomanip>

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

struct supervisor_state {
    program_ptr prog;
    int N_val;
    std::shared_ptr<MatrixPool> pool;
    actor exit_actor;
    actor stats_actor;
    response_token_ptr res;
    std::mt19937 rng;
    int retries = 0;
};

struct stats_actor_state {
    std::map<int, int> succeeded_by_retries;
    int failed = 0;
    double runtime = 0.0;

    // prevents double-finalize
    bool finalized = false;
};



behavior stats_actor_fun(stateful_actor<stats_actor_state>* self) {
    return {

        // SUCCESS / FAILURE REPORTING
        [=](bool success, int retries_taken) {
            auto& st = self->state();

            if (success) {
                st.succeeded_by_retries[retries_taken]++;
            } else {
                st.failed++;
            }

            int total = 0;
            for (auto const& [r, c] : st.succeeded_by_retries)
                total += c;
            total += st.failed;

            if (total % 1000 == 0 && total > 0) {
                std::cout << "[STATS] Progress: "
                          << total << " jobs processed\n";
            }
        },

        // RUNTIME UPDATE
        [=](double runtime) {
            self->state().runtime = runtime;
        },

        // FINAL REPORT (IMPORTANT FIX)
        [=](char finalize_stats) {
            auto& st = self->state();

            if (st.finalized)
                return;

            st.finalized = true;

            int total_succeeded = 0;
            for (auto const& [r, c] : st.succeeded_by_retries)
                total_succeeded += c;

            int total = total_succeeded + st.failed;

            double success_pct = (total > 0)
                ? (100.0 * total_succeeded / total)
                : 0.0;

            std::cout << "\n=====================================\n";
            std::cout << "[STATS REPORT] Iteration Complete\n";
            std::cout << "  Total Processed: " << total << "\n";
            std::cout << "  Total Succeeded: " << total_succeeded << "\n";
            std::cout << "  Total Failed:    " << st.failed << "\n";

            std::cout << "  Succeeded by Retries:\n";
            for (auto const& [r, c] : st.succeeded_by_retries) {
                double pct = (total_succeeded > 0)
                    ? (100.0 * c / total_succeeded)
                    : 0.0;

                std::cout << "    " << r << " retries: "
                          << c << " jobs ("
                          << std::fixed << std::setprecision(2)
                          << pct << "%)\n";
            }

            std::cout << "  Overall Success %: "
                      << std::fixed << std::setprecision(2)
                      << success_pct << "%\n";

            if (st.runtime > 0.0) {
                std::cout << "  Runtime: "
                          << std::fixed << std::setprecision(3)
                          << st.runtime << " s\n";
            }

            std::cout << "=====================================\n";
        }
    };
}

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
behavior task_worker_fun(stateful_actor<task_actor_state>* self, caf::actor stats_actor, caf::actor exit_actor) {
   
   
    //   self->attach_functor([](const caf::error& reason) {
    //     std::cout << "[worker EXIT] "
    //               << to_string(reason)
    //               << std::endl;
    // });
   
   
   
    return {
        
        [=](response_token_ptr res) mutable {
            try {
                // std::cout << "[WORKER] Starting task N=" << self->state().N_val << std::endl;
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
                int THREADS= 32;

                  nd_range range((N + THREADS - 1) / THREADS, 
                           (N + THREADS - 1) / THREADS, 1, 
                           THREADS, THREADS, 1);




                // 2. Launch Work asynchronously using the stream and device assigned by the scheduler.
                auto result_tuple = runner.run_async(st.prog, range, res, in_a, in_b, out_c, in_n);
                auto d_c = std::get<2>(result_tuple);

                // 3. Asynchronous Copyback.
                // Allocate a local buffer for the result to keep the total system memory low.
                auto h_c = std::make_shared<std::vector<int>>(N * N);
                auto self_hdl = actor_cast<actor>(self);

                // The launch_response_token is released inside the callback 
                // to signal to the scheduler that the resource is free. (No serial verification here)
                // runner.copy_to_host_async(d_c, h_c->data(), h_c->size(), [res, stats_actor, exit_actor, h_c, self_hdl](int* /*ptr*/, size_t /*sz*/) mutable {
                //     res->release(); // Release the token
                //     // The worker no longer sends directly to stats_actor.
                //     // It signals normal completion to itself, which causes its supervisor to be notified.
                //     anon_mail(1).send(exit_actor);
                //     anon_mail(0).send(self_hdl); // Signal normal completion
                // });


                runner.add_callback(res->getStreamId(),res->getDeviceNumber(),[res,stats_actor,exit_actor,self_hdl](){
                    res->release(); // Release the token
                    // The worker no longer sends directly to stats_actor.
                    // It signals normal completion to itself, which causes its supervisor to be notified.
                    anon_mail(1).send(exit_actor);
                    anon_mail(0).send(self_hdl); // Signal normal completion

                });

            } catch (const std::exception& e) {
                std::cerr << "[WORKER] Exception caught: " << e.what() << std::endl;
                self->quit(sec::runtime_error);
            }
        },
        [=](int signal) {
            if (signal == 0)
                self->quit();
        }
    };
}

// Helper function to initialize task_actor_state
behavior make_task_worker_behavior(stateful_actor<task_actor_state>* self, program_ptr prog, int N_val, std::shared_ptr<MatrixPool> pool, caf::actor stats_actor, caf::actor exit_actor) {
    auto& st = self->state();
    st.prog = std::move(prog);
    st.N_val = N_val;
    st.pool = std::move(pool);
    return task_worker_fun(self, stats_actor, exit_actor);
}

behavior task_supervisor_fun(stateful_actor<supervisor_state>* self) {
   
   
    //   self->attach_functor([](const caf::error& reason) {
    //     std::cout << "[SUPERVISOR EXIT] "
    //               << to_string(reason)
    //               << std::endl;
    // });
   
   
    return {
        [=](response_token_ptr res) {
            if (res->getType() == LAUNCH_RESPONSE) {
                self->state().res = res;
                auto w = self->spawn(make_task_worker_behavior,
                                     self->state().prog,
                                     self->state().N_val,
                                     self->state().pool,
                                     self->state().stats_actor,
                                     self->state().exit_actor);

                self->monitor(w, [self, res](const error& err) mutable {
                    if (err) {
                        self->state().retries++;
                        if (self->state().retries > 0) {
                            // Max retries reached, permanent failure
                            std::cout << "[SUPERVISOR] Task N=" << self->state().N_val 
                                      << " failed permanently after 5 retries. Giving up." << std::endl;
                            
                            // Send failure to stats_actor with the retry count (5)
                            self->mail(false, self->state().retries).send(self->state().stats_actor);
                            
                            self->mail(1).send(self->state().exit_actor);
                            res->release();
                            self->quit();
                        } else {
                            res -> release();
                            std::uniform_int_distribution<> dis(10, 100);
                            auto backoff = std::chrono::milliseconds(dis(self->state().rng));

                            std::cout << "[SUPERVISOR] Task worker failed (" << to_string(err)
                                      << "). Restarting N=" << self->state().N_val 
                                      << " after " << backoff.count() << "ms backoff (Retry " 
                                      << self->state().retries << "/5)." << std::endl;


                            auto token = make_launch_token(self->state().prog, nd_range(1,1,1,1,1,1), 0, 
                             res -> name(), self);
                            self->mail(token).delay(backoff).send(self); // Trigger restart logic with delay
                        }
                    } else {
                        // Worker exited normally (success)
                        // Send success to stats_actor with the retry count
                        self->mail(true, self->state().retries).send(self->state().stats_actor);
                        self->quit();
                    }
                });
                
                self->mail(res).send(w);
            
            
            }
            
        },
        [=](token_ptr token) {
                caf::cuda::manager::get().send_scheduler_actor_message(token);
        }

    };
}

behavior make_task_supervisor_behavior(stateful_actor<supervisor_state>* self, 
                                      program_ptr prog, 
                                      int N_val, 
                                      std::shared_ptr<MatrixPool> pool, 
                                      actor exit_actor, 
                                      actor stats_actor) {
    auto& st = self->state();
    st.prog = std::move(prog);
    st.N_val = N_val;
    st.pool = std::move(pool);
    st.exit_actor = exit_actor;
    st.stats_actor = stats_actor;
    st.rng.seed(std::random_device{}());
    return task_supervisor_fun(self);
}

template <class Fn>
double time_run(Fn&& fn) {
    auto start = std::chrono::steady_clock::now();
    fn();
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    return elapsed.count();
}



enum class memory_pressure_level {
    none,
    low,
    medium,
    high
};

inline const char* to_string(memory_pressure_level lvl) {
    switch (lvl) {
        case memory_pressure_level::none:   return "NONE";
        case memory_pressure_level::low:    return "LOW";
        case memory_pressure_level::medium: return "MEDIUM";
        case memory_pressure_level::high:   return "HIGH";
    }
    return "UNKNOWN";
}


struct pressure_profile {
    size_t target_free_bytes;
};

inline pressure_profile get_profile(memory_pressure_level lvl, size_t total_mem) {
    switch (lvl) {
        case memory_pressure_level::none:
            return { total_mem };              // no allocation pressure

        case memory_pressure_level::low:
            return { static_cast<size_t>(total_mem * 0.40) }; // 40% free

        case memory_pressure_level::medium:
            return { static_cast<size_t>(total_mem * 0.20) }; // 20% free

        case memory_pressure_level::high:
            return { static_cast<size_t>(total_mem * 0.05) }; // 5% free
    }

    return { static_cast<size_t>(total_mem * 0.20) };
}



static std::vector<caf::cuda::mem_ptr<char>> pressure_holder;


void apply_memory_pressure(int device_id,
                           size_t total_mem_bytes,
                           memory_pressure_level level)
{
    auto dev = manager::get().find_device(device_id);

    size_t target_free_bytes = get_profile(level, total_mem_bytes).target_free_bytes;
    size_t available = dev->available_memory_bytes();

    std::cout << "\n[PRESSURE] ================================\n";
    std::cout << "[PRESSURE] Device " << device_id
              << " | Level: " << to_string(level) << "\n";
    std::cout << "[PRESSURE] Available: " << available / (1024 * 1024) << " MB\n";
    std::cout << "[PRESSURE] Target free: " << target_free_bytes / (1024 * 1024) << " MB\n";

    // NONE mode → explicitly do nothing
    if (level == memory_pressure_level::none) {
        std::cout << "[PRESSURE] NONE selected → no allocations performed\n";
        return;
    }

    if (available <= target_free_bytes) {
        std::cout << "[PRESSURE] Already under target pressure\n";
        return;
    }

    size_t to_allocate = available - target_free_bytes;

    try {
        auto arg = create_out_arg_with_size<char>(to_allocate);

        static command_runner<out<char>> p_runner;

        pressure_holder.push_back(
            p_runner.transfer_memory(device_id, 0, arg)
        );

        auto stream = p_runner.get_stream(0, device_id);
        CHECK_CUDA(cuStreamSynchronize(stream));

        size_t post_available = dev->available_memory_bytes();

        std::cout << "[PRESSURE] Allocated: "
                  << to_allocate / (1024 * 1024) << " MB\n";

        std::cout << "[PRESSURE] Post-free: "
                  << post_available / (1024 * 1024) << " MB\n";

        if (post_available > target_free_bytes + (50ULL * 1024 * 1024)) {
            std::cerr << "[WARNING] Pressure mismatch at level "
                      << to_string(level) << "\n";
        }

    } catch (const std::exception& e) {
        std::cerr << "[PRESSURE] FAILED (" << to_string(level)
                  << "): " << e.what() << "\n";
    }
}




void run_single_pressure_test(actor_system& sys,
                              memory_pressure_level level,
                              int num_tasks,
                              int streams)
{
    caf::cuda::manager::init(sys);
    auto& mgr = manager::get();

    std::cout << "\n\n====================================================\n";
    std::cout << "[TEST] Scheduler run under pressure: "
              << to_string(level) << "\n";
    std::cout << "====================================================\n";

    // Apply pressure BEFORE launching workload
    for (int i = 0; i < mgr.get_num_devices(); i++) {
        auto dev = mgr.find_device(i);
        size_t total_mem = dev->total_memory_bytes();

        apply_memory_pressure(i, total_mem, level);
    }

    mgr.toggle_scheduler_actor(streams, 1);

    auto program = mgr.create_program_from_cubin("../mmul.cubin", "matrixMul");

    const int THREADS = 32;

    std::vector<token_ptr> tokens;
    std::mt19937 rng(42);
 
    const int min_N = 2048;
    const int max_N = 4096;
    const int num_distinct_sizes = 20;
    // const std::vector<int> actor_counts = {50000};
    const std::vector<int> actor_counts = {200};
    // int streams = 8;

    // Generate deterministic random pool once
    auto pool_ptr = std::make_shared<MatrixPool>(
        create_matrix_pool_random(num_distinct_sizes, min_N, max_N, 42));
    
    std::vector<int> available_Ns;
    for (auto& p : pool_ptr->A)
        available_Ns.push_back(p.first);

    std::uniform_int_distribution<size_t> dist(0, available_Ns.size() - 1);

    auto exit_actor = sys.spawn(caf::cuda::exit_actor_fun, num_tasks);
    auto stats_actor = sys.spawn(stats_actor_fun);

    std::vector<caf::actor> sups;

    for (int i = 0; i < num_tasks; ++i) {
        int N = available_Ns[dist(rng)];

        nd_range range((N + THREADS - 1) / THREADS,
                       (N + THREADS - 1) / THREADS,
                       1,
                       THREADS,
                       THREADS,
                       1);

        auto supervisor = sys.spawn(make_task_supervisor_behavior,
                                    program,
                                    N,
                                    pool_ptr,
                                    exit_actor,
                                    stats_actor);

        sups.push_back(supervisor);

        tokens.push_back(
            make_launch_token(program, range, 0,
                              "task_" + std::to_string(i),
                              supervisor));
    }

    scoped_actor self{sys};

    double elapsed = time_run([&]() {
        std::cout << "[TEST] Dispatching workload...\n";

        mgr.send_scheduler_actor_message(std::move(tokens));

        self->wait_for(exit_actor);
    });

    std::cout << "[RESULT] Pressure=" << to_string(level)
              << " | time=" << elapsed << " s\n";

    self->mail('c').send(stats_actor);
    anon_send_exit(stats_actor, exit_reason::user_shutdown);
    self->wait_for(stats_actor);

    pressure_holder.clear();
    manager::shutdown();
}

void run_pressure_benchmark(actor_system& sys)
{
    std::vector<memory_pressure_level> levels = {
        memory_pressure_level::none,
        memory_pressure_level::low,
        memory_pressure_level::medium,
        memory_pressure_level::high
    };

    const int num_tasks = 200;
    const int streams = 8;

    for (auto level : levels) {
        run_single_pressure_test(sys, level, num_tasks, streams);
    }
}




void caf_main(actor_system& sys) {
    run_pressure_benchmark(sys);
    std::cout << "[MAIN] Integration test complete." << std::endl;
}

CAF_MAIN(id_block::cuda_control)
