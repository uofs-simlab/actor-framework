#include <caf/all.hpp>
#include <caf/cuda/all.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <deque>
#include <string>
#include <chrono>
#include <numeric>
#include <cmath>
#include <algorithm>
#include <filesystem>
#include <memory>
#include "caf/actorSOLVE/actorSOLVE.hpp"
#include "sparse_utils.hpp"

using namespace caf;
using namespace caf::cuda;
namespace fs = std::filesystem;

constexpr uint32_t WORKLOAD_SEED = 42;

template <class Inspector>
bool inspect(Inspector& f, SolverType& x) {
    auto val = static_cast<int>(x);
    if (f.apply(val)) {
        if constexpr (Inspector::is_loading)
            x = static_cast<SolverType>(val);
        return true;
    }
    return false;
}

CAF_BEGIN_TYPE_ID_BLOCK(workload_test, caf::id_block::cuda::end)
    CAF_ADD_ATOM(workload_test, get_work_atom)
    CAF_ADD_ATOM(workload_test, release_memory_atom)
    CAF_ADD_ATOM(workload_test, request_work_atom)
    CAF_ADD_ATOM(workload_test, worker_done_atom)
    CAF_ADD_ATOM(workload_test, work_tick_atom)
    CAF_ADD_ATOM(workload_test, add_work_atom)
    CAF_ADD_ATOM(workload_test, steal_work_atom)
    CAF_ADD_ATOM(workload_test, shutdown_atom)
    CAF_ADD_TYPE_ID(workload_test, (SolverType))
    CAF_ADD_TYPE_ID(workload_test, (MatrixTask))
    CAF_ADD_TYPE_ID(workload_test, (std::vector<MatrixTask>))
    CAF_ADD_TYPE_ID(workload_test, (std::shared_ptr<MatrixData>))
CAF_END_TYPE_ID_BLOCK(workload_test)

CAF_ALLOW_UNSAFE_MESSAGE_TYPE(MatrixData)
CAF_ALLOW_UNSAFE_MESSAGE_TYPE(std::shared_ptr<MatrixData>)
CAF_ALLOW_UNSAFE_MESSAGE_TYPE(MatrixTask)
CAF_ALLOW_UNSAFE_MESSAGE_TYPE(std::vector<MatrixTask>)

// ---------------------------- WORKER ACTOR ----------------------------
struct worker_state {
    caf::actor supervisor;
    std::vector<caf::actor> peers;
    std::deque<MatrixTask> work_queue;
    int device_id;
    int stream_id;
    std::string current_matrix_path;
    std::chrono::steady_clock::time_point task_start;
    caf::actor cg_facade;
    bool stealing = false;
};

behavior sparse_worker_fun(stateful_actor<worker_state>* self,
                           caf::actor supervisor, int dev_id, int stream_id) {
    auto& st = self->state();
    st.supervisor = supervisor;
    st.device_id = dev_id;
    st.stream_id = stream_id;
    st.cg_facade = self->spawn<sparse_cg_facade<float>, linked>(0);

    return {
        [=](std::vector<caf::actor>& peers) {
            self->state().peers = std::move(peers);
        },
        [=](add_work_atom, std::vector<MatrixTask>& batch) {
            auto& st = self->state();
            bool was_idle = st.work_queue.empty() && !st.stealing;
            for (auto& t : batch)
                st.work_queue.push_back(std::move(t));
            
            if (was_idle)
                self->mail(request_work_atom_v).send(self);
        },
        [=](request_work_atom) {
            auto& st = self->state();
            if (!st.work_queue.empty()) {
                auto task = std::move(st.work_queue.front());
                st.work_queue.pop_front();

                auto& data = *task.data;
                st.current_matrix_path = task.path;
                st.task_start = std::chrono::steady_clock::now();

                if (task.type == CGS_SOLVER) {
                    self->mail(create_in_arg((const std::vector<int32_t>&)data.row_ptr),
                               create_in_arg((const std::vector<int32_t>&)data.col_indices),
                               create_in_arg(data.values), create_in_arg(data.b),
                               create_in_out_arg(data.x_guess), matrix_format::csr,
                               (int)data.row_ptr.size() - 1, (int)data.values.size(),
                               1e-5f, 2000, st.device_id, st.stream_id).send(st.cg_facade);
                } else {
                    auto solver = self->spawn<sparse_bicgstab_actor<float>>(
                        create_in_arg((const std::vector<int32_t>&)data.row_ptr),
                        create_in_arg((const std::vector<int32_t>&)data.col_indices),
                        create_in_arg(data.values), create_in_arg(data.b),
                        create_in_out_arg(data.x_guess), matrix_format::csr,
                        (int)data.row_ptr.size() - 1, (int)data.values.size(),
                        1e-5f, 2000, st.device_id, st.stream_id, actor_cast<actor>(self));
                    self->mail(start_atom_v).send(solver);
                }
            } else {
                // Idle: try to steal from a random peer
                if (st.peers.empty() || st.stealing) return;
                
                st.stealing = true;
                static std::mt19937 prng(std::random_device{}());
                std::uniform_int_distribution<size_t> dist(0, st.peers.size() - 1);
                auto victim = st.peers[dist(prng)];

                if (victim == self) {
                    st.stealing = false;
                    self->mail(request_work_atom_v).delay(std::chrono::milliseconds(10)).send(self);
                    return;
                }

                self->mail(steal_work_atom_v).request(victim, std::chrono::seconds(1)).then(
                    [=](MatrixTask& stolen_task) {
                        auto& st = self->state();
                        st.work_queue.push_back(std::move(stolen_task));
                        st.stealing = false;
                        self->mail(request_work_atom_v).send(self);
                    },
                    [=](const error&) {
                        auto& st = self->state();
                        st.stealing = false;
                        self->mail(request_work_atom_v).delay(std::chrono::milliseconds(50)).send(self);
                    }
                );
            }
        },
        [=](steal_work_atom) -> result<MatrixTask> {
            auto& st = self->state();
            if (st.work_queue.size() > 1) {
                auto task = std::move(st.work_queue.back());
                st.work_queue.pop_back();
                return task;
            }
            return sec::no_context;
        },
        [=](shutdown_atom) {
            self->mail(worker_done_atom_v).send(self->state().supervisor);
            self->quit();
        },
        [=](std::vector<float>& solution, solver_result_meta meta) {
            auto task_end = std::chrono::steady_clock::now();
            std::chrono::duration<double> task_duration = task_end - self->state().task_start;
            self->println("Worker {}: Round-trip for {} (BICSTAB_SOLVER) took {} s (Iters: {})", 
                          self->state().stream_id, self->state().current_matrix_path, task_duration.count(), meta.iterations);
            self->mail(1).send(self->state().supervisor);
            self->mail(request_work_atom_v).send(self);
        },
        [=](uint32_t /*r_id*/, int /*index*/, std::vector<float>& solution, solver_result_meta meta) {
            auto task_end = std::chrono::steady_clock::now();
            std::chrono::duration<double> task_duration = task_end - self->state().task_start;
            self->println("Worker {}: Round-trip for {} (CGS_SOLVER_OPTIMIZED) took {} s (Iters: {})", 
                          self->state().stream_id, self->state().current_matrix_path, task_duration.count(), meta.iterations);
            self->mail(1).send(self->state().supervisor);
            self->mail(request_work_atom_v).send(self);
        }
    };
}

// ---------------------------- SUPERVISOR ACTOR ----------------------------
struct supervisor_state {
    std::vector<MatrixTask> matrix_pool;
    std::vector<caf::actor> workers;
    std::mt19937 rng;
    int total_tasks;
    int completed = 0;
    int batches_remaining;
    int batch_size;
    double mean_arrival_ms;
    caf::actor parent;
    int workers_shutdown = 0;
};

behavior supervisor_actor_fun(stateful_actor<supervisor_state>* self, 
                              std::vector<MatrixTask> matrix_pool,
                              int num_streams, int num_batches, int batch_size, double mean_arrival_ms, caf::actor parent) {
    auto& st = self->state();
    st.matrix_pool = std::move(matrix_pool);
    st.parent = std::move(parent);
    st.total_tasks = num_batches * batch_size;
    st.batches_remaining = num_batches;
    st.batch_size = batch_size;
    st.mean_arrival_ms = mean_arrival_ms;
    st.rng.seed(WORKLOAD_SEED);
    
    manager& mgr = manager::get();
    int num_gpus = mgr.get_num_devices();

    // Initializing workers
    for (int i = 0; i < num_gpus; ++i) {
        for (int j = 0; j < num_streams; ++j) {
            st.workers.push_back(self->spawn(sparse_worker_fun, self, i, (i * 100) + j));
        }
    }

    // Inform workers of their peers
    for (auto& w : st.workers)
        self->mail(st.workers).send(w);

    // Start the production cycle
    self->mail(work_tick_atom_v).send(self);

    return {
        [=](work_tick_atom) {
            auto& st = self->state();
            if (st.batches_remaining > 0) {
                auto batch = generate_batch(st.matrix_pool, st.rng, st.batch_size);
                st.batches_remaining--;

                // Partition this specific batch among workers
                size_t tasks_per_worker = batch.size() / st.workers.size();
                for (size_t i = 0; i < st.workers.size(); ++i) {
                    auto start = batch.begin() + i * tasks_per_worker;
                    auto end = (i == st.workers.size() - 1) ? batch.end() : start + tasks_per_worker;
                    
                    std::vector<MatrixTask> segment;
                    for (auto it = start; it != end; ++it)
                        segment.push_back(std::move(*it));
                    
                    self->mail(add_work_atom_v, std::move(segment)).send(st.workers[i]);
                }

                if (st.batches_remaining > 0) {
                    auto delay = generate_random_interval(st.rng, st.mean_arrival_ms);
                    self->println("[SUPERVISOR] Next batch in {}ms", delay.count());
                    self->mail(work_tick_atom_v).delay(delay).send(self);
                }
            }
        },
        [=](int done) {
            auto& st = self->state();
            st.completed += done;
            if (st.completed >= st.total_tasks) {
                self->println("\n[DONE] All {} tasks processed. Terminating workers...", st.total_tasks);
                for (auto& worker : st.workers)
                    self->mail(shutdown_atom_v).send(worker);
            }
        },
        [=](worker_done_atom) {
            auto& st = self->state();
            if (++st.workers_shutdown == (int)st.workers.size()) {
                self->mail(worker_done_atom_v).send(st.parent);
                self->quit();
            }
        }
    };
}

void caf_main(actor_system& sys) {
    manager::init(sys, manager_config(true, true));
    
    int num_streams = 4;
    int num_batches = 25;
    int batch_size = 100;
    double mean_arrival_ms = 1000.0;

    // // Basic command line argument parsing similar to native
    // auto& args = sys.config().remainder;
    // if (args.size() > 0) num_streams = std::max(1, std::stoi(args[0]));
    // if (args.size() > 1) num_batches = std::max(1, std::stoi(args[1]));
    // if (args.size() > 2) batch_size = std::max(1, std::stoi(args[2]));
    // if (args.size() > 3) mean_arrival_ms = std::stod(args[3]);

    std::cout << "[INFO] Loading matrices into memory...\n";
    auto tasks_vec = scan_for_matrices("/scratch/nqr159/matrix-collection/matrix_corpus_v2/matrices/spd", CGS_SOLVER);
    //scan("/scratch/nqr159/matrix-collection/matrices/unsymmetric", BICSTAB_SOLVER);
    
    if (tasks_vec.empty()) {
        std::cerr << "No matrix files found in search paths.\n";
        manager::shutdown();
        return;
    }

    std::cout << "[INFO] Matrix pool size: " << tasks_vec.size() << "\n";
    std::cout << "[INFO] Streams/GPU:      " << num_streams << "\n";
    std::cout << "[INFO] Batches:          " << num_batches << "\n";
    std::cout << "[INFO] Batch size:       " << batch_size << "\n";
    std::cout << "[INFO] Mean arrival:     " << mean_arrival_ms << " ms\n";

    auto start = std::chrono::steady_clock::now();
    scoped_actor self{sys};

    self->spawn(supervisor_actor_fun, std::move(tasks_vec), num_streams, num_batches, batch_size, mean_arrival_ms, actor_cast<actor>(self));
    
    self->receive(
        [&](worker_done_atom) {
            std::cout << "[INFO] Supervisor signaled completion. Shutting down...\n";
        }
    );

    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    int total_tasks = num_batches * batch_size;

    std::cout << "\n===== BENCHMARK COMPLETE =====\n";
    std::cout << "Seed:               " << WORKLOAD_SEED << "\n";
    std::cout << "Streams per GPU:    " << num_streams << "\n";
    std::cout << "Batches:            " << num_batches << "\n";
    std::cout << "Batch Size:         " << batch_size << "\n";
    std::cout << "Mean Arrival (ms):  " << mean_arrival_ms << "\n";
    std::cout << "Tasks Processed:    " << total_tasks << "\n";
    std::cout << "Total Runtime:      " << elapsed.count() << " s\n";
    std::cout << "Throughput:         " << total_tasks / elapsed.count() << " tasks/s\n";
    std::cout << "==============================\n";

    manager::shutdown();
}
CAF_MAIN(id_block::cuda,id_block::workload_test)