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
    CAF_ADD_TYPE_ID(workload_test, (SolverType))
    CAF_ADD_TYPE_ID(workload_test, (MatrixTask))
    CAF_ADD_TYPE_ID(workload_test, (std::vector<MatrixTask>))
    CAF_ADD_TYPE_ID(workload_test, (std::shared_ptr<MatrixData>))
CAF_END_TYPE_ID_BLOCK(workload_test)

CAF_ALLOW_UNSAFE_MESSAGE_TYPE(MatrixData)
CAF_ALLOW_UNSAFE_MESSAGE_TYPE(std::shared_ptr<MatrixData>)
CAF_ALLOW_UNSAFE_MESSAGE_TYPE(MatrixTask)
CAF_ALLOW_UNSAFE_MESSAGE_TYPE(std::vector<MatrixTask>)

// ---------------------------- GLOBAL TASK POOL ----------------------------
struct pending_request {
    caf::response_promise promise;
    size_t requested_size;
};

struct pool_state {
    std::vector<MatrixTask> matrix_pool;
    std::deque<MatrixTask> work_buffer;
    std::vector<pending_request> pending_requests;
    std::mt19937 rng;
    int batches_remaining;
    int batch_size;
    double mean_arrival_ms;
    bool production_finished = false;
};

behavior global_task_pool(stateful_actor<pool_state>* self, 
                          std::vector<MatrixTask> matrix_pool,
                          int num_batches, int batch_size, double mean_arrival_ms) {
    auto& st = self->state();
    st.matrix_pool = std::move(matrix_pool);
    st.batches_remaining = num_batches;
    st.batch_size = batch_size;
    st.mean_arrival_ms = mean_arrival_ms;
    st.rng.seed(WORKLOAD_SEED);

    self->mail(work_tick_atom_v).send(self);

    return {
        [=](work_tick_atom) {
            auto& st = self->state();
            if (st.batches_remaining > 0) {
                auto tasks = generate_batch(st.matrix_pool, st.rng, st.batch_size);
                for (auto& t : tasks)
                    st.work_buffer.push_back(std::move(t));
                st.batches_remaining--;

                // Satisfy pending requests from the new work
                while (!st.pending_requests.empty() && !st.work_buffer.empty()) {
                    auto req = std::move(st.pending_requests.front());
                    st.pending_requests.erase(st.pending_requests.begin());

                    size_t count = std::min(req.requested_size, st.work_buffer.size());
                    std::vector<MatrixTask> batch;
                    for (size_t i = 0; i < count; ++i) {
                        batch.push_back(std::move(st.work_buffer.front()));
                        st.work_buffer.pop_front();
                    }
                    req.promise.deliver(std::move(batch));
                }

                if (st.batches_remaining > 0) {
                    auto interval = generate_random_interval(st.rng, st.mean_arrival_ms);
                    std::cout << "[PRODUCER] going to sleep for " << interval.count() << " ms\n";
                    self->mail(work_tick_atom_v).delay(interval).send(self);
                } else {
                    st.production_finished = true;
                    // If the buffer is empty and no more batches are coming, signal EOS to anyone waiting
                    if (st.work_buffer.empty()) {
                        for (auto& req : st.pending_requests)
                            req.promise.deliver(sec::end_of_stream);
                        st.pending_requests.clear();
                    }
                }
            }
        },
        [=](get_work_atom, size_t batch_size) -> result<std::vector<MatrixTask>> {
            auto& st = self->state();
            if (!st.work_buffer.empty()) {
                size_t count = std::min(batch_size, st.work_buffer.size());
                std::vector<MatrixTask> batch;
                for (size_t i = 0; i < count; ++i) {
                    batch.push_back(std::move(st.work_buffer.front()));
                    st.work_buffer.pop_front();
                }
                return batch;
            }

            if (st.production_finished)
                return sec::end_of_stream;

            auto promise = self->make_response_promise();
            st.pending_requests.push_back({promise, batch_size});
            return promise;
        }
    };
}

// ---------------------------- WORKER ACTOR ----------------------------
struct worker_state {
    caf::actor global_pool;
    caf::actor supervisor;
    int device_id;
    int stream_id;
    std::shared_ptr<MatrixData> current_data;
    std::string current_matrix_path;
    std::chrono::steady_clock::time_point task_start;
    SolverType current_solver_type;
    caf::actor cg_facade;
};

behavior sparse_worker_fun(stateful_actor<worker_state>* self,
                           caf::actor supervisor, caf::actor global_pool, int dev_id, int stream_id) {
    auto& st = self->state();
    st.supervisor = supervisor;
    st.global_pool = global_pool;
    st.device_id = dev_id;
    st.stream_id = stream_id;
    st.cg_facade = self->spawn<sparse_cg_facade<float>, linked>(0);

    self->mail(request_work_atom_v).send(self);

    return {
        [=](request_work_atom) {
            self->mail(get_work_atom_v, size_t{1}).request(self->state().global_pool, infinite).then(
                [=](std::vector<MatrixTask>& batch) {
                    if (batch.empty()) {
                        self->mail(request_work_atom_v).send(self);
                        return;
                    }
                    auto& task = batch.front();
                    auto& data = *task.data;

                    self->state().current_matrix_path = task.path;
                    self->state().current_solver_type = task.type;
                    self->state().task_start = std::chrono::steady_clock::now();
                    self->state().current_data = task.data;

                    if (task.type == CGS_SOLVER) {
                        // Use the optimized CG facade. It responds with (r_id, index, solution, meta)
                        self->mail(create_in_arg((const std::vector<int32_t>&)data.row_ptr),
                                   create_in_arg((const std::vector<int32_t>&)data.col_indices),
                                   create_in_arg(data.values), create_in_arg(data.b),
                                   create_in_out_arg(data.x_guess), matrix_format::csr,
                                   (int)data.row_ptr.size() - 1, (int)data.values.size(),
                                   1e-5f, 2000, dev_id, stream_id).send(self->state().cg_facade);
                    } else {
                        // BiCGSTAB still uses the standard stateful actor. It responds with (solution, meta)
                        auto solver = self->spawn<sparse_bicgstab_actor<float>>(
                            create_in_arg((const std::vector<int32_t>&)data.row_ptr),
                            create_in_arg((const std::vector<int32_t>&)data.col_indices),
                            create_in_arg(data.values), create_in_arg(data.b),
                            create_in_out_arg(data.x_guess), matrix_format::csr,
                            (int)data.row_ptr.size() - 1, (int)data.values.size(),
                            1e-5f, 2000, dev_id, stream_id, actor_cast<actor>(self));
                        self->mail(start_atom_v).send(solver);
                    }
                },
                [=](error& err) {
                    if (err == sec::end_of_stream)
                        self->quit();
                }
            );
        },
        // Result handler for standard stateful actors (e.g., BiCGSTAB)
        [=](std::vector<float>& solution, solver_result_meta meta) {
            auto task_end = std::chrono::steady_clock::now();
            std::chrono::duration<double> task_duration = task_end - self->state().task_start;
            self->println("Worker {}: Round-trip time (Spawn to Result) for {} (BICSTAB_SOLVER) took {} s (Iters: {})", 
                          self->state().stream_id, self->state().current_matrix_path, task_duration.count(), meta.iterations);
            self->mail(1).send(self->state().supervisor);
            self->state().current_data.reset();
            self->mail(request_work_atom_v).send(self);
        },
        // Result handler for the optimized facade actor
        [=](uint32_t /*r_id*/, int /*index*/, std::vector<float>& solution, solver_result_meta meta) {
            auto task_end = std::chrono::steady_clock::now();
            std::chrono::duration<double> task_duration = task_end - self->state().task_start;
            self->println("Worker {}: Round-trip time (Spawn to Result) for {} (CGS_SOLVER_OPTIMIZED) took {} s (Iters: {})", 
                          self->state().stream_id, self->state().current_matrix_path, task_duration.count(), meta.iterations);
            self->mail(1).send(self->state().supervisor);
            self->state().current_data.reset();
            self->mail(request_work_atom_v).send(self);
        }
    };
}

// ---------------------------- SUPERVISOR ACTOR ----------------------------
struct supervisor_state {
    int total_tasks;
    int completed = 0;
};

behavior supervisor_actor_fun(stateful_actor<supervisor_state>* self, 
                              std::vector<MatrixTask> matrix_pool,
                              int num_streams, int num_batches, int batch_size, double mean_arrival_ms) {
    self->state().total_tasks = num_batches * batch_size;
    auto pool = self->spawn(global_task_pool, std::move(matrix_pool), num_batches, batch_size, mean_arrival_ms);
    
    manager& mgr = manager::get();
    int num_gpus = mgr.get_num_devices();

    for (int i = 0; i < num_gpus; ++i) {
        for (int j = 0; j < num_streams; ++j) {
            self->spawn(sparse_worker_fun, self, pool, i, (i * 100) + j);
        }
    }

    return {
        [=](int done) {
            self->state().completed += done;
            if (self->state().completed >= self->state().total_tasks) {
                std::cout << "\n[DONE] All " << self->state().total_tasks << " tasks processed.\n";
                self->quit();
            }
        }
    };
}

void caf_main(actor_system& sys) {
    manager::init(sys, manager_config(true, true));
    
    int num_streams = 8;
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

    sys.spawn(supervisor_actor_fun, std::move(tasks_vec), num_streams, num_batches, batch_size, mean_arrival_ms);
    sys.await_all_actors_done();

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