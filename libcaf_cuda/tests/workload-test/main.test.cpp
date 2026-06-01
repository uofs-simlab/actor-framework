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

enum SolverType { CGS_SOLVER, BICSTAB_SOLVER };

struct MatrixData {
    std::vector<int> row_ptr;
    std::vector<int> col_indices;
    std::vector<float> values;
    std::vector<float> b;
    std::vector<float> x_guess;
};

struct MatrixTask {
    std::string path;
    SolverType type;
    std::shared_ptr<MatrixData> data;
};

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
struct pool_state {
    std::shared_ptr<std::vector<MatrixTask>> tasks;
    size_t next_task_idx = 0;
};

behavior global_task_pool(stateful_actor<pool_state>* self, std::shared_ptr<std::vector<MatrixTask>> tasks) {
    self->state().tasks = std::move(tasks);
    return {
        [=](get_work_atom, size_t batch_size) -> result<std::vector<MatrixTask>> {
            auto& st = self->state();
            if (st.next_task_idx >= st.tasks->size())
                return sec::end_of_stream;
            size_t count = std::min(batch_size, st.tasks->size() - st.next_task_idx);
            std::vector<MatrixTask> batch(st.tasks->begin() + st.next_task_idx, 
                                          st.tasks->begin() + st.next_task_idx + count);
            st.next_task_idx += count;
            return batch;
        }
    };
}

// ---------------------------- DEVICE/GPU ACTOR ----------------------------
struct device_actor_state {
    caf::actor global_pool;
    std::deque<MatrixTask> local_tasks;
    std::vector<caf::response_promise> pending_promises; // Store untyped promises cleanly
    int active_workers = 0;
    int device_id = -1;
    bool fetching = false;
    size_t low_water_mark;
    size_t batch_size;
};

behavior gpu_device_actor(stateful_actor<device_actor_state>* self,
                          caf::actor global_pool, int num_workers, int dev_id, int max_in_flight) {
    self->state().global_pool = global_pool;
    self->state().device_id = dev_id;
    self->state().active_workers = num_workers;

    self->state().low_water_mark = static_cast<size_t>(num_workers * max_in_flight);
    self->state().batch_size = self->state().low_water_mark * 2;

    auto satisfy_promises = [=]() {
        auto& st = self->state();
        while (!st.pending_promises.empty() && !st.local_tasks.empty()) {
            auto promise = std::move(st.pending_promises.front());
            st.pending_promises.erase(st.pending_promises.begin());

            MatrixTask t = std::move(st.local_tasks.front());
            st.local_tasks.pop_front();

            auto& data = *t.data;
            promise.deliver(t.type, t.path, create_in_arg(data.row_ptr), create_in_arg(data.col_indices), 
                            create_in_arg(data.values), create_in_arg(data.b), create_in_out_arg(data.x_guess), 
                            (int)data.row_ptr.size() - 1, (int)data.values.size(), t.data);
        }
    };

    auto refill = [=]() {
        auto& st = self->state();
        if (st.fetching)
            return;

        if (st.local_tasks.size() >= st.low_water_mark && st.pending_promises.empty())
            return;

        st.fetching = true;
        self->mail(get_work_atom_v, st.batch_size).request(st.global_pool, infinite).then(
            [=](std::vector<MatrixTask>& batch) {
                auto& st_inner = self->state();
                st_inner.fetching = false;

                for (auto& task : batch)
                    st_inner.local_tasks.push_back(std::move(task));

                satisfy_promises();
            },
            [=](error& err) {
                auto& st_inner = self->state();
                st_inner.fetching = false;
                
                for (auto& promise : st_inner.pending_promises) {
                    promise.deliver(err);
                }
                st_inner.pending_promises.clear();
            }
        );
    };

    refill();

    return {
        [=](get_work_atom) -> result<SolverType, std::string, in<int>, in<int>, in<float>, in<float>, in_out<float>, int, int, std::shared_ptr<MatrixData>> {
            auto& st = self->state();
            
            // Fixed Type Mismatch: Explicitly use untyped response_promise
            caf::response_promise promise = self->make_response_promise();
            st.pending_promises.push_back(promise);
            
            satisfy_promises();
            refill();

            return promise;
        },
        [=](release_memory_atom, std::string path) {
            // Managed entirely by shared_ptrs
        },
        [=](worker_done_atom) {
            if (--self->state().active_workers <= 0)
                self->quit();
        }
    };
}

// ---------------------------- WORKER ACTOR ----------------------------
struct worker_state {
    caf::actor device_actor;
    caf::actor supervisor;
    int device_id;
    int stream_id;
    std::shared_ptr<MatrixData> current_data;
    std::string current_matrix_path;
    std::chrono::steady_clock::time_point task_start;
    SolverType current_solver_type;
};

behavior sparse_worker_fun(stateful_actor<worker_state>* self,
                           caf::actor supervisor, caf::actor device_actor, int dev_id, int stream_id) {
    self->state().supervisor = supervisor;
    self->state().device_actor = device_actor;
    self->state().device_id = dev_id;
    self->state().stream_id = stream_id;

    self->mail(request_work_atom_v).send(self);

    return {
        [=](request_work_atom) {
            self->mail(get_work_atom_v).request(self->state().device_actor, infinite).then(
                [=](SolverType type, std::string path, in<int> rp, in<int> ci, in<float> val,
                    in<float> b, in_out<float> x, int rows, int nnz, std::shared_ptr<MatrixData> data) {
                    self->state().current_matrix_path = path;
                    self->state().current_solver_type = type;
                    self->state().task_start = std::chrono::steady_clock::now();
                    auto start_spawn = std::chrono::steady_clock::now();
                    self->state().current_data = data;
                    if (type == CGS_SOLVER) {
                        // Use the optimized CG facade. It responds with (r_id, index, solution, meta)
                        auto facade = self->spawn<sparse_cg_facade<float>>(0);
                        self->mail(std::move(rp), std::move(ci), std::move(val),
                                   std::move(b), std::move(x),
                                   matrix_format::csr, rows, nnz, 1e-5f, 2000, dev_id, stream_id).send(facade);
                    } else {
                        // BiCGSTAB still uses the standard stateful actor. It responds with (solution, meta)
                        auto solver = self->spawn<sparse_bicgstab_actor<float>>(
                            std::move(rp), std::move(ci), std::move(val),
                            std::move(b), std::move(x),
                            matrix_format::csr, rows, nnz, 1e-5f, 2000, dev_id, stream_id, actor_cast<actor>(self));
                        self->mail(start_atom_v).send(solver);
                    }
                },
                [=](error& err) {
                    if (err == sec::end_of_stream) {
                        self->mail(worker_done_atom_v).send(self->state().device_actor);
                        self->quit();
                    }
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
    std::shared_ptr<std::vector<MatrixTask>> tasks_holder; // Anchors shared_ptr reference count
};

behavior supervisor_actor_fun(stateful_actor<supervisor_state>* self, int total, std::shared_ptr<std::vector<MatrixTask>> tasks) {
    self->state().total_tasks = total;
    self->state().tasks_holder = tasks; // Retain ownership within supervisor state
    auto pool = self->spawn(global_task_pool, tasks); // Pass a copy, don't move it
    
    manager& mgr = manager::get();
    int num_gpus = mgr.get_num_devices();
    int workers_per_gpu = 8;
    int max_in_flight_tasks_per_worker = 1;

    for (int i = 0; i < num_gpus; ++i) {
        auto broker = self->spawn(gpu_device_actor, pool, workers_per_gpu, i, max_in_flight_tasks_per_worker);
        for (int j = 0; j < workers_per_gpu; ++j) {
            self->spawn(sparse_worker_fun, self, broker, i, (i * 100) + j);
        }
    }

    return {
        [=](int done) {
            self->state().completed += done;
            if (self->state().completed >= self->state().total_tasks) {
                std::cout << "\n[DONE] All " << self->state().total_tasks << " matrices processed.\n";
                self->quit();
            }
        }
    };
}

void caf_main(actor_system& sys) {
    manager::init(sys, manager_config(true, true));
    auto tasks = std::make_shared<std::vector<MatrixTask>>();
    
    auto scan = [&](const std::string& dir, SolverType type) {
        if (!fs::exists(dir)) return;
        for (const auto& entry : fs::directory_iterator(dir)) {
            if (entry.path().extension() == ".bin")
                tasks->push_back({entry.path().string(), type, nullptr});
        }
    };

    scan("/scratch/nqr159/matrix-collection/matrix_corpus_v2/matrices/spd", CGS_SOLVER);
    //scan("/scratch/nqr159/matrix-collection/matrices/unsymmetric", BICSTAB_SOLVER);

    std::cout << "[INFO] Pre-loading " << tasks->size() << " matrices into memory...\n";
    for (auto& t : *tasks) {
        auto coo = load_binary_coo(t.path);
        auto A = convert_coo_to_csr(coo);
        t.data = std::make_shared<MatrixData>();
        t.data->b = compute_rhs_spmv(A, std::vector<float>(A.cols, 1.0f));
        t.data->row_ptr = std::move(A.row_ptr);
        t.data->col_indices = std::move(A.col_indices);
        t.data->values = std::move(A.values);
        t.data->x_guess.assign(A.cols, 0.0f);
    }

    if (tasks->empty()) {
        std::cerr << "No matrix files found in search paths.\n";
        manager::shutdown();
        return;
    }

    auto task_count = tasks->size();
    std::cout << "[INFO] Found " << task_count << " matrices. Spawning workload...\n";

    auto start = std::chrono::steady_clock::now();

    // Fixed Exit Race: Pass tasks directly by copy to keep it active inside caf_main frame
    sys.spawn(supervisor_actor_fun, static_cast<int>(task_count), tasks);
    sys.await_all_actors_done();

    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    std::cout << "\n===== BENCHMARK COMPLETE =====\n";
    std::cout << "Tasks Processed: " << task_count << "\n";
    std::cout << "Total Runtime:   " << elapsed.count() << " s\n";
    std::cout << "==============================\n";

    manager::shutdown();
}
CAF_MAIN(id_block::cuda,id_block::workload_test)